"""Async worker pool, capped at config.POOL_SIZE, dispatching to the isolated
subprocess runner (Phase 1). The web process (app.py) never blocks on
separation: `submit()` only enqueues a job id onto a shared asyncio.Queue;
exactly `size` worker tasks pull from it, so at most `size` separations run
concurrently and anything beyond that simply waits in the queue instead of
piling load onto the box.

`process_job()` is the full per-job pipeline (ingest -> cache check ->
separate -> encode) and is deliberately a plain blocking function, run via
`asyncio.to_thread` — everything it calls (yt-dlp, ffmpeg, the ONNX
subprocess runner) is itself a blocking subprocess call, so it must never run
directly on the event loop thread.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

import soundfile as sf

from backend import cache, jobs
from backend import runner as runner_mod
from backend.config import (
    ENCODE_TIMEOUT_SECONDS,
    MP3_BITRATE,
    PIPELINE_VERSION,
    POOL_SIZE,
    SEPARATE_TIMEOUT_SECONDS,
    TIER_RTF,
)
from backend.ingest import IngestError
from backend.ingest import ingest as ingest_fn
from backend.ingest._sandbox import run_subprocess
from backend.ingest.decode import DecodeError
from backend.ingest.fetch import FetchError, InvalidURLError
from backend.runner import SeparationFailed, SeparationTimeout
from backend.storage import job_dir

logger = logging.getLogger(__name__)

# Failures with a clear, user-facing cause — recorded on the job row as-is.
# Anything else is an unexpected bug and gets logged with a traceback too.
_KNOWN_FAILURES = (InvalidURLError, FetchError, DecodeError, IngestError, SeparationTimeout, SeparationFailed)


class WorkerPool:
    def __init__(self, size: int = POOL_SIZE) -> None:
        self.size = size
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [asyncio.create_task(self._worker_loop(i)) for i in range(self.size)]

    async def shutdown(self) -> None:
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def submit(self, job_id: str) -> None:
        await self._queue.put(job_id)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def _worker_loop(self, worker_index: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await asyncio.to_thread(process_job, job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker %d: job %s crashed outside process_job's own handling", worker_index, job_id)
            finally:
                self._queue.task_done()


def process_job(job_id: str) -> None:
    """ingest (or reuse a prior job's persisted audio, for a re-run) -> cache
    check -> (copy cached stems | separate + encode).

    Synchronous/blocking by design (see module docstring). Any failure is
    caught here and written to the job's `error` field rather than raised,
    so one bad job never wedges its worker task.
    """
    job = jobs.get_job(job_id)
    if job is None:
        logger.error("process_job: job %s not found", job_id)
        return

    wav_dir_to_clean: Path | None = None
    try:
        if job.source_type == "rerun":
            jobs.update_stage(job_id, jobs.Stage.DECODING)
            wav_path, content_hash = _reuse_source_audio(job_id, job.source_ref)
        else:
            if job.source_type == "upload":
                jobs.update_stage(job_id, jobs.Stage.DECODING)
                result = ingest_fn(file=job.source_ref)
            else:
                jobs.update_stage(job_id, jobs.Stage.DOWNLOADING)
                result = ingest_fn(url=job.source_ref)

            wav_dir_to_clean = result.wav_path.parent
            wav_path, content_hash = result.wav_path, result.content_hash
            jobs.set_content_hash(job_id, content_hash)
            jobs.set_source_info(
                job_id, result.source_codec, result.source_bitrate, result.source_sample_rate, result.source_channels
            )
            _persist_source_audio(job_id, wav_path)

        cached_job_id = cache.find(content_hash, job.mode, job.tier, PIPELINE_VERSION)
        if cached_job_id is not None:
            jobs.mark_from_cache(job_id)
            _copy_cached_stems(cached_job_id, job_id)
            jobs.update_stage(job_id, jobs.Stage.DONE)
            return

        jobs.update_stage(job_id, jobs.Stage.SEPARATING)
        audio_duration = sf.info(wav_path).duration
        jobs.set_initial_eta(job_id, audio_duration * TIER_RTF.get(job.tier, TIER_RTF["balanced"]))
        output_dir = job_dir(job_id)
        stems = runner_mod.separate_in_subprocess(
            wav_path,
            output_dir,
            mode=job.mode,
            tier=job.tier,
            stem_count=job.stem_count,
            timeout=SEPARATE_TIMEOUT_SECONDS,
            job_id=job_id,
        )

        jobs.update_stage(job_id, jobs.Stage.ENCODING)
        for name, wav_stem_path in stems.items():
            duration = sf.info(wav_stem_path).duration
            jobs.add_stem(job_id, name, "wav", str(wav_stem_path), duration)
            mp3_path = wav_stem_path.with_suffix(".mp3")
            _encode_mp3_preview(wav_stem_path, mp3_path)
            jobs.add_stem(job_id, name, "mp3", str(mp3_path), duration)

        jobs.update_stage(job_id, jobs.Stage.DONE)
        cache.put(content_hash, job.mode, job.tier, PIPELINE_VERSION, job_id)

    except _KNOWN_FAILURES as exc:
        jobs.mark_error(job_id, str(exc))
    except Exception as exc:
        logger.exception("job %s failed unexpectedly", job_id)
        jobs.mark_error(job_id, f"internal error: {exc}")
    finally:
        if wav_dir_to_clean is not None:
            shutil.rmtree(wav_dir_to_clean, ignore_errors=True)
        if job.source_type == "upload" and job.source_ref:
            shutil.rmtree(Path(job.source_ref).parent, ignore_errors=True)


def _persist_source_audio(job_id: str, wav_path: Path) -> None:
    """Copies the normalized source WAV (plus an mp3 preview, for the
    in-browser A/B toggle) into this job's own job_dir — same directory,
    same TTL as its stems — so it survives past ingest's own cleanup
    (`wav_dir_to_clean` above) for as long as the job itself is kept around.
    Enables "re-run at a different mode/tier" (no re-upload/re-fetch needed)
    and "A/B against the original" (PRD Addendum §2.3, §2.5) without adding
    any persistence beyond what job_dir/TTL already provide."""
    output_dir = job_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_wav_path = output_dir / "source.wav"
    shutil.copyfile(wav_path, source_wav_path)
    _encode_mp3_preview(source_wav_path, output_dir / "source.mp3")
    jobs.set_source_wav_path(job_id, str(source_wav_path))


def _reuse_source_audio(job_id: str, original_job_id: str | None) -> tuple[Path, str]:
    """Re-run path: instead of re-fetching/re-decoding, reuse a prior job's
    already-persisted source WAV + content hash directly, copying it into
    this job's own job_dir (so its lifetime doesn't depend on the original
    job surviving). Raises IngestError (a known, user-facing failure) if the
    original has since expired — its job_dir would already be gone."""
    original = jobs.get_job(original_job_id) if original_job_id else None
    if original is None or not original.source_wav_path or not Path(original.source_wav_path).exists():
        raise IngestError("the original audio for this job has expired — please re-upload or re-submit the link")

    jobs.set_source_info(
        job_id, original.source_codec, original.source_bitrate, original.source_sample_rate, original.source_channels
    )
    _persist_source_audio(job_id, Path(original.source_wav_path))
    jobs.set_content_hash(job_id, original.content_hash)
    return job_dir(job_id) / "source.wav", original.content_hash


def _copy_cached_stems(cached_job_id: str, new_job_id: str) -> None:
    """Copy a previously-separated job's stem files into the new job's own
    directory rather than just pointing at the same paths, so each job's
    files/TTL stay independent — the cached job could be purged later
    without breaking this one. Precomputed peaks (peaks.py) ride along too —
    a cache-hit job never runs runner.py itself, which is where those are
    normally written."""
    cached_job = jobs.get_job(cached_job_id)
    if cached_job is None:
        raise RuntimeError(f"cache pointed at missing job {cached_job_id}")
    output_dir = job_dir(new_job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in cached_job.stems:
        src = Path(stem.path)
        dst = output_dir / src.name
        shutil.copyfile(src, dst)
        jobs.add_stem(new_job_id, stem.name, stem.format, str(dst), stem.duration)
        if stem.format == "wav":
            peaks_src = job_dir(cached_job_id) / f"{stem.name}.peaks.json"
            if peaks_src.exists():
                shutil.copyfile(peaks_src, output_dir / f"{stem.name}.peaks.json")


def _encode_mp3_preview(wav_path: Path, mp3_path: Path, timeout: float = ENCODE_TIMEOUT_SECONDS) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [ffmpeg, "-nostdin", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE, str(mp3_path)]
    proc = run_subprocess(cmd, timeout=timeout)
    if proc.returncode != 0 or not mp3_path.exists():
        raise RuntimeError(f"ffmpeg mp3 encode failed (exit {proc.returncode}):\n{proc.stderr.strip()}")
