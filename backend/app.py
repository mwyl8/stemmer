"""FastAPI: endpoints only — wires modules, no heavy logic. Thin switchboard.

Nothing in here does fetch/decode/separate/encode work itself: POST /jobs
does the minimal synchronous work an HTTP handler can't defer (stream an
upload to disk, or a cheap structural URL check) and hands the job straight
to the worker pool. All heavy lifting happens in pool.py's background
workers — the request handler never blocks on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from backend import jobs
from backend.config import DEFAULT_MODE, DEFAULT_STEM_COUNT, DEFAULT_TIER, MAX_UPLOAD_BYTES, MODES, STEM_COUNTS, TIERS
from backend.ingest.fetch import InvalidURLError, classify_platform, validate_url
from backend.mixdown import TrackAdjustment, mix_stems
from backend.pool import WorkerPool
from backend.storage import init_db, job_dir

_UPLOAD_CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    pool = WorkerPool()
    await pool.start()
    app.state.pool = pool
    purge_task = asyncio.create_task(jobs.run_purge_loop())
    try:
        yield
    finally:
        purge_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await purge_task
        await pool.shutdown()


app = FastAPI(title="Stemmer", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs", status_code=201)
async def submit_job(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        job_id = await _submit_file_job(request)
    else:
        job_id = await _submit_url_job(request)

    await request.app.state.pool.submit(job_id)
    return {"job_id": job_id}


async def _submit_file_job(request: Request) -> str:
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(422, "multipart submission requires a 'file' field")
    mode, tier, stem_count = _validate_mode_tier_stems(form.get("mode"), form.get("tier"), form.get("stem_count"))

    upload_dir = Path(tempfile.mkdtemp(prefix="stemmer-upload-"))
    dest = upload_dir / (upload.filename or "upload")
    size = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES} byte cap")
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    finally:
        await upload.close()

    return jobs.create_job(mode, tier, "upload", str(dest), stem_count=stem_count)


async def _submit_url_job(request: Request) -> str:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(422, "expected a JSON body with a 'url' field") from exc

    url = body.get("url")
    if not url:
        raise HTTPException(422, "JSON submission requires a 'url' field")
    mode, tier, stem_count = _validate_mode_tier_stems(body.get("mode"), body.get("tier"), body.get("stem_count"))

    try:
        validate_url(url)
        source_type = classify_platform(url)
    except InvalidURLError as exc:
        raise HTTPException(400, str(exc)) from exc

    return jobs.create_job(mode, tier, source_type, url, stem_count=stem_count)


def _validate_mode_tier_stems(mode: str | None, tier: str | None, stem_count: str | int | None) -> tuple[str, str, int]:
    mode = mode or DEFAULT_MODE
    tier = tier or DEFAULT_TIER
    stem_count = DEFAULT_STEM_COUNT if stem_count is None else stem_count
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    if tier not in TIERS:
        raise HTTPException(422, f"tier must be one of {tuple(TIERS)}")
    try:
        stem_count = int(stem_count)
    except (TypeError, ValueError):
        raise HTTPException(422, f"stem_count must be one of {STEM_COUNTS}") from None
    if stem_count not in STEM_COUNTS:
        raise HTTPException(422, f"stem_count must be one of {STEM_COUNTS}")
    return mode, tier, stem_count


def _job_summary(job: jobs.Job) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress_pct": job.progress_pct,
        "mode": job.mode,
        "tier": job.tier,
        "stem_count": job.stem_count,
        "error": job.error,
        "created_at": job.created_at,
        "expires_at": job.expires_at,
        "submitted_at": job.submitted_at,
        "stage_started_at": job.stage_started_at,
        "stage_timings": job.stage_timings,
        "chunks_done": job.chunks_done,
        "chunks_total": job.chunks_total,
        "elapsed_seconds": job.elapsed_seconds,
        "eta_seconds": job.eta_seconds,
        "from_cache": job.from_cache,
        "source_codec": job.source_codec,
        "source_bitrate": job.source_bitrate,
        "source_sample_rate": job.source_sample_rate,
        "source_channels": job.source_channels,
        # Whether this job's normalized source audio is still persisted
        # (job_dir, same TTL as its stems) — gates the "A/B against the
        # original" toggle and the "re-run at a different mode/tier" action
        # in the UI (PRD Addendum §2.3, §2.5). Checked on disk, not just the
        # DB column: a purged job keeps its source_wav_path string around
        # (only the jobs.status flips to 'expired' and the files are
        # deleted), so the column alone can't tell "persisted" from "purged".
        "has_original": bool(job.source_wav_path) and Path(job.source_wav_path).exists(),
        "stems": [{"name": s.name, "format": s.format, "duration": s.duration} for s in job.stems],
    }


def _get_job_or_404(job_id: str) -> jobs.Job:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job_summary(_get_job_or_404(job_id))


@app.get("/jobs/{job_id}/stems")
def list_stems(job_id: str) -> list[dict]:
    job = _get_job_or_404(job_id)
    return [{"name": s.name, "format": s.format, "duration": s.duration} for s in job.stems]


@app.get("/jobs/{job_id}/stems/{name}")
def get_stem(job_id: str, name: str, format: str = "mp3") -> FileResponse:
    if format not in ("wav", "mp3"):
        raise HTTPException(422, "format must be 'wav' or 'mp3'")
    job = _get_job_or_404(job_id)
    stem = next((s for s in job.stems if s.name == name and s.format == format), None)
    if stem is None:
        raise HTTPException(404, f"no {format} stem named {name!r} for this job")
    media_type = "audio/wav" if format == "wav" else "audio/mpeg"
    return FileResponse(stem.path, media_type=media_type, filename=f"{name}.{format}")


@app.get("/jobs/{job_id}/stems/{name}/peaks")
def get_stem_peaks(job_id: str, name: str) -> list:
    """Precomputed waveform peaks (peaks.py) for one wav stem — lets the
    player skip decoding the full mp3 client-side (PRD Addendum §2.5)."""
    job = _get_job_or_404(job_id)
    stem = next((s for s in job.stems if s.name == name and s.format == "wav"), None)
    if stem is None:
        raise HTTPException(404, f"no stem named {name!r} for this job")
    peaks_path = job_dir(job_id) / f"{name}.peaks.json"
    if not peaks_path.exists():
        raise HTTPException(404, f"no precomputed peaks for stem {name!r}")
    return json.loads(peaks_path.read_text())


@app.get("/jobs/{job_id}/original")
def get_original(job_id: str) -> FileResponse:
    """The source mixture, as an mp3 preview — for the A/B toggle (PRD
    Addendum §2.3). Persisted alongside the stems by pool.py's
    `_persist_source_audio`; 404s once the job (and its job_dir) expires."""
    job = _get_job_or_404(job_id)
    if not job.source_wav_path:
        raise HTTPException(404, "no original audio persisted for this job")
    mp3_path = job_dir(job_id) / "source.mp3"
    if not mp3_path.exists():
        raise HTTPException(404, "original audio preview is missing")
    return FileResponse(mp3_path, media_type="audio/mpeg", filename="original.mp3")


@app.post("/jobs/{job_id}/rerun", status_code=201)
async def rerun_job(job_id: str, request: Request) -> dict:
    """Re-run this job's audio at a different mode/tier/stem_count without
    re-uploading or re-fetching (PRD Addendum §2.5) — reuses the persisted
    source WAV (pool.py's `_reuse_source_audio`) instead of the original
    upload/URL, which may no longer exist by now."""
    original = _get_job_or_404(job_id)
    if not original.source_wav_path or not Path(original.source_wav_path).exists():
        raise HTTPException(409, "original audio for this job has expired — please re-upload or re-submit the link")

    try:
        body = await request.json()
    except Exception:
        body = {}
    mode, tier, stem_count = _validate_mode_tier_stems(
        body.get("mode") or original.mode,
        body.get("tier") or original.tier,
        original.stem_count if body.get("stem_count") is None else body.get("stem_count"),
    )

    new_job_id = jobs.create_job(mode, tier, "rerun", job_id, stem_count=stem_count)
    await request.app.state.pool.submit(new_job_id)
    return {"job_id": new_job_id}


@app.post("/jobs/{job_id}/export-mix")
async def export_mix(job_id: str, request: Request) -> FileResponse:
    """Mix down the full-quality wav stems with the given per-stem
    volume/pan/mute/solo adjustments and hand back one downloadable wav (PRD
    Addendum §2.3 "export a custom mix") — see mixdown.py for the math."""
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"job is not done yet (status={job.status})")
    wav_stems = {s.name: s.path for s in job.stems if s.format == "wav"}
    if not wav_stems:
        raise HTTPException(404, "no stems available for this job")

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(422, "expected a JSON body") from exc

    adjustments = [
        TrackAdjustment(
            name=t["name"],
            volume=float(t.get("volume", 1.0)),
            pan=float(t.get("pan", 0.0)),
            muted=bool(t.get("muted", False)),
            solo=bool(t.get("solo", False)),
        )
        for t in body.get("tracks") or []
        if t.get("name") in wav_stems
    ]
    master_volume = float(body.get("master_volume", 1.0))
    master_muted = bool(body.get("master_muted", False))

    try:
        mixed = mix_stems(wav_stems, adjustments, master_volume, master_muted)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="stemmer-mix-")
    os.close(fd)
    tmp_path = Path(tmp_name)
    sf.write(tmp_path, mixed.audio.T, mixed.sample_rate)

    return FileResponse(
        tmp_path,
        media_type="audio/wav",
        filename=f"{job_id}-mix.wav",
        background=BackgroundTask(lambda: tmp_path.unlink(missing_ok=True)),
    )


@app.get("/jobs/{job_id}/download")
def download_all(job_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"job is not done yet (status={job.status})")
    wav_stems = [s for s in job.stems if s.format == "wav"]
    if not wav_stems:
        raise HTTPException(404, "no stems available for this job")

    fd, zip_name = tempfile.mkstemp(suffix=".zip", prefix="stemmer-download-")
    os.close(fd)  # mkstemp hands back an open fd; ZipFile below reopens by path
    zip_path = Path(zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem in wav_stems:
            zf.write(stem.path, arcname=f"{stem.name}.wav")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{job_id}.zip",
        background=BackgroundTask(lambda: zip_path.unlink(missing_ok=True)),
    )


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    if not jobs.purge_job(job_id):
        raise HTTPException(404, "job not found")
    return Response(status_code=204)
