"""Pool cap enforcement and the content-hash cache short-circuiting
re-separation. The separator itself is always mocked — these tests exercise
pool.py's own logic (concurrency, caching, cleanup), not the ONNX model.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest
import soundfile as sf

from backend import jobs, pool


def _fake_separate_factory(audio_data, sr):
    """Writes 4 fake stem wavs and returns the manifest dict, matching
    runner.separate_in_subprocess's real contract."""

    def _fake(input_wav, output_dir, mode="music", tier="balanced", stem_count=4, timeout=300, job_id=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for name in ("vocals", "drums", "bass", "other"):
            p = output_dir / f"{name}.wav"
            sf.write(p, audio_data * 0.5, sr)
            out[name] = p
        if job_id is not None:
            jobs.update_progress(job_id, chunks_done=1, chunks_total=1)
        return out

    return _fake


def test_process_job_happy_path_writes_wav_and_mp3_stems(db, make_upload, monkeypatch):
    import numpy as np

    sr = 44100
    data = (0.05 * np.random.default_rng(1).standard_normal((sr, 2))).astype(np.float32)
    monkeypatch.setattr(pool.runner_mod, "separate_in_subprocess", _fake_separate_factory(data, sr))

    upload_path = make_upload(seconds=1.0, seed=1)
    job_id = jobs.create_job("music", "balanced", "upload", str(upload_path))
    pool.process_job(job_id)

    job = jobs.get_job(job_id)
    assert job.status == "done"
    assert job.error is None
    formats = {(s.name, s.format) for s in job.stems}
    assert formats == {
        ("vocals", "wav"), ("vocals", "mp3"),
        ("drums", "wav"), ("drums", "mp3"),
        ("bass", "wav"), ("bass", "mp3"),
        ("other", "wav"), ("other", "mp3"),
    }
    for stem in job.stems:
        assert Path(stem.path).exists()
    # the temp upload dir is cleaned up once ingest is done with it
    assert not upload_path.parent.exists()


def test_process_job_records_failure_on_job(db, monkeypatch):
    def _boom(*args, **kwargs):
        raise pool.SeparationFailed("model exploded")

    monkeypatch.setattr(pool.runner_mod, "separate_in_subprocess", _boom)
    job_id = jobs.create_job("music", "balanced", "upload", "/no/such/file.wav")
    pool.process_job(job_id)

    job = jobs.get_job(job_id)
    assert job.status == "error"
    assert job.error  # ingest() itself fails first (missing file) — still recorded


def test_process_job_cache_hit_skips_separation(db, make_upload, monkeypatch):
    import numpy as np

    sr = 44100
    data = (0.05 * np.random.default_rng(2).standard_normal((sr, 2))).astype(np.float32)
    call_count = {"n": 0}
    fake = _fake_separate_factory(data, sr)

    def counting_fake(*args, **kwargs):
        call_count["n"] += 1
        return fake(*args, **kwargs)

    monkeypatch.setattr(pool.runner_mod, "separate_in_subprocess", counting_fake)

    upload1 = make_upload(seconds=1.0, seed=42)
    job1 = jobs.create_job("music", "balanced", "upload", str(upload1))
    pool.process_job(job1)
    assert jobs.get_job(job1).status == "done"
    assert call_count["n"] == 1

    # identical audio content, submitted again
    upload2 = make_upload(seconds=1.0, seed=42)
    job2 = jobs.create_job("music", "balanced", "upload", str(upload2))
    pool.process_job(job2)

    job2_result = jobs.get_job(job2)
    assert job2_result.status == "done"
    assert call_count["n"] == 1  # separation was NOT run again

    # job2 has its own independent stem files, not shared paths with job1
    job1_result = jobs.get_job(job1)
    paths1 = {(s.name, s.format): s.path for s in job1_result.stems}
    paths2 = {(s.name, s.format): s.path for s in job2_result.stems}
    assert paths1.keys() == paths2.keys()
    assert all(paths1[k] != paths2[k] for k in paths1)
    assert all(Path(p).exists() for p in paths2.values())


def test_process_job_cache_miss_on_different_tier(db, make_upload, monkeypatch):
    import numpy as np

    sr = 44100
    data = (0.05 * np.random.default_rng(3).standard_normal((sr, 2))).astype(np.float32)
    call_count = {"n": 0}
    fake = _fake_separate_factory(data, sr)

    def counting_fake(*args, **kwargs):
        call_count["n"] += 1
        return fake(*args, **kwargs)

    monkeypatch.setattr(pool.runner_mod, "separate_in_subprocess", counting_fake)

    upload1 = make_upload(seconds=1.0, seed=7)
    job1 = jobs.create_job("music", "balanced", "upload", str(upload1))
    pool.process_job(job1)

    upload2 = make_upload(seconds=1.0, seed=7)  # same audio, different tier
    job2 = jobs.create_job("music", "fast", "upload", str(upload2))
    pool.process_job(job2)

    assert call_count["n"] == 2  # different tier -> cache miss -> separated again


# ---------------------------------------------------------------------------
# Pool cap enforcement
# ---------------------------------------------------------------------------


def test_worker_pool_caps_concurrency(db, monkeypatch):
    current = 0
    max_seen = 0
    state_lock = threading.Lock()

    def blocking_job(job_id):
        nonlocal current, max_seen
        with state_lock:
            current += 1
            max_seen = max(max_seen, current)
        time.sleep(0.2)
        with state_lock:
            current -= 1

    monkeypatch.setattr(pool, "process_job", blocking_job)

    async def run():
        worker_pool = pool.WorkerPool(size=2)
        await worker_pool.start()
        job_ids = [jobs.create_job("music", "fast", "upload", f"job-{i}.wav") for i in range(6)]
        for job_id in job_ids:
            await worker_pool.submit(job_id)
        await worker_pool._queue.join()
        await worker_pool.shutdown()

    asyncio.run(run())
    assert max_seen == 2


@pytest.mark.parametrize("pool_size", [1, 3])
def test_worker_pool_respects_configured_size(db, monkeypatch, pool_size):
    current = 0
    max_seen = 0
    state_lock = threading.Lock()

    def blocking_job(job_id):
        nonlocal current, max_seen
        with state_lock:
            current += 1
            max_seen = max(max_seen, current)
        time.sleep(0.15)
        with state_lock:
            current -= 1

    monkeypatch.setattr(pool, "process_job", blocking_job)

    async def run():
        worker_pool = pool.WorkerPool(size=pool_size)
        await worker_pool.start()
        job_ids = [jobs.create_job("music", "fast", "upload", f"job-{i}.wav") for i in range(pool_size * 3)]
        for job_id in job_ids:
            await worker_pool.submit(job_id)
        await worker_pool._queue.join()
        await worker_pool.shutdown()

    asyncio.run(run())
    assert max_seen == pool_size


# ---------------------------------------------------------------------------
# Phase 7b Part B: persisted source audio, re-run, and peaks copy-on-cache-hit.
# ---------------------------------------------------------------------------


def _fake_separate_with_peaks_factory(audio_data, sr):
    """Like _fake_separate_factory, but also drops a {name}.peaks.json next
    to each stem — matching what runner.py's real _main() does — so
    cache-hit copying of peaks files (pool.py's _copy_cached_stems) has
    something to actually copy."""

    def _fake(input_wav, output_dir, mode="music", tier="balanced", stem_count=4, timeout=300, job_id=None):
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for name in ("vocals", "drums", "bass", "other"):
            p = output_dir / f"{name}.wav"
            sf.write(p, audio_data * 0.5, sr)
            (output_dir / f"{name}.peaks.json").write_text(json.dumps([[0.1, -0.1]]))
            out[name] = p
        if job_id is not None:
            jobs.update_progress(job_id, chunks_done=1, chunks_total=1)
        return out

    return _fake


def test_process_job_persists_source_audio_for_rerun_and_ab(db, make_upload, monkeypatch):
    import numpy as np

    sr = 44100
    data = (0.05 * np.random.default_rng(50).standard_normal((sr, 2))).astype(np.float32)
    monkeypatch.setattr(pool.runner_mod, "separate_in_subprocess", _fake_separate_factory(data, sr))

    upload_path = make_upload(seconds=1.0, seed=50)
    job_id = jobs.create_job("music", "balanced", "upload", str(upload_path))
    pool.process_job(job_id)

    job = jobs.get_job(job_id)
    assert job.status == "done"
    assert job.source_wav_path is not None
    assert Path(job.source_wav_path).exists()
    assert (Path(job.source_wav_path).parent / "source.mp3").exists()


def test_rerun_reuses_persisted_source_without_calling_ingest(db, make_upload, monkeypatch):
    import numpy as np

    sr = 44100
    data = (0.05 * np.random.default_rng(51).standard_normal((sr, 2))).astype(np.float32)
    monkeypatch.setattr(pool.runner_mod, "separate_in_subprocess", _fake_separate_factory(data, sr))

    def _boom_if_called(*args, **kwargs):
        raise AssertionError("rerun must not call ingest() -- it should reuse the persisted source wav")

    upload_path = make_upload(seconds=1.0, seed=51)
    original_id = jobs.create_job("music", "fast", "upload", str(upload_path))
    pool.process_job(original_id)
    assert jobs.get_job(original_id).status == "done"

    # only patch ingest_fn AFTER the original job's own ingest has already run
    monkeypatch.setattr(pool, "ingest_fn", _boom_if_called)

    rerun_id = jobs.create_job("music", "balanced", "rerun", original_id)
    pool.process_job(rerun_id)

    rerun_job = jobs.get_job(rerun_id)
    assert rerun_job.status == "done"
    assert rerun_job.tier == "balanced"
    assert rerun_job.content_hash == jobs.get_job(original_id).content_hash
    assert len(rerun_job.stems) == 8


def test_rerun_fails_cleanly_once_original_source_is_gone(db):
    original_id = jobs.create_job("music", "fast", "upload", "/no/such/file.wav")
    # never processed -> source_wav_path was never set

    rerun_id = jobs.create_job("music", "balanced", "rerun", original_id)
    pool.process_job(rerun_id)

    rerun_job = jobs.get_job(rerun_id)
    assert rerun_job.status == "error"
    assert "expired" in rerun_job.error or "no such" in rerun_job.error.lower()


def test_cache_hit_copies_peaks_json_alongside_stems(db, make_upload, monkeypatch):
    import numpy as np

    sr = 44100
    data = (0.05 * np.random.default_rng(52).standard_normal((sr, 2))).astype(np.float32)
    monkeypatch.setattr(pool.runner_mod, "separate_in_subprocess", _fake_separate_with_peaks_factory(data, sr))

    upload1 = make_upload(seconds=1.0, seed=52)
    job1 = jobs.create_job("music", "balanced", "upload", str(upload1))
    pool.process_job(job1)
    assert jobs.get_job(job1).status == "done"

    upload2 = make_upload(seconds=1.0, seed=52)  # identical content -> cache hit
    job2 = jobs.create_job("music", "balanced", "upload", str(upload2))
    pool.process_job(job2)
    job2_result = jobs.get_job(job2)
    assert job2_result.status == "done"
    assert job2_result.from_cache is True

    from backend.storage import job_dir

    for name in ("vocals", "drums", "bass", "other"):
        assert (job_dir(job2) / f"{name}.peaks.json").exists()
