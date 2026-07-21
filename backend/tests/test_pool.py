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
