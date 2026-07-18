"""TTL purge: a background sweep deletes source+stem files for expired jobs
and marks them expired. test_jobs.py covers purge_job/purge_expired_jobs unit
behavior directly; this file covers the multi-job sweep and the async
background-loop wrapper used by app.py's lifespan.
"""

from __future__ import annotations

import asyncio
import time

from backend import jobs


def _make_job_with_files(db, expires_delta_seconds):
    job_id = jobs.create_job("music", "fast", "upload", "song.mp3")

    # backdate/forward the expiry directly so we don't have to wait real time
    from backend.storage import get_connection

    expires_at = jobs._iso(time.time() + expires_delta_seconds)
    with get_connection() as conn:
        conn.execute("UPDATE jobs SET expires_at = ? WHERE id = ?", (expires_at, job_id))

    job_dir = db / "data" / "stems" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "vocals.wav").write_bytes(b"fake wav bytes")
    (job_dir / "vocals.mp3").write_bytes(b"fake mp3 bytes")
    jobs.add_stem(job_id, "vocals", "wav", str(job_dir / "vocals.wav"), 3.0)
    jobs.add_stem(job_id, "vocals", "mp3", str(job_dir / "vocals.mp3"), 3.0)
    return job_id, job_dir


def test_purge_sweep_removes_only_expired_jobs_files(db):
    expired_id, expired_dir = _make_job_with_files(db, expires_delta_seconds=-10)
    fresh_id, fresh_dir = _make_job_with_files(db, expires_delta_seconds=3600)

    purged = jobs.purge_expired_jobs()

    assert purged == [expired_id]
    assert not expired_dir.exists()
    assert fresh_dir.exists()
    assert jobs.get_job(expired_id).status == "expired"
    assert jobs.get_job(fresh_id).status == "queued"


def test_purge_sweep_removes_stem_rows_for_expired_job(db):
    expired_id, _ = _make_job_with_files(db, expires_delta_seconds=-10)
    jobs.purge_expired_jobs()
    assert jobs.get_job(expired_id).stems == []


def test_run_purge_loop_sweeps_on_each_tick_until_cancelled(db):
    expired_id, expired_dir = _make_job_with_files(db, expires_delta_seconds=-10)

    async def run():
        task = asyncio.create_task(jobs.run_purge_loop(interval=0.05))
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())

    assert not expired_dir.exists()
    assert jobs.get_job(expired_id).status == "expired"
