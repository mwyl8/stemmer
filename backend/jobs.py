"""Job lifecycle + status/progress model, persisted via storage.py (SQLite).

`status` is coarse (queued|running|done|error|expired) — what pool.py filters
on to find work, and what a client checks to know "is this finished". `stage`
is the granular pipeline position (queued -> downloading/decoding ->
separating -> encoding -> done, or error) — what a client polls for a
progress bar.

Also owns TTL purge: `purge_expired_jobs()` is the testable synchronous unit
(delete stem files, mark the job row expired); `run_purge_loop()` wraps it in
a periodic background task for app.py's lifespan.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from backend.config import PURGE_INTERVAL_SECONDS, TTL_SECONDS
from backend.storage import get_connection, job_dir

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DECODING = "decoding"
    SEPARATING = "separating"
    ENCODING = "encoding"
    DONE = "done"
    ERROR = "error"


# Coarse status derived from stage — what pool.py/cache.py filter on.
_STATUS_FOR_STAGE = {
    Stage.QUEUED: "queued",
    Stage.DOWNLOADING: "running",
    Stage.DECODING: "running",
    Stage.SEPARATING: "running",
    Stage.ENCODING: "running",
    Stage.DONE: "done",
    Stage.ERROR: "error",
}

# Approximate overall progress when a stage begins. Coarse on purpose: a
# single blocking subprocess call backs separating/encoding, so there's no
# finer-grained signal to report without deeper callback plumbing (out of
# scope here — SSE/streamed progress is a v2 upgrade per the PRD).
_PROGRESS_FOR_STAGE = {
    Stage.QUEUED: 0.0,
    Stage.DOWNLOADING: 0.05,
    Stage.DECODING: 0.15,
    Stage.SEPARATING: 0.25,
    Stage.ENCODING: 0.9,
    Stage.DONE: 1.0,
}


@dataclass
class Stem:
    name: str
    format: str
    path: str
    duration: float | None


@dataclass
class Job:
    id: str
    status: str
    stage: str
    mode: str
    tier: str
    source_type: str
    source_ref: str | None
    content_hash: str | None
    progress: float
    error: str | None
    created_at: str
    expires_at: str
    stems: list[Stem] = field(default_factory=list)


def create_job(mode: str, tier: str, source_type: str, source_ref: str | None) -> str:
    job_id = uuid.uuid4().hex
    now = _now_iso()
    expires_at = _iso(time.time() + TTL_SECONDS)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (id, status, stage, mode, tier, source_type, source_ref, "
            "content_hash, progress, error, created_at, expires_at) "
            "VALUES (?, 'queued', ?, ?, ?, ?, ?, NULL, 0, NULL, ?, ?)",
            (job_id, Stage.QUEUED.value, mode, tier, source_type, source_ref, now, expires_at),
        )
    return job_id


def get_job(job_id: str) -> Job | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        stem_rows = conn.execute(
            "SELECT name, format, path, duration FROM stems WHERE job_id = ? ORDER BY name, format",
            (job_id,),
        ).fetchall()
    stems = [Stem(name=r["name"], format=r["format"], path=r["path"], duration=r["duration"]) for r in stem_rows]
    return Job(
        id=row["id"],
        status=row["status"],
        stage=row["stage"],
        mode=row["mode"],
        tier=row["tier"],
        source_type=row["source_type"],
        source_ref=row["source_ref"],
        content_hash=row["content_hash"],
        progress=row["progress"],
        error=row["error"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        stems=stems,
    )


def update_stage(job_id: str, stage: Stage, progress: float | None = None) -> None:
    if progress is None:
        progress = _PROGRESS_FOR_STAGE[stage]
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET stage = ?, status = ?, progress = ? WHERE id = ?",
            (stage.value, _STATUS_FOR_STAGE[stage], progress, job_id),
        )


def mark_error(job_id: str, error: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET stage = ?, status = 'error', error = ? WHERE id = ?",
            (Stage.ERROR.value, error, job_id),
        )


def set_content_hash(job_id: str, content_hash: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE jobs SET content_hash = ? WHERE id = ?", (content_hash, job_id))


def add_stem(job_id: str, name: str, format: str, path: str, duration: float | None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO stems (id, job_id, name, format, path, duration) VALUES (?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, job_id, name, format, path, duration),
        )


# ---------------------------------------------------------------------------
# Manual + TTL purge — same underlying action, different trigger.
# ---------------------------------------------------------------------------


def purge_job(job_id: str) -> bool:
    """Delete a job's stem directory from disk and mark it expired. Used by
    both DELETE /jobs/{id} (purge now) and the TTL sweep (purge_expired_jobs).
    Returns False if the job doesn't exist or was already purged — the row
    stays around (marked expired) after a purge, so re-checking status here
    is what makes a second DELETE 404 instead of silently "succeeding" again.
    """
    with get_connection() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or row["status"] == "expired":
            return False
        conn.execute("DELETE FROM stems WHERE job_id = ?", (job_id,))
        conn.execute("UPDATE jobs SET status = 'expired', progress = 0 WHERE id = ?", (job_id,))
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    return True


def purge_expired_jobs(now: float | None = None) -> list[str]:
    """One purge sweep: find every non-expired job past its TTL, purge it.
    Returns the list of purged job ids. Synchronous and directly testable —
    `run_purge_loop` just calls this on a timer."""
    now_iso = _iso(now if now is not None else time.time())
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE expires_at <= ? AND status != 'expired'", (now_iso,)
        ).fetchall()
    purged = []
    for row in rows:
        if purge_job(row["id"]):
            purged.append(row["id"])
    return purged


async def run_purge_loop(interval: float = PURGE_INTERVAL_SECONDS) -> None:
    """Background task: sweep for expired jobs every `interval` seconds until
    cancelled (app.py's lifespan cancels it on shutdown)."""
    while True:
        try:
            purged = await asyncio.to_thread(purge_expired_jobs)
            if purged:
                logger.info("TTL purge: removed %d expired job(s)", len(purged))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("TTL purge sweep failed")
        await asyncio.sleep(interval)


def _now_iso() -> str:
    return _iso(time.time())


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
