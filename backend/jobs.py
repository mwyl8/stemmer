"""Job lifecycle + status/progress model, persisted via storage.py (SQLite).

`status` is coarse (queued|running|done|error|expired) — what pool.py filters
on to find work, and what a client checks to know "is this finished". `stage`
is the granular pipeline position (queued -> downloading/decoding ->
separating -> encoding -> done, or error) — what a client polls for a
progress bar. On top of that, `chunks_done`/`chunks_total` + `eta_seconds`
give live sub-progress *within* the separating stage — runner.py's child
subprocess calls `update_progress()` directly (it's a separate process, but
shares the same SQLite file) after each chunk it finishes; see runner.py's
module docstring for why a direct per-chunk DB write beats a pipe/callback
here (chunk counts are small — tens, not thousands — so it never hammers
SQLite).

Also owns TTL purge: `purge_expired_jobs()` is the testable synchronous unit
(delete stem files, mark the job row expired); `run_purge_loop()` wraps it in
a periodic background task for app.py's lifespan.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from backend.config import DEFAULT_STEM_COUNT, PURGE_INTERVAL_SECONDS, TTL_SECONDS
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

# Approximate overall progress (percent, 0-100) when a stage begins/ends.
# Separating spans [SEPARATING, ENCODING) — update_progress() interpolates
# within that band from chunks_done/chunks_total, so progress_pct keeps
# climbing smoothly through the slow stage instead of jumping 25 -> 90 in
# one step.
_PROGRESS_PCT_FOR_STAGE = {
    Stage.QUEUED: 0,
    Stage.DOWNLOADING: 5,
    Stage.DECODING: 15,
    Stage.SEPARATING: 25,
    Stage.ENCODING: 90,
    Stage.DONE: 100,
}

_TERMINAL_STATUSES = ("done", "error", "expired")


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
    stem_count: int
    source_type: str
    source_ref: str | None
    content_hash: str | None
    progress_pct: int
    error: str | None
    created_at: str
    expires_at: str
    chunks_done: int
    chunks_total: int
    stage_timings: dict
    stage_started_at: str | None
    eta_seconds: float | None
    from_cache: bool
    elapsed_seconds: float
    source_codec: str | None
    source_bitrate: int | None
    source_sample_rate: int | None
    source_channels: int | None
    source_wav_path: str | None
    runtime_arch: str | None
    runtime_provider: str | None
    runtime_model: str | None
    stems: list[Stem] = field(default_factory=list)

    @property
    def submitted_at(self) -> str:
        return self.created_at


def create_job(
    mode: str,
    tier: str,
    source_type: str,
    source_ref: str | None,
    stem_count: int = DEFAULT_STEM_COUNT,
) -> str:
    job_id = uuid.uuid4().hex
    now = _now_iso()
    expires_at = _iso(time.time() + TTL_SECONDS)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (id, status, stage, mode, tier, stem_count, source_type, source_ref, "
            "content_hash, progress_pct, error, created_at, expires_at, stage_timings, stage_started_at) "
            "VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, NULL, 0, NULL, ?, ?, ?, ?)",
            (
                job_id,
                Stage.QUEUED.value,
                mode,
                tier,
                stem_count,
                source_type,
                source_ref,
                now,
                expires_at,
                json.dumps({Stage.QUEUED.value: {"started_at": now}}),
                now,
            ),
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
    stage_timings = json.loads(row["stage_timings"]) if row["stage_timings"] else {}
    return Job(
        id=row["id"],
        status=row["status"],
        stage=row["stage"],
        mode=row["mode"],
        tier=row["tier"],
        stem_count=row["stem_count"],
        source_type=row["source_type"],
        source_ref=row["source_ref"],
        content_hash=row["content_hash"],
        progress_pct=row["progress_pct"],
        error=row["error"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        chunks_done=row["chunks_done"],
        chunks_total=row["chunks_total"],
        stage_timings=stage_timings,
        stage_started_at=row["stage_started_at"],
        eta_seconds=row["eta_seconds"],
        from_cache=bool(row["from_cache"]),
        elapsed_seconds=_compute_elapsed(row["created_at"], row["status"], stage_timings),
        source_codec=row["source_codec"],
        source_bitrate=row["source_bitrate"],
        source_sample_rate=row["source_sample_rate"],
        source_channels=row["source_channels"],
        source_wav_path=row["source_wav_path"],
        runtime_arch=row["runtime_arch"],
        runtime_provider=row["runtime_provider"],
        runtime_model=row["runtime_model"],
        stems=stems,
    )


def update_stage(job_id: str, stage: Stage, progress_pct: int | None = None) -> None:
    if progress_pct is None:
        progress_pct = _PROGRESS_PCT_FOR_STAGE[stage]
    now = _now_iso()
    with get_connection() as conn:
        row = conn.execute("SELECT stage, stage_timings FROM jobs WHERE id = ?", (job_id,)).fetchone()
        timings = json.loads(row["stage_timings"]) if row and row["stage_timings"] else {}
        prev_stage = row["stage"] if row else None
        if prev_stage and prev_stage in timings and "ended_at" not in timings[prev_stage]:
            timings[prev_stage]["ended_at"] = now
        timings.setdefault(stage.value, {})["started_at"] = now
        conn.execute(
            "UPDATE jobs SET stage = ?, status = ?, progress_pct = ?, stage_timings = ?, stage_started_at = ? WHERE id = ?",
            (stage.value, _STATUS_FOR_STAGE[stage], progress_pct, json.dumps(timings), now, job_id),
        )


def update_progress(job_id: str, chunks_done: int, chunks_total: int) -> None:
    """Called once per finished chunk (batching-of-one is fine at these chunk
    counts — see module docstring) from inside the separating stage, whether
    that's a single Demucs pass or a chained Bandit-then-Demucs run where the
    caller (chained_sep.py) has already folded both passes into one
    contiguous [0, chunks_total) range so this never regresses partway
    through. Interpolates the coarse SEPARATING->ENCODING progress_pct band
    (integer 0-100, per PRD §4) from chunk fraction, and refines eta_seconds
    from measured per-chunk throughput within the current stage."""
    now = time.time()
    frac = (chunks_done / chunks_total) if chunks_total else 0.0
    stage_start = _PROGRESS_PCT_FOR_STAGE[Stage.SEPARATING]
    stage_end = _PROGRESS_PCT_FOR_STAGE[Stage.ENCODING]
    progress_pct = round(stage_start + frac * (stage_end - stage_start))

    with get_connection() as conn:
        row = conn.execute("SELECT stage_started_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
        eta = None
        if chunks_total and chunks_done >= chunks_total:
            eta = 0.0
        elif row and row["stage_started_at"] and chunks_done > 0:
            elapsed_in_stage = now - datetime.fromisoformat(row["stage_started_at"]).timestamp()
            eta = (elapsed_in_stage / chunks_done) * (chunks_total - chunks_done)
        conn.execute(
            "UPDATE jobs SET chunks_done = ?, chunks_total = ?, progress_pct = ?, eta_seconds = ? WHERE id = ?",
            (chunks_done, chunks_total, progress_pct, eta, job_id),
        )


def set_initial_eta(job_id: str, eta_seconds: float) -> None:
    """Seed eta_seconds from audio_duration * config.TIER_RTF before the
    first chunk lands (pool.py calls this right before dispatching to the
    separator); update_progress() takes over and refines it once real chunk
    throughput is available."""
    with get_connection() as conn:
        conn.execute("UPDATE jobs SET eta_seconds = ? WHERE id = ?", (eta_seconds, job_id))


def mark_from_cache(job_id: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE jobs SET from_cache = 1 WHERE id = ?", (job_id,))


def mark_error(job_id: str, error: str) -> None:
    now = _now_iso()
    with get_connection() as conn:
        row = conn.execute("SELECT stage, stage_timings FROM jobs WHERE id = ?", (job_id,)).fetchone()
        timings = json.loads(row["stage_timings"]) if row and row["stage_timings"] else {}
        prev_stage = row["stage"] if row else None
        if prev_stage and prev_stage in timings and "ended_at" not in timings[prev_stage]:
            timings[prev_stage]["ended_at"] = now
        conn.execute(
            "UPDATE jobs SET stage = ?, status = 'error', error = ?, stage_timings = ? WHERE id = ?",
            (Stage.ERROR.value, error, json.dumps(timings), job_id),
        )


def set_content_hash(job_id: str, content_hash: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE jobs SET content_hash = ? WHERE id = ?", (content_hash, job_id))


def set_source_info(
    job_id: str,
    codec: str | None,
    bitrate: int | None,
    sample_rate: int | None,
    channels: int | None,
) -> None:
    """What the pipeline actually started from (PRD Addendum §2.5 job
    metadata panel) — read off the source file by ingest.probe_source_info
    before decode() normalizes it away."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET source_codec = ?, source_bitrate = ?, source_sample_rate = ?, source_channels = ? WHERE id = ?",
            (codec, bitrate, sample_rate, channels, job_id),
        )


def set_runtime_info(job_id: str, arch: str, provider: str, model: str) -> None:
    """Records which host arch bucket, ONNX Runtime execution provider, and
    model file actually separated this job (backend.arch, separator's
    `runtime_info()`) — surfaced in GET /jobs/{id} and the job metadata panel
    (PRD Addendum §2.5), same idea as set_source_info for the input side."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET runtime_arch = ?, runtime_provider = ?, runtime_model = ? WHERE id = ?",
            (arch, provider, model, job_id),
        )


def set_source_wav_path(job_id: str, path: str) -> None:
    """Records where the normalized source WAV was persisted (job_dir(id),
    same TTL as the job's stems) — read back by pool.py for re-run and by
    app.py to serve "the original" for A/B comparison (PRD Addendum §2.3,
    §2.5)."""
    with get_connection() as conn:
        conn.execute("UPDATE jobs SET source_wav_path = ? WHERE id = ?", (path, job_id))


def add_stem(job_id: str, name: str, format: str, path: str, duration: float | None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO stems (id, job_id, name, format, path, duration) VALUES (?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, job_id, name, format, path, duration),
        )


def _compute_elapsed(created_at: str, status: str, stage_timings: dict) -> float:
    start = datetime.fromisoformat(created_at)
    if status in _TERMINAL_STATUSES:
        ends = [datetime.fromisoformat(v["ended_at"]) for v in stage_timings.values() if v.get("ended_at")]
        end = max(ends) if ends else start
    else:
        end = datetime.now(timezone.utc)
    return (end - start).total_seconds()


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
        conn.execute("UPDATE jobs SET status = 'expired', progress_pct = 0 WHERE id = ?", (job_id,))
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
