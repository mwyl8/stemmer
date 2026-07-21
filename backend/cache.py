"""Content-hash dedup: skip re-separating identical audio. Keyed by
(content_hash, mode, tier, pipeline_version) — the same audio at a different
tier is a different quality result, so it gets its own cache entry, and a
different pipeline_version (config.PIPELINE_VERSION) means the separation/
limiting/encoding code has changed since that entry was written, so it must
not be served as a hit either.

The cache row's job_id has an ON DELETE CASCADE FK to jobs (storage.py):
once a job's stems are purged (TTL or DELETE /jobs/{id}), its cache entries
disappear automatically, so a cache hit can never point at deleted files.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from backend.storage import get_connection


def find(content_hash: str, mode: str, tier: str, pipeline_version: int) -> str | None:
    """Returns a prior job_id whose stems are ready for this exact
    (content_hash, mode, tier, pipeline_version), or None on a cache miss."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT cache.job_id FROM cache "
            "JOIN jobs ON jobs.id = cache.job_id "
            "WHERE cache.content_hash = ? AND cache.mode = ? AND cache.tier = ? "
            "AND cache.pipeline_version = ? AND jobs.status = 'done'",
            (content_hash, mode, tier, pipeline_version),
        ).fetchone()
    return row["job_id"] if row else None


def put(content_hash: str, mode: str, tier: str, pipeline_version: int, job_id: str) -> None:
    """Record that `job_id` holds the finished stems for this
    (content_hash, mode, tier, pipeline_version). Call only once the job is
    actually done."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache "
            "(content_hash, mode, tier, pipeline_version, job_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (content_hash, mode, tier, pipeline_version, job_id, _now_iso()),
        )


def _now_iso() -> str:
    return datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat()
