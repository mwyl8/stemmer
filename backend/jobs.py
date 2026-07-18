"""Job lifecycle + status/progress model. Wiring lands in Phase 3 (jobs + API + pool).

Stub only — no persistence logic yet, just the shape the API and pool will share.
"""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DECODING = "decoding"
    SEPARATING = "separating"
    ENCODING = "encoding"
    DONE = "done"
    ERROR = "error"


def create_job(mode: str, tier: str, source_type: str, source_ref: str | None) -> str:
    """Insert a job row, return job_id. Phase 3."""
    raise NotImplementedError


def get_job(job_id: str) -> dict:
    """Fetch job status/stage/progress/stems. Phase 3."""
    raise NotImplementedError


def update_stage(job_id: str, stage: Stage, progress: float = 0.0) -> None:
    """Worker-side progress update, polled by GET /jobs/{id}. Phase 3."""
    raise NotImplementedError


def mark_error(job_id: str, error: str) -> None:
    """Phase 3."""
    raise NotImplementedError
