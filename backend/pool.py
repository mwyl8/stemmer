"""Async worker pool, capped at config.POOL_SIZE, dispatching to subprocess runners.

Stub only — the web process must never block on separation; wired in Phase 3.
"""

from __future__ import annotations

from backend.config import POOL_SIZE


class WorkerPool:
    def __init__(self, size: int = POOL_SIZE) -> None:
        self.size = size

    async def submit(self, job_id: str) -> None:
        """Queue a job for an isolated subprocess run. Phase 3."""
        raise NotImplementedError

    async def start(self) -> None:
        """Phase 3."""
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Phase 3."""
        raise NotImplementedError
