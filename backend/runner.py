"""Isolated-subprocess entrypoint: runs one separation, then exits.

Stub only — real separation wiring lands in Phase 1 (behind separators/base.py),
dispatched by pool.py starting Phase 3. Load the model once per worker, not per job.
"""

from __future__ import annotations


def run(job_id: str) -> None:
    """Load inputs for job_id, call the routed Separator, write stems. Phase 1/3."""
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError("runner is invoked by pool.py, not directly, once implemented")
