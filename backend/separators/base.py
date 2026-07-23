"""The one interface every separator implements. Swapping/adding a model behind
this never touches the API or frontend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class Separator(ABC):
    @abstractmethod
    def separate(self, audio: Any, on_chunk: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """audio -> {stem_name: waveform}. Implemented per-model in Phase 1/4.

        `on_chunk(chunks_done, chunks_total)`, if given, is called once
        immediately (0, total) so the total is known before any work
        completes, then again after each chunk finishes — runner.py wires
        this to jobs.update_progress() for live progress (PRD §4)."""
        raise NotImplementedError

    def runtime_info(self) -> dict[str, str] | None:
        """Optional: {"arch", "provider", "model"} describing which host
        architecture bucket, ONNX Runtime execution provider, and model file
        this separator actually resolved to (config.ARCH_RUNTIME_PROFILES) —
        surfaced in job metadata (GET /jobs/{id}, runner.py's `_main`). None
        for separators with no such runtime (e.g. Bandit's plain PyTorch CPU
        path, which has no ONNX provider concept)."""
        return None
