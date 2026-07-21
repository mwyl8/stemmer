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
