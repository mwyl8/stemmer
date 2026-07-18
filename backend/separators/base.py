"""The one interface every separator implements. Swapping/adding a model behind
this never touches the API or frontend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Separator(ABC):
    @abstractmethod
    def separate(self, audio: Any) -> dict[str, Any]:
        """audio -> {stem_name: waveform}. Implemented per-model in Phase 1/4."""
        raise NotImplementedError
