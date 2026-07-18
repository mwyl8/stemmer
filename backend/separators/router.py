"""Mode + tier -> pick/chain separators. Music mode = Demucs; Video mode = Bandit;
Full mode chains Bandit -> Demucs on the music bus.

Stub only — concrete separators (demucs_onnx.py, bandit_sep.py) land in Phase 1/4.
"""

from __future__ import annotations

from backend.separators.base import Separator


def select_separator(mode: str, tier: str) -> Separator:
    """Phase 1 (music), Phase 4 (video/full chained)."""
    raise NotImplementedError
