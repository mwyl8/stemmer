"""Full mode's chained pipeline: Bandit first (speech/music/effects), then
Demucs on the *music* stem (not the original mixture) to split it into
vocals/drums/bass/other. Final stem set: speech · vocals · drums · bass ·
other — spoken dialogue, the sung vocal, and the instruments, all from one
clip (PRD §2's showcase). Bandit's "effects" stem is intentionally dropped
from the merged output — the spec for full mode is exactly these five stems.

A thin composition over two already-independent Separators (dependency
injection, not model logic of its own), so it's trivial to test with fakes
that just record call order — see test_chained_sep.py. The actual two-pass
mechanics (which stem feeds Demucs, how progress across both passes is
combined into one range) live in _two_stage_chain.py, shared with
singing_sep.py's Speech-vs-Singing mode and lead_backing_sep.py's Lead-vs-
Backing-Vocals mode — the only thing that differs between chained modes is
how the two separators' output stems get merged.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from backend.separators._two_stage_chain import run_chain
from backend.separators.base import Separator

MUSIC_STEM_NAME = "music"

__all__ = ["MUSIC_STEM_NAME", "ChainedSeparator"]


class ChainedSeparator(Separator):
    def __init__(self, bandit: Separator, demucs: Separator):
        self.bandit = bandit
        self.demucs = demucs

    def runtime_info(self) -> dict[str, str] | None:
        return self.demucs.runtime_info()

    def separate(self, audio: np.ndarray, on_chunk: Callable[[int, int], None] | None = None) -> dict[str, np.ndarray]:
        bandit_stems, demucs_stems = run_chain(self.bandit, self.demucs, audio, MUSIC_STEM_NAME, on_chunk)
        return {"speech": bandit_stems["speech"], **demucs_stems}
