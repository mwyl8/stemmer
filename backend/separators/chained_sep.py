"""Full mode's chained pipeline: Bandit first (speech/music/effects), then
Demucs on the *music* stem (not the original mixture) to split it into
vocals/drums/bass/other. Final stem set: speech · vocals · drums · bass ·
other — spoken dialogue, the sung vocal, and the instruments, all from one
clip (PRD §2's showcase). Bandit's "effects" stem is intentionally dropped
from the merged output — the spec for full mode is exactly these five stems.

A thin composition over two already-independent Separators (dependency
injection, not model logic of its own), so it's trivial to test with fakes
that just record call order — see test_chained_sep.py.
"""

from __future__ import annotations

import numpy as np

from backend.separators.base import Separator

MUSIC_STEM_NAME = "music"


class ChainedSeparator(Separator):
    def __init__(self, bandit: Separator, demucs: Separator):
        self.bandit = bandit
        self.demucs = demucs

    def separate(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        bandit_stems = self.bandit.separate(audio)
        demucs_stems = self.demucs.separate(bandit_stems[MUSIC_STEM_NAME])
        return {"speech": bandit_stems["speech"], **demucs_stems}
