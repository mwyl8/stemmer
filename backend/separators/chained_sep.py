"""Full mode's chained pipeline: Bandit first (speech/music/effects), then
Demucs on the *music* stem (not the original mixture) to split it into
vocals/drums/bass/other. Final stem set: speech · vocals · drums · bass ·
other — spoken dialogue, the sung vocal, and the instruments, all from one
clip (PRD §2's showcase). Bandit's "effects" stem is intentionally dropped
from the merged output — the spec for full mode is exactly these five stems.

A thin composition over two already-independent Separators (dependency
injection, not model logic of its own), so it's trivial to test with fakes
that just record call order — see test_chained_sep.py.

Progress across the two passes (PRD §4): both Bandit and Demucs preserve
audio length (source separation, not resampling), so each side's chunk count
depends only on `audio`'s length, not on what the other pass produced —
`num_chunks(length)` can be asked of both *before* either one runs. That lets
`on_chunk` report one continuous [0, bandit_total + demucs_total) range from
the very first Bandit chunk onward: Bandit fills [0, bandit_total), Demucs
continues the same counter through [bandit_total, bandit_total+demucs_total)
instead of restarting at 0 when it takes over — progress never resets
partway through a chained job.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from backend.separators.base import Separator

MUSIC_STEM_NAME = "music"


class ChainedSeparator(Separator):
    def __init__(self, bandit: Separator, demucs: Separator):
        self.bandit = bandit
        self.demucs = demucs

    def separate(self, audio: np.ndarray, on_chunk: Callable[[int, int], None] | None = None) -> dict[str, np.ndarray]:
        if on_chunk is None:
            bandit_stems = self.bandit.separate(audio)
            demucs_stems = self.demucs.separate(bandit_stems[MUSIC_STEM_NAME])
            return {"speech": bandit_stems["speech"], **demucs_stems}

        length = audio.shape[-1]
        bandit_total = self.bandit.num_chunks(length)
        demucs_total = self.demucs.num_chunks(length)  # same length in -> same length out
        grand_total = bandit_total + demucs_total

        def bandit_cb(done: int, _total: int) -> None:
            on_chunk(done, grand_total)

        def demucs_cb(done: int, _total: int) -> None:
            on_chunk(bandit_total + done, grand_total)

        bandit_stems = self.bandit.separate(audio, on_chunk=bandit_cb)
        demucs_stems = self.demucs.separate(bandit_stems[MUSIC_STEM_NAME], on_chunk=demucs_cb)
        return {"speech": bandit_stems["speech"], **demucs_stems}
