"""Shared Bandit-then-Demucs chain mechanics, used by both Full mode
(chained_sep.py) and Speech-vs-Singing mode (singing_sep.py) — every mode
that runs Bandit first and feeds its "music" stem (not the original mixture)
into Demucs shares the exact same two concerns: which stem to hand off, and
how to fold both passes' chunk progress into one continuous range. Only the
*merge* step (which stems end up in the final dict, under what names) differs
between modes, so that stays in each mode's own module.

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


def run_chain(
    bandit: Separator,
    demucs: Separator,
    audio: np.ndarray,
    on_chunk: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Runs Bandit, then Demucs on Bandit's music stem. Returns
    (bandit_stems, demucs_stems) — merging them into a mode's final stem set
    is the caller's job."""
    if on_chunk is None:
        bandit_stems = bandit.separate(audio)
        demucs_stems = demucs.separate(bandit_stems[MUSIC_STEM_NAME])
        return bandit_stems, demucs_stems

    length = audio.shape[-1]
    bandit_total = bandit.num_chunks(length)
    demucs_total = demucs.num_chunks(length)  # same length in -> same length out
    grand_total = bandit_total + demucs_total

    def bandit_cb(done: int, _total: int) -> None:
        on_chunk(done, grand_total)

    def demucs_cb(done: int, _total: int) -> None:
        on_chunk(bandit_total + done, grand_total)

    bandit_stems = bandit.separate(audio, on_chunk=bandit_cb)
    demucs_stems = demucs.separate(bandit_stems[MUSIC_STEM_NAME], on_chunk=demucs_cb)
    return bandit_stems, demucs_stems
