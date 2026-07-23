"""Shared two-pass chain mechanics: run one Separator, then a second on a
named stem from the first's output (not the original mixture). Every chained
mode shares this exact shape — Full mode and Speech-vs-Singing mode both run
Bandit first and feed its "music" stem to Demucs; Lead-vs-Backing-Vocals mode
runs Demucs first and feeds its "vocals" stem to the karaoke model — so the
mechanics (which stem to hand off, how to fold both passes' chunk progress
into one continuous range) live here once. Only the *merge* step (which
stems end up in the final dict, under what names) differs between modes, so
that stays in each mode's own module (chained_sep.py, singing_sep.py,
lead_backing_sep.py).

Progress across the two passes (PRD §4): both stages preserve audio length
(source separation, not resampling), so each side's chunk count depends
only on `audio`'s length, not on what the other pass produced —
`num_chunks(length)` can be asked of both *before* either one runs. That lets
`on_chunk` report one continuous [0, first_total + second_total) range from
the very first chunk onward: stage one fills [0, first_total), stage two
continues the same counter through [first_total, first_total+second_total)
instead of restarting at 0 when it takes over — progress never resets
partway through a chained job.

(This module was named `_bandit_demucs_chain.py` until Lead-vs-Backing-
Vocals mode needed the exact same mechanics for a different pair of models —
renamed and genericized rather than duplicated.)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from backend.separators.base import Separator


def run_chain(
    first: Separator,
    second: Separator,
    audio: np.ndarray,
    feed_stem_name: str,
    on_chunk: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Runs `first`, then `second` on `first`'s `feed_stem_name` stem.
    Returns (first_stems, second_stems) — merging them into a mode's final
    stem set is the caller's job."""
    if on_chunk is None:
        first_stems = first.separate(audio)
        second_stems = second.separate(first_stems[feed_stem_name])
        return first_stems, second_stems

    length = audio.shape[-1]
    first_total = first.num_chunks(length)
    second_total = second.num_chunks(length)  # same length in -> same length out
    grand_total = first_total + second_total

    def first_cb(done: int, _total: int) -> None:
        on_chunk(done, grand_total)

    def second_cb(done: int, _total: int) -> None:
        on_chunk(first_total + done, grand_total)

    first_stems = first.separate(audio, on_chunk=first_cb)
    second_stems = second.separate(first_stems[feed_stem_name], on_chunk=second_cb)
    return first_stems, second_stems
