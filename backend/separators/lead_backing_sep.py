"""Lead-vs-Backing-Vocals mode's chained pipeline ("lyrical separation"):
Demucs first (vocals/drums/bass/other), then the MelBandRoformer karaoke
model on the *vocals* stem (not the original mixture) to split it further
into lead_vocal vs backing_vocals — same two-pass shape as Full mode and
Speech-vs-Singing mode (see _two_stage_chain.py for the shared mechanics),
just a different pair of models and a different feed stem ("vocals", not
Bandit's "music"). Demucs's non-vocal sources (drums/bass/other, or
+guitar/piano at stem_count=6) are summed into one `instruments` stem, same
posture as singing_sep.py's Speech-vs-Singing mode: per-instrument
separation isn't this mode's question.

Do not attempt singer-by-singer separation here (splitting two distinct
*lead* singers apart) — that's a separate, unsolved, diffusion-heavy problem
and explicitly out of scope. This mode only answers "is this the lead vocal
or a backing/harmony vocal", the same question the underlying karaoke model
was trained on (see karaoke_onnx.py's docstring for how a full-mix-trained
"Vocals vs Instrumental" model gets repurposed here).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from backend.separators._two_stage_chain import run_chain
from backend.separators.base import Separator

LEAD_STEM_NAME = "lead_vocal"
BACKING_STEM_NAME = "backing_vocals"
INSTRUMENTS_STEM_NAME = "instruments"

_DEMUCS_VOCAL_SOURCE = "vocals"
_KARAOKE_LEAD_SOURCE = "lead_vocal"
_KARAOKE_BACKING_SOURCE = "backing_vocals"


class LeadBackingSeparator(Separator):
    def __init__(self, demucs: Separator, karaoke: Separator):
        self.demucs = demucs
        self.karaoke = karaoke

    def runtime_info(self) -> dict[str, str] | None:
        return self.demucs.runtime_info()

    def separate(self, audio: np.ndarray, on_chunk: Callable[[int, int], None] | None = None) -> dict[str, np.ndarray]:
        demucs_stems, karaoke_stems = run_chain(self.demucs, self.karaoke, audio, _DEMUCS_VOCAL_SOURCE, on_chunk)
        instrument_sources = [demucs_stems[name] for name in demucs_stems if name != _DEMUCS_VOCAL_SOURCE]
        instruments = np.sum(instrument_sources, axis=0).astype(np.float32)
        return {
            LEAD_STEM_NAME: karaoke_stems[_KARAOKE_LEAD_SOURCE],
            BACKING_STEM_NAME: karaoke_stems[_KARAOKE_BACKING_SOURCE],
            INSTRUMENTS_STEM_NAME: instruments,
        }
