"""Speech-vs-Singing mode's chained pipeline: the same Bandit-then-Demucs
chain as Full mode (chained_sep.py — see _two_stage_chain.py for the
shared two-pass mechanics), but merged for a different question. Full mode
asks "what are all the instrument stems"; this mode asks "does the talking
voice end up in a different file than the singing voice". Bandit's "speech"
stem already is the spoken voice; Demucs's "vocals" stem (split off Bandit's
music bus) is the sung voice; every other Demucs source — drums/bass/other,
or +guitar/piano at stem_count=6 — is summed into one `instruments` stem,
since per-instrument separation isn't this mode's point. Bandit's "effects"
stem is dropped, same as Full mode.

Singing-voice-vs-speech is a known-hard MSS problem, not a solved one: Bandit
was trained to split speech/music/effects by *source type*, and Demucs was
trained to split a music mix into instrument roles — neither model was ever
trained on the specific question "is this frame spoken or sung", so a belted
or half-spoken vocal line, or dialogue over a musical bed, can bleed across
both stems. See scripts/eval_singing_vs_speech.py for a measured, honestly-
reported bleed number on a synthetic fixture — this module does not claim
clean separation, only that the two voices are *routed* to different named
outputs when the underlying models agree on which is which.

Bandit's "speech" and Demucs's "vocals" outputs are each computed
independently and, left alone, can both claim the same vocal energy at once
(observed directly in a real take: the cached Thriller job's 5:50-6:30 has
nonzero, near-full-strength energy in both stems simultaneously). Rather
than passing the two raw estimates straight through, `_vocal_partition.py`
treats their sum as one shared "vocal bus" and splits it with a Wiener-style
soft mask that always sums to 1 per time-frequency bin — see that module's
docstring for the mask math and the frame-level pitch/harmonicity
arbitration that biases it — so the two stems partition the vocal energy
instead of each claiming all of it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from backend.config import MUSIC_SAMPLE_RATE
from backend.separators._two_stage_chain import run_chain
from backend.separators._vocal_partition import partition_vocal_bus
from backend.separators.base import Separator
from backend.separators.chained_sep import MUSIC_STEM_NAME

SPOKEN_STEM_NAME = "spoken_speech"
SUNG_STEM_NAME = "sung_vocals"
INSTRUMENTS_STEM_NAME = "instruments"

_DEMUCS_VOCAL_SOURCE = "vocals"


class SpeechVsSingingSeparator(Separator):
    def __init__(self, bandit: Separator, demucs: Separator):
        self.bandit = bandit
        self.demucs = demucs

    def runtime_info(self) -> dict[str, str] | None:
        return self.demucs.runtime_info()

    def separate(self, audio: np.ndarray, on_chunk: Callable[[int, int], None] | None = None) -> dict[str, np.ndarray]:
        bandit_stems, demucs_stems = run_chain(self.bandit, self.demucs, audio, MUSIC_STEM_NAME, on_chunk)
        instrument_sources = [demucs_stems[name] for name in demucs_stems if name != _DEMUCS_VOCAL_SOURCE]
        instruments = np.sum(instrument_sources, axis=0).astype(np.float32)
        spoken, sung = partition_vocal_bus(
            bandit_stems["speech"], demucs_stems[_DEMUCS_VOCAL_SOURCE], MUSIC_SAMPLE_RATE, accompaniment=instruments
        )
        return {
            SPOKEN_STEM_NAME: spoken,
            SUNG_STEM_NAME: sung,
            INSTRUMENTS_STEM_NAME: instruments,
        }
