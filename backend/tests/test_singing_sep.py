"""Speech-vs-Singing mode's chained pipeline: same Bandit-then-Demucs chain
as Full mode (see test_chained_sep.py), but merged into spoken_speech /
sung_vocals / instruments. Both separators are fakes here (no models
loaded), so this is a pure test of singing_sep.py's own merge logic, not
either model's actual separation quality — see scripts/eval_singing_vs_speech.py
for that measurement.
"""

from __future__ import annotations

import numpy as np

from backend.separators.base import Separator
from backend.separators.singing_sep import SpeechVsSingingSeparator


class FakeBandit(Separator):
    def __init__(self, music_stem, calls, n_chunks=3):
        self.music_stem = music_stem
        self.calls = calls
        self.n_chunks = n_chunks

    def num_chunks(self, length):
        return self.n_chunks

    def separate(self, audio, on_chunk=None):
        self.calls.append(("bandit", audio))
        if on_chunk is not None:
            for done in range(self.n_chunks + 1):
                on_chunk(done, self.n_chunks)
        return {
            "speech": np.full_like(audio, 0.1),
            "music": self.music_stem,
            "effects": np.full_like(audio, 0.3),
        }


class FakeDemucs(Separator):
    def __init__(self, calls, n_chunks=2):
        self.calls = calls
        self.n_chunks = n_chunks

    def num_chunks(self, length):
        return self.n_chunks

    def separate(self, audio, on_chunk=None):
        self.calls.append(("demucs", audio))
        if on_chunk is not None:
            for done in range(self.n_chunks + 1):
                on_chunk(done, self.n_chunks)
        return {
            "vocals": np.full_like(audio, 1.0),
            "drums": np.full_like(audio, 2.0),
            "bass": np.full_like(audio, 3.0),
            "other": np.full_like(audio, 4.0),
        }


def _make(n_bandit_chunks=3, n_demucs_chunks=2):
    calls = []
    mixture = np.zeros((2, 100), dtype=np.float32)
    music_stem = np.full((2, 100), 0.2, dtype=np.float32)
    bandit = FakeBandit(music_stem, calls, n_chunks=n_bandit_chunks)
    demucs = FakeDemucs(calls, n_chunks=n_demucs_chunks)
    return SpeechVsSingingSeparator(bandit=bandit, demucs=demucs), calls, mixture, music_stem


def test_calls_bandit_then_demucs_in_order():
    sep, calls, mixture, _music_stem = _make()
    sep.separate(mixture)
    assert [name for name, _ in calls] == ["bandit", "demucs"]


def test_feeds_bandits_music_stem_to_demucs_not_the_mixture():
    sep, calls, mixture, music_stem = _make()
    sep.separate(mixture)
    demucs_input = calls[1][1]
    np.testing.assert_array_equal(demucs_input, music_stem)
    assert not np.array_equal(demucs_input, mixture)


def test_output_is_spoken_speech_sung_vocals_and_summed_instruments():
    sep, _calls, mixture, _music_stem = _make()
    result = sep.separate(mixture)

    assert set(result.keys()) == {"spoken_speech", "sung_vocals", "instruments"}
    # spoken_speech and sung_vocals no longer pass Bandit's/Demucs's raw
    # estimates straight through -- _vocal_partition.py splits their combined
    # energy so the two stems partition it instead of each claiming all of
    # it (see test_vocal_partition.py for that invariant in detail). Here,
    # just check the combined vocal energy is preserved: 0.1 (bandit speech)
    # + 1.0 (demucs vocals) = 1.1.
    combined = result["spoken_speech"].astype(np.float64) + result["sung_vocals"].astype(np.float64)
    np.testing.assert_allclose(combined, np.full((2, 100), 1.1), atol=1e-3)
    # instruments = drums(2.0) + bass(3.0) + other(4.0), summed -- untouched by the partition fix
    np.testing.assert_allclose(result["instruments"], np.full((2, 100), 9.0, dtype=np.float32))


def test_bandits_effects_stem_is_dropped():
    sep, _calls, mixture, _music_stem = _make()
    result = sep.separate(mixture)
    assert "effects" not in result
    assert "music" not in result  # replaced by demucs's split


def test_progress_is_monotonic_and_never_resets_across_passes():
    sep, _calls, mixture, _music_stem = _make(n_bandit_chunks=3, n_demucs_chunks=2)

    progress_events = []
    sep.separate(mixture, on_chunk=lambda done, total: progress_events.append((done, total)))

    grand_total = 3 + 2
    assert all(total == grand_total for _done, total in progress_events)
    done_values = [done for done, _total in progress_events]
    assert done_values == sorted(done_values)
    assert done_values[0] == 0
    assert done_values[-1] == grand_total
    assert done_values.count(0) == 1


def test_progress_is_none_when_no_callback_given():
    sep, _calls, mixture, _music_stem = _make()
    result = sep.separate(mixture)
    assert set(result.keys()) == {"spoken_speech", "sung_vocals", "instruments"}


def test_runtime_info_delegates_to_demucs():
    sep, _calls, _mixture, _music_stem = _make()
    sep.demucs.runtime_info = lambda: {"arch": "arm64", "provider": "CoreMLExecutionProvider", "model": "htdemucs_core"}
    assert sep.runtime_info() == {"arch": "arm64", "provider": "CoreMLExecutionProvider", "model": "htdemucs_core"}
