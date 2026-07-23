"""Lead-vs-Backing-Vocals mode's chained pipeline: Demucs first, then the
karaoke model on Demucs's *vocals* stem (see test_chained_sep.py /
test_singing_sep.py for the analogous Bandit-then-Demucs pairs). Both
separators are fakes here (no models loaded), so this is a pure test of
lead_backing_sep.py's own merge logic, not either model's actual separation
quality — see scripts/eval_lead_vs_backing.py for that measurement.
"""

from __future__ import annotations

import numpy as np

from backend.separators.base import Separator
from backend.separators.lead_backing_sep import LeadBackingSeparator


class FakeDemucs(Separator):
    def __init__(self, vocals_stem, calls, n_chunks=3):
        self.vocals_stem = vocals_stem
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
            "vocals": self.vocals_stem,
            "drums": np.full_like(audio, 2.0),
            "bass": np.full_like(audio, 3.0),
            "other": np.full_like(audio, 4.0),
        }


class FakeKaraoke(Separator):
    def __init__(self, calls, n_chunks=2):
        self.calls = calls
        self.n_chunks = n_chunks

    def num_chunks(self, length):
        return self.n_chunks

    def separate(self, audio, on_chunk=None):
        self.calls.append(("karaoke", audio))
        if on_chunk is not None:
            for done in range(self.n_chunks + 1):
                on_chunk(done, self.n_chunks)
        return {
            "lead_vocal": np.full_like(audio, 1.0),
            "backing_vocals": np.full_like(audio, 0.5),
        }


def _make(n_demucs_chunks=3, n_karaoke_chunks=2):
    calls = []
    mixture = np.zeros((2, 100), dtype=np.float32)
    vocals_stem = np.full((2, 100), 0.2, dtype=np.float32)
    demucs = FakeDemucs(vocals_stem, calls, n_chunks=n_demucs_chunks)
    karaoke = FakeKaraoke(calls, n_chunks=n_karaoke_chunks)
    return LeadBackingSeparator(demucs=demucs, karaoke=karaoke), calls, mixture, vocals_stem


def test_calls_demucs_then_karaoke_in_order():
    sep, calls, mixture, _vocals_stem = _make()
    sep.separate(mixture)
    assert [name for name, _ in calls] == ["demucs", "karaoke"]


def test_feeds_demucs_vocals_stem_to_karaoke_not_the_mixture():
    sep, calls, mixture, vocals_stem = _make()
    sep.separate(mixture)
    karaoke_input = calls[1][1]
    np.testing.assert_array_equal(karaoke_input, vocals_stem)
    assert not np.array_equal(karaoke_input, mixture)


def test_output_is_lead_vocal_backing_vocals_and_summed_instruments():
    sep, _calls, mixture, _vocals_stem = _make()
    result = sep.separate(mixture)

    assert set(result.keys()) == {"lead_vocal", "backing_vocals", "instruments"}
    np.testing.assert_array_equal(result["lead_vocal"], np.full((2, 100), 1.0, dtype=np.float32))
    np.testing.assert_array_equal(result["backing_vocals"], np.full((2, 100), 0.5, dtype=np.float32))
    # instruments = drums(2.0) + bass(3.0) + other(4.0), summed
    np.testing.assert_allclose(result["instruments"], np.full((2, 100), 9.0, dtype=np.float32))


def test_demucs_vocals_source_is_not_in_final_output():
    sep, _calls, mixture, _vocals_stem = _make()
    result = sep.separate(mixture)
    assert "vocals" not in result
    assert "drums" not in result
    assert "bass" not in result
    assert "other" not in result


def test_progress_is_monotonic_and_never_resets_across_passes():
    sep, _calls, mixture, _vocals_stem = _make(n_demucs_chunks=3, n_karaoke_chunks=2)

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
    sep, _calls, mixture, _vocals_stem = _make()
    result = sep.separate(mixture)
    assert set(result.keys()) == {"lead_vocal", "backing_vocals", "instruments"}


def test_runtime_info_delegates_to_demucs():
    sep, _calls, _mixture, _vocals_stem = _make()
    sep.demucs.runtime_info = lambda: {"arch": "arm64", "provider": "CPUExecutionProvider", "model": "htdemucs_core"}
    assert sep.runtime_info() == {"arch": "arm64", "provider": "CPUExecutionProvider", "model": "htdemucs_core"}
