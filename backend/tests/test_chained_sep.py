"""Full mode's chained pipeline: Bandit runs first, its "music" stem (not
the original mixture) feeds Demucs, and the merged output is exactly
speech + vocals/drums/bass/other — effects is dropped. Both separators are
fakes here (no models loaded), so this stays fast/offline; it's a pure test
of chained_sep.py's own composition logic, not either model's correctness.
"""

from __future__ import annotations

import numpy as np

from backend.separators.base import Separator
from backend.separators.chained_sep import ChainedSeparator


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


def test_chained_separator_calls_bandit_then_demucs_in_order():
    calls = []
    mixture = np.zeros((2, 100), dtype=np.float32)
    music_stem = np.full((2, 100), 0.2, dtype=np.float32)

    bandit = FakeBandit(music_stem, calls)
    demucs = FakeDemucs(calls)
    chained = ChainedSeparator(bandit=bandit, demucs=demucs)

    chained.separate(mixture)

    assert [name for name, _ in calls] == ["bandit", "demucs"]


def test_chained_separator_feeds_bandits_music_stem_to_demucs_not_the_mixture():
    calls = []
    mixture = np.zeros((2, 100), dtype=np.float32)
    music_stem = np.full((2, 100), 0.2, dtype=np.float32)

    bandit = FakeBandit(music_stem, calls)
    demucs = FakeDemucs(calls)
    chained = ChainedSeparator(bandit=bandit, demucs=demucs)

    chained.separate(mixture)

    demucs_input = calls[1][1]
    np.testing.assert_array_equal(demucs_input, music_stem)
    assert not np.array_equal(demucs_input, mixture)


def test_chained_separator_output_is_speech_plus_demucs_four_stems():
    calls = []
    mixture = np.zeros((2, 100), dtype=np.float32)
    music_stem = np.full((2, 100), 0.2, dtype=np.float32)

    bandit = FakeBandit(music_stem, calls)
    demucs = FakeDemucs(calls)
    chained = ChainedSeparator(bandit=bandit, demucs=demucs)

    result = chained.separate(mixture)

    assert set(result.keys()) == {"speech", "vocals", "drums", "bass", "other"}
    assert "music" not in result  # replaced by demucs's split
    assert "effects" not in result  # dropped per PRD's full-mode stem set
    np.testing.assert_array_equal(result["speech"], np.full((2, 100), 0.1, dtype=np.float32))
    np.testing.assert_array_equal(result["vocals"], np.full((2, 100), 1.0, dtype=np.float32))
    np.testing.assert_array_equal(result["drums"], np.full((2, 100), 2.0, dtype=np.float32))
    np.testing.assert_array_equal(result["bass"], np.full((2, 100), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(result["other"], np.full((2, 100), 4.0, dtype=np.float32))


def test_chained_separator_progress_is_monotonic_and_never_resets_across_passes():
    """Bandit runs first (3 chunks here), then Demucs (2 chunks) — the
    combined on_chunk stream must climb from 0 to bandit_total+demucs_total
    without ever going backwards when Demucs takes over."""
    calls = []
    mixture = np.zeros((2, 100), dtype=np.float32)
    music_stem = np.full((2, 100), 0.2, dtype=np.float32)

    bandit = FakeBandit(music_stem, calls, n_chunks=3)
    demucs = FakeDemucs(calls, n_chunks=2)
    chained = ChainedSeparator(bandit=bandit, demucs=demucs)

    progress_events = []
    chained.separate(mixture, on_chunk=lambda done, total: progress_events.append((done, total)))

    grand_total = 3 + 2
    assert all(total == grand_total for _done, total in progress_events)
    done_values = [done for done, _total in progress_events]
    assert done_values == sorted(done_values)  # monotonically non-decreasing throughout
    assert done_values[0] == 0
    assert done_values[-1] == grand_total
    # Demucs's first callback (done=0 within its own pass) must land at
    # bandit_total, not reset back to 0 — this is the "never resets" guarantee.
    assert 3 in done_values
    assert done_values.count(0) == 1  # only Bandit's very first callback is a 0


def test_chained_separator_progress_is_none_when_no_callback_given():
    calls = []
    mixture = np.zeros((2, 100), dtype=np.float32)
    music_stem = np.full((2, 100), 0.2, dtype=np.float32)
    bandit = FakeBandit(music_stem, calls)
    demucs = FakeDemucs(calls)
    chained = ChainedSeparator(bandit=bandit, demucs=demucs)

    result = chained.separate(mixture)  # no on_chunk — shouldn't blow up
    assert set(result.keys()) == {"speech", "vocals", "drums", "bass", "other"}
