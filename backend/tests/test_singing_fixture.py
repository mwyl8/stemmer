"""Synthetic Speech-vs-Singing fixture generator (backend/eval/fixtures.py):
correct shapes/rate, the mixture really is the sum of its parts, and it's
deterministic given a seed (scripts/eval_singing_vs_speech.py's report needs
to be reproducible run-to-run).
"""

from __future__ import annotations

import numpy as np

from backend.eval.fixtures import make_speech_vs_singing_fixture


def test_fixture_shapes_and_sample_rate():
    fixture = make_speech_vs_singing_fixture(duration=1.0, sr=8000, seed=0)
    expected_samples = 8000
    assert fixture.sample_rate == 8000
    for track in (fixture.mixture, fixture.spoken_speech, fixture.sung_vocals, fixture.instruments):
        assert track.shape == (2, expected_samples)
        assert track.dtype == np.float32


def test_mixture_is_the_sum_of_its_ground_truth_parts():
    fixture = make_speech_vs_singing_fixture(duration=1.0, sr=8000, seed=1)
    expected = fixture.spoken_speech + fixture.sung_vocals + fixture.instruments
    np.testing.assert_allclose(fixture.mixture, expected, atol=1e-6)


def test_deterministic_given_same_seed():
    a = make_speech_vs_singing_fixture(duration=1.0, sr=8000, seed=42)
    b = make_speech_vs_singing_fixture(duration=1.0, sr=8000, seed=42)
    np.testing.assert_array_equal(a.mixture, b.mixture)


def test_different_seeds_differ():
    a = make_speech_vs_singing_fixture(duration=1.0, sr=8000, seed=1)
    b = make_speech_vs_singing_fixture(duration=1.0, sr=8000, seed=2)
    assert not np.array_equal(a.mixture, b.mixture)


def test_both_voices_carry_real_energy_throughout_the_clip():
    """Both voices overlap for the whole clip (the harder eval case) — not
    just present somewhere, but present in both halves."""
    fixture = make_speech_vs_singing_fixture(duration=2.0, sr=8000, seed=0)
    midpoint = fixture.spoken_speech.shape[-1] // 2
    for track in (fixture.spoken_speech, fixture.sung_vocals):
        first_half_energy = np.sum(track[:, :midpoint] ** 2)
        second_half_energy = np.sum(track[:, midpoint:] ** 2)
        assert first_half_energy > 0
        assert second_half_energy > 0
