"""Synthetic Lead-vs-Backing-Vocals fixture generator (backend/eval/fixtures.py):
correct shapes/rate, the mixture really is the sum of its parts, the backing
harmony is genuinely silent during the lead-only section, and it's
deterministic given a seed (scripts/eval_lead_vs_backing.py's report needs
to be reproducible run-to-run). Structural correctness only — whether the
real karaoke model routes this fixture's synthetic tones sensibly is a
separate, documented finding (see that script's module docstring).
"""

from __future__ import annotations

import numpy as np

from backend.eval.fixtures import make_lead_vs_backing_fixture


def test_fixture_shapes_and_sample_rate():
    fixture = make_lead_vs_backing_fixture(lead_only_seconds=1.0, harmony_seconds=1.0, sr=8000, seed=0)
    expected_samples = 8000 * 2
    assert fixture.sample_rate == 8000
    for track in (fixture.mixture, fixture.lead_vocal, fixture.backing_vocals, fixture.instruments):
        assert track.shape == (2, expected_samples)
        assert track.dtype == np.float32


def test_mixture_is_the_sum_of_its_ground_truth_parts():
    fixture = make_lead_vs_backing_fixture(lead_only_seconds=1.0, harmony_seconds=1.0, sr=8000, seed=1)
    expected = fixture.lead_vocal + fixture.backing_vocals + fixture.instruments
    np.testing.assert_allclose(fixture.mixture, expected, atol=1e-6)


def test_backing_vocals_silent_during_lead_only_section():
    fixture = make_lead_vs_backing_fixture(lead_only_seconds=1.0, harmony_seconds=1.0, sr=8000, seed=0)
    lead_only_samples = int(fixture.lead_only_seconds * fixture.sample_rate)
    assert np.all(fixture.backing_vocals[:, :lead_only_samples] == 0.0)


def test_backing_vocals_present_during_harmony_section():
    fixture = make_lead_vs_backing_fixture(lead_only_seconds=1.0, harmony_seconds=1.0, sr=8000, seed=0)
    lead_only_samples = int(fixture.lead_only_seconds * fixture.sample_rate)
    assert np.any(fixture.backing_vocals[:, lead_only_samples:] != 0.0)


def test_lead_vocal_present_throughout_both_sections():
    fixture = make_lead_vs_backing_fixture(lead_only_seconds=1.0, harmony_seconds=1.0, sr=8000, seed=0)
    lead_only_samples = int(fixture.lead_only_seconds * fixture.sample_rate)
    assert np.sum(fixture.lead_vocal[:, :lead_only_samples] ** 2) > 0
    assert np.sum(fixture.lead_vocal[:, lead_only_samples:] ** 2) > 0


def test_deterministic_given_same_seed():
    a = make_lead_vs_backing_fixture(lead_only_seconds=1.0, harmony_seconds=1.0, sr=8000, seed=42)
    b = make_lead_vs_backing_fixture(lead_only_seconds=1.0, harmony_seconds=1.0, sr=8000, seed=42)
    np.testing.assert_array_equal(a.mixture, b.mixture)
