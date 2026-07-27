"""Sanity checks for the small SDR/SIR/SAR projection (backend/eval/metrics.py)
used by scripts/eval_singing_vs_speech.py — not a claim about any real
separator's quality, just that the metric itself behaves as a metric should.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.eval.metrics import cross_stem_leakage_fraction, sdr_sir_sar, stem_envelope_correlation, window_rms_ratio

SR = 8000
T = np.arange(SR) / SR


def _tone(freq):
    return np.sin(2 * np.pi * freq * T).astype(np.float64)


def test_perfect_estimate_scores_very_high_on_all_three():
    target = _tone(220.0)
    interferer = _tone(440.0)
    metrics = sdr_sir_sar(estimate=target.copy(), target=target, interferer=interferer)
    assert metrics["sdr_db"] > 100
    assert metrics["sir_db"] > 100
    assert metrics["sar_db"] > 100


def test_pure_interferer_scores_very_low_sir():
    target = _tone(220.0)
    interferer = _tone(440.0)
    metrics = sdr_sir_sar(estimate=interferer.copy(), target=target, interferer=interferer)
    assert metrics["sir_db"] < -60  # ~no target energy relative to interferer
    assert metrics["sdr_db"] < -60


def test_half_and_half_mixture_lands_between_the_two_extremes():
    target = _tone(220.0)
    interferer = _tone(440.0)
    clean = sdr_sir_sar(estimate=target.copy(), target=target, interferer=interferer)["sir_db"]
    mixed = sdr_sir_sar(estimate=(target + interferer), target=target, interferer=interferer)["sir_db"]
    dirty = sdr_sir_sar(estimate=interferer.copy(), target=target, interferer=interferer)["sir_db"]
    assert dirty < mixed < clean


def test_pure_artifacts_score_low_sar_but_defined_sir():
    target = _tone(220.0)
    interferer = _tone(440.0)
    noise = np.random.default_rng(0).standard_normal(len(target))
    metrics = sdr_sir_sar(estimate=noise, target=target, interferer=interferer)
    assert metrics["sar_db"] < 0


def _stereo_bursts(active_windows: list[tuple[float, float]], duration: float, sr: int, amplitude: float = 1.0) -> np.ndarray:
    """A stereo signal that's `amplitude` during each (start, end) window in
    `active_windows` and silent elsewhere -- a stand-in for a stem that's
    only "on" during specific parts of a clip."""
    mono = np.zeros(int(duration * sr), dtype=np.float64)
    for start, end in active_windows:
        mono[int(start * sr) : int(end * sr)] = amplitude
    return np.stack([mono, mono], axis=0)


def test_cross_stem_leakage_fraction_is_zero_for_mutually_exclusive_stems():
    duration = 10.0
    stem_a = _stereo_bursts([(0.0, 5.0)], duration, SR)
    stem_b = _stereo_bursts([(5.0, 10.0)], duration, SR)
    assert cross_stem_leakage_fraction(stem_a, stem_b, SR) == 0.0


def test_cross_stem_leakage_fraction_is_high_when_both_active_together():
    duration = 10.0
    stem_a = _stereo_bursts([(0.0, 10.0)], duration, SR)
    stem_b = _stereo_bursts([(0.0, 10.0)], duration, SR, amplitude=0.5)
    assert cross_stem_leakage_fraction(stem_a, stem_b, SR) == 1.0


def test_cross_stem_leakage_fraction_partial_overlap_lands_between():
    duration = 10.0
    stem_a = _stereo_bursts([(0.0, 10.0)], duration, SR)
    stem_b = _stereo_bursts([(4.0, 6.0)], duration, SR)
    fraction = cross_stem_leakage_fraction(stem_a, stem_b, SR)
    assert 0.0 < fraction < 1.0


def test_cross_stem_leakage_fraction_start_end_restricts_which_windows_count():
    duration = 10.0
    # Both stems overlap only in [4, 6); pass the FULL arrays (not
    # pre-sliced) with start/end so "own peak" is measured over the whole
    # clip, not just the queried slice -- restricting to [4, 6) should find
    # the overlap, while [0, 2) (no overlap there) should not.
    stem_a = _stereo_bursts([(0.0, 10.0)], duration, SR)
    stem_b = _stereo_bursts([(4.0, 6.0)], duration, SR)
    assert cross_stem_leakage_fraction(stem_a, stem_b, SR, start_seconds=4.0, end_seconds=6.0) == 1.0
    assert cross_stem_leakage_fraction(stem_a, stem_b, SR, start_seconds=0.0, end_seconds=2.0) == 0.0


def test_cross_stem_leakage_fraction_pre_sliced_window_hides_a_real_energy_reduction():
    """Documents the exact pitfall start_seconds/end_seconds exists to avoid.
    stem_b is genuinely loud elsewhere in the clip (its real peak) and has a
    much quieter *residual* bleed during [4, 6) -- standing in for a fix
    that reduced, but didn't eliminate, cross-stem bleed in one window."""
    duration = 10.0

    def _stem_b(bleed_amplitude: float) -> np.ndarray:
        mono = np.zeros(int(duration * SR))
        mono[: int(2 * SR)] = 1.0  # a genuinely loud, unrelated passage -- sets stem_b's real peak
        mono[int(4 * SR) : int(6 * SR)] = bleed_amplitude  # the bleed under test
        return np.stack([mono, mono], axis=0)

    stem_a = _stereo_bursts([(4.0, 6.0)], duration, SR, amplitude=1.0)
    loud_bleed = _stem_b(0.5)  # still clearly active relative to its own peak
    quiet_bleed = _stem_b(0.02)  # a fix reduced it to near-nothing relative to its own peak

    def _sliced(stem: np.ndarray, start: float, end: float) -> np.ndarray:
        return stem[..., int(start * SR) : int(end * SR)]

    # Pre-sliced: within [4, 6) alone, quiet_bleed's *local* peak IS 0.02, so
    # it reads as fully "at its own peak" there -- identical to loud_bleed.
    # The fix is invisible.
    fraction_loud_sliced = cross_stem_leakage_fraction(_sliced(stem_a, 4, 6), _sliced(loud_bleed, 4, 6), SR)
    fraction_quiet_sliced = cross_stem_leakage_fraction(_sliced(stem_a, 4, 6), _sliced(quiet_bleed, 4, 6), SR)
    assert fraction_loud_sliced == fraction_quiet_sliced == 1.0

    # start_seconds/end_seconds: "own peak" is measured over the whole
    # array, so quiet_bleed's 0.02 correctly reads as far below its own real
    # peak of 1.0 and drops out of "active" -- the fix is visible.
    fraction_full_loud = cross_stem_leakage_fraction(stem_a, loud_bleed, SR, start_seconds=4.0, end_seconds=6.0)
    fraction_full_quiet = cross_stem_leakage_fraction(stem_a, quiet_bleed, SR, start_seconds=4.0, end_seconds=6.0)
    assert fraction_full_loud == 1.0
    assert fraction_full_quiet == 0.0


def test_window_rms_ratio_high_when_target_dominates():
    duration = 10.0
    target = _stereo_bursts([(0.0, 10.0)], duration, SR, amplitude=1.0)
    other = _stereo_bursts([(4.0, 5.0)], duration, SR, amplitude=1.0 / 15.2)
    ratio = window_rms_ratio(target, other, SR, start_seconds=4.0, end_seconds=5.0)
    assert ratio == pytest.approx(15.2, rel=0.05)


def test_window_rms_ratio_is_one_when_equally_loud():
    duration = 10.0
    target = _stereo_bursts([(0.0, 10.0)], duration, SR, amplitude=1.0)
    other = _stereo_bursts([(0.0, 10.0)], duration, SR, amplitude=1.0)
    ratio = window_rms_ratio(target, other, SR, start_seconds=2.0, end_seconds=8.0)
    assert ratio == pytest.approx(1.0, rel=1e-6)


def _envelope_modulated_tone(envelope: np.ndarray, freq: float, sr: int) -> np.ndarray:
    """A stereo tone at `freq` amplitude-modulated by `envelope` (one sample
    of envelope per output sample) -- stands in for one half of a single
    performance spectrally split into a different carrier band."""
    t = np.arange(len(envelope)) / sr
    mono = (envelope * np.sin(2 * np.pi * freq * t)).astype(np.float64)
    return np.stack([mono, mono], axis=0)


def test_stem_envelope_correlation_high_when_one_performance_is_split_spectrally():
    duration = 10.0
    sr = SR
    n = int(duration * sr)
    rng = np.random.default_rng(3)
    # A single shared syllable-rate envelope (slow, ~4Hz-ish random walk of
    # bursts) driving two *different* carrier frequencies -- the spectral-
    # split failure mode: same performance, different frequency content.
    shared_envelope = np.repeat(rng.uniform(0.0, 1.0, size=40), n // 40 + 1)[:n]
    stem_a = _envelope_modulated_tone(shared_envelope, freq=220.0, sr=sr)
    stem_b = _envelope_modulated_tone(shared_envelope, freq=880.0, sr=sr)

    correlation = stem_envelope_correlation(stem_a, stem_b, sr, start_seconds=0.0, end_seconds=duration)
    assert correlation > 0.8


def test_stem_envelope_correlation_low_for_independent_sources():
    duration = 10.0
    sr = SR
    n = int(duration * sr)
    rng = np.random.default_rng(4)
    envelope_a = np.repeat(rng.uniform(0.0, 1.0, size=40), n // 40 + 1)[:n]
    envelope_b = np.repeat(rng.uniform(0.0, 1.0, size=40), n // 40 + 1)[:n]
    stem_a = _envelope_modulated_tone(envelope_a, freq=220.0, sr=sr)
    stem_b = _envelope_modulated_tone(envelope_b, freq=880.0, sr=sr)

    correlation = stem_envelope_correlation(stem_a, stem_b, sr, start_seconds=0.0, end_seconds=duration)
    assert abs(correlation) < 0.5


def test_stem_envelope_correlation_is_zero_for_true_silence():
    duration = 10.0
    stem_a = _stereo_bursts([(0.0, 10.0)], duration, SR, amplitude=1.0)
    silent = np.zeros_like(stem_a)
    correlation = stem_envelope_correlation(stem_a, silent, SR, start_seconds=0.0, end_seconds=duration)
    assert correlation == 0.0
