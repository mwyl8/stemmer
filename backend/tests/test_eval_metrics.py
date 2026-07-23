"""Sanity checks for the small SDR/SIR/SAR projection (backend/eval/metrics.py)
used by scripts/eval_singing_vs_speech.py — not a claim about any real
separator's quality, just that the metric itself behaves as a metric should.
"""

from __future__ import annotations

import numpy as np

from backend.eval.metrics import sdr_sir_sar

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
