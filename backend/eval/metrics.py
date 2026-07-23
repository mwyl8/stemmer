"""Small BSS-eval-style separation metrics (SDR/SIR/SAR) against known
ground-truth sources — a short least-squares projection (the two-term case
of Vincent et al. 2006's decomposition), not a mir_eval/museval dependency,
since this codebase is otherwise open-source-ML-only and these three
numbers don't need a whole library.

Used by scripts/eval_singing_vs_speech.py to report how cleanly Speech-vs-
Singing mode's spoken_speech/sung_vocals stems separate a known synthetic
mixture — see that script for why the fixture is synthetic, and its printed
report for why the numbers should be read as a regression check, not a
solved-problem claim.
"""

from __future__ import annotations

import numpy as np


def project_two_sources(
    estimate: np.ndarray, target: np.ndarray, interferer: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares decomposition of `estimate` onto {target, interferer}:
    estimate ~= a*target + b*interferer + artifacts. Returns
    (target_component, interferer_component, artifacts), each the same shape
    as `estimate`. Restricted to the two sources we actually know about here
    (rather than a full basis over every source in the mix) — enough to
    answer "how much of the *other known voice* bled into this stem"."""
    e = estimate.reshape(-1).astype(np.float64)
    t = target.reshape(-1).astype(np.float64)
    i = interferer.reshape(-1).astype(np.float64)
    basis = np.stack([t, i], axis=1)
    (a, b), *_ = np.linalg.lstsq(basis, e, rcond=None)
    target_component = (a * t).reshape(estimate.shape)
    interferer_component = (b * i).reshape(estimate.shape)
    artifacts = estimate - target_component - interferer_component
    return target_component, interferer_component, artifacts


def _db(numerator: float, denominator: float, eps: float = 1e-10) -> float:
    return float(10 * np.log10((numerator + eps) / (denominator + eps)))


def sdr_sir_sar(estimate: np.ndarray, target: np.ndarray, interferer: np.ndarray) -> dict[str, float]:
    """SDR: target energy vs. everything else (interference + artifacts) —
    overall distortion. SIR: target energy vs. specifically the known
    interferer's leaked energy — directly "how much did the other voice
    bleed in". SAR: (target+interference) vs. whatever's left over — model
    artifacts attributable to neither known source."""
    target_component, interferer_component, artifacts = project_two_sources(estimate, target, interferer)
    target_energy = float(np.sum(target_component**2))
    interferer_energy = float(np.sum(interferer_component**2))
    artifact_energy = float(np.sum(artifacts**2))
    return {
        "sdr_db": _db(target_energy, interferer_energy + artifact_energy),
        "sir_db": _db(target_energy, interferer_energy),
        "sar_db": _db(target_energy + interferer_energy, artifact_energy),
    }
