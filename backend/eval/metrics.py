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

`cross_stem_leakage_fraction`/`window_rms_ratio` below are a second, blunter
measure that doesn't need ground truth at all — just two stems that are
*supposed* to be mutually exclusive in time (like spoken_speech vs.
sung_vocals). Used by scripts/eval_singing_bleed_thriller.py to quantify the
bleed bug directly on a real cached job (no synthetic fixture involved),
before vs. after _vocal_partition.py's fix.

`stem_envelope_correlation` distinguishes the two ways "both stems active at
once" can happen, which `cross_stem_leakage_fraction` alone cannot: one
shared performance split spectrally across both stems (high envelope
correlation — a bug) vs. two genuinely different, independently-modulated
sources that merely co-occur (low correlation — not a bug).
"""

from __future__ import annotations

import numpy as np

_RMS_EPS = 1e-10


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


def cross_stem_leakage_fraction(
    stem_a: np.ndarray,
    stem_b: np.ndarray,
    sample_rate: int,
    frame_seconds: float = 0.5,
    active_ratio: float = 0.1,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> float:
    """Fraction of `frame_seconds` windows where BOTH stems are "active" at
    once — the direct measure of "did the two voices bleed into each
    other's stem". A window counts as active for a stem when that window's
    RMS is at least `active_ratio` of *that same stem's own peak RMS* across
    the whole clip (so a naturally quiet stem — e.g. sparse dialogue — isn't
    unfairly counted as silent throughout). 0.0 means the two stems never
    overlap in time; 1.0 means they're active together everywhere.

    `start_seconds`/`end_seconds`, if given, restrict which windows are
    *counted* toward the fraction — but the "own peak RMS" each window is
    compared against is still measured over the full `stem_a`/`stem_b`
    arrays passed in. Passing already-sliced arrays instead (and no
    start/end) silently changes the denominator to that slice's own local
    peak, which washes out any real difference a fix makes to a specific
    window — every slice trivially has *some* frame at its own local peak,
    so the fraction stops responding to what actually changed there."""
    frame_len = max(int(frame_seconds * sample_rate), 1)
    rms_a = _frame_rms(stem_a, frame_len)
    rms_b = _frame_rms(stem_b, frame_len)
    n = min(len(rms_a), len(rms_b))
    if n == 0:
        return 0.0
    rms_a, rms_b = rms_a[:n], rms_b[:n]
    active_a = rms_a >= active_ratio * max(float(rms_a.max()), _RMS_EPS)
    active_b = rms_b >= active_ratio * max(float(rms_b.max()), _RMS_EPS)
    both_active = active_a & active_b
    start_frame = int((start_seconds * sample_rate) / frame_len) if start_seconds is not None else 0
    end_frame = int((end_seconds * sample_rate) / frame_len) if end_seconds is not None else n
    windowed = both_active[max(start_frame, 0) : min(end_frame, n)]
    return float(np.mean(windowed)) if windowed.size else 0.0


def stem_envelope_correlation(
    stem_a: np.ndarray,
    stem_b: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    frame_seconds: float = 0.05,
) -> float:
    """Pearson correlation, over [start_seconds, end_seconds), between
    `stem_a` and `stem_b`'s short-frame RMS *envelopes* -- not the raw
    waveforms, and not `cross_stem_leakage_fraction`'s coarse on/off co-
    activity check.

    The failure mode this exists to catch: a single vocal performance
    getting divided *spectrally* between two stems (one keeps the
    consonant/formant energy, the other the tonal/harmonic energy) rather
    than being routed whole to one of them. Both halves of a spectral split
    rise and fall in lockstep with the same syllables and notes, so their
    envelopes correlate strongly (>0.8 in practice) even though the raw
    waveforms look quite different (different frequency content) -- exactly
    the case `cross_stem_leakage_fraction` cannot distinguish from two
    genuinely different voices that simply happen to co-occur, since it only
    checks whether both stems are "active" in a window, not whether their
    activity is *the same event*. Two independent sources active in the same
    window (real co-occurrence, not duplication) have independently
    modulated envelopes and correlate weakly. Frame size defaults to 50ms --
    fine enough to track syllable-rate envelope changes, coarse enough to be
    a stable statistic (unlike sample-rate raw cross-correlation, which is
    dominated by phase/frequency mismatch between the two spectral halves).

    Returns 0.0 (no evidence of duplication) if either envelope is constant
    over the window (e.g. true silence), since Pearson correlation is
    undefined there.
    """
    frame_len = max(int(frame_seconds * sample_rate), 1)
    env_a = _frame_rms(_slice_seconds(stem_a, sample_rate, start_seconds, end_seconds), frame_len)
    env_b = _frame_rms(_slice_seconds(stem_b, sample_rate, start_seconds, end_seconds), frame_len)
    n = min(len(env_a), len(env_b))
    if n < 2:
        return 0.0
    env_a, env_b = env_a[:n], env_b[:n]
    if env_a.std() < _RMS_EPS or env_b.std() < _RMS_EPS:
        return 0.0
    return float(np.clip(np.corrcoef(env_a, env_b)[0, 1], -1.0, 1.0))


def window_rms_ratio(target: np.ndarray, other: np.ndarray, sample_rate: int, start_seconds: float, end_seconds: float) -> float:
    """RMS(target) / RMS(other) over [start_seconds, end_seconds) — how many
    times louder the stem that *should* be active in this window is than the
    stem that shouldn't be. Higher is better; 1.0 means the two stems are
    equally loud in a window that should belong to only one of them."""
    a = _slice_seconds(target, sample_rate, start_seconds, end_seconds)
    b = _slice_seconds(other, sample_rate, start_seconds, end_seconds)
    return float(_rms(a) / max(_rms(b), _RMS_EPS))


def _frame_rms(stem: np.ndarray, frame_len: int) -> np.ndarray:
    mono = np.mean(stem, axis=0) if stem.ndim > 1 else stem
    n_frames = mono.shape[-1] // frame_len
    if n_frames == 0:
        return np.array([_rms(mono)])
    trimmed = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))


def _slice_seconds(stem: np.ndarray, sample_rate: int, start_seconds: float, end_seconds: float) -> np.ndarray:
    start = max(int(start_seconds * sample_rate), 0)
    end = min(int(end_seconds * sample_rate), stem.shape[-1])
    return stem[..., start:end]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
