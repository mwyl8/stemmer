"""Peak safety net applied to separator output before it's written to disk.

Root-caused via data/out/drums.wav from a real prior run: htdemucs routinely
produces samples beyond +/-1.0 on percussive stems (peak=1.00000 with 2095
samples at/above 0.999 there, out of 13M — a real, reproducible clip, not a
hypothetical one). soundfile's default WAV subtype is PCM_16, which *hard*
clips (flat-tops, not wraps) anything outside [-1, 1] on write — verified
directly: writing [0.5, 1.5, -1.8, 0.9] round-trips as
[0.5, 0.99997, -1.0, 0.89999]. A flat-topped waveform is broadband harmonic
distortion, which is exactly what "static"/harsh crackle sounds like.

`apply_peak_safety` is the single-array primitive: scale one waveform down
to `ceiling` if (and only if) it's the one that clips. Callers with more
than one stem from the *same* separation must use `apply_peak_safety_global`
instead, not call this per stem in a loop — different stems clip by
different amounts (drums routinely peak far above vocals), so scaling each
one independently by its own peak/ceiling ratio changes the *balance*
between stems: drums might get scaled by ~1/4.3 while vocals only get
~1/1.38, making drums ~10 dB too quiet relative to vocals in any mix a user
reconstructs from the downloaded stems. `apply_peak_safety_global` finds the
single loudest stem across the whole set and applies that one scale factor
to all of them, so relative levels between stems (not just within one
stem) survive.
"""

from __future__ import annotations

import numpy as np


def apply_peak_safety(wav: np.ndarray, ceiling: float = 1.0) -> tuple[np.ndarray, float, bool]:
    """Returns (safe_wav, pre_limit_peak, was_limited).

    `wav` is untouched (returned as-is) when its peak is already within
    ceiling. Otherwise every sample is scaled down uniformly by
    `ceiling / peak` — plain peak normalization, not dynamic-range
    compression — so relative levels between samples are preserved exactly,
    just brought under the ceiling PCM_16 can represent without clipping.

    This is a single-stem primitive — see the module docstring for why
    multi-stem callers need `apply_peak_safety_global` instead.
    """
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= ceiling or peak == 0.0:
        return wav, peak, False
    return wav * (ceiling / peak), peak, True


def apply_peak_safety_global(
    stems: dict[str, np.ndarray], ceiling: float = 1.0
) -> tuple[dict[str, np.ndarray], dict[str, float], bool]:
    """Returns (safe_stems, pre_limit_peaks, was_limited).

    Finds the single loudest stem across `stems`, and — only if that peak
    exceeds `ceiling` — scales every stem by that one factor
    (`ceiling / global_peak`). Stems that individually never came close to
    clipping still get scaled by the same factor as the stem that did, so
    the balance between stems (e.g. drums vs. vocals) is unchanged; only
    the overall level of the whole set is brought under the ceiling.
    """
    peaks = {name: (float(np.max(np.abs(wav))) if wav.size else 0.0) for name, wav in stems.items()}
    global_peak = max(peaks.values(), default=0.0)
    if global_peak <= ceiling or global_peak == 0.0:
        return dict(stems), peaks, False
    scale = ceiling / global_peak
    safe_stems = {name: wav * scale for name, wav in stems.items()}
    return safe_stems, peaks, True
