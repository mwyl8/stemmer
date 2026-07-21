"""Peak-safety net applied to separator output before it's written as
PCM_16 WAV (see limiter.py's module docstring for the root-cause evidence:
a real prior run's drums.wav hit exactly 1.00000 peak with thousands of
samples clipped)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.separators.limiter import apply_peak_safety, apply_peak_safety_global


def test_leaves_in_range_audio_untouched():
    wav = np.array([[0.5, -0.3, 0.9]], dtype=np.float32)
    safe, peak, limited = apply_peak_safety(wav)
    assert limited is False
    assert peak == pytest.approx(0.9)
    assert np.array_equal(safe, wav)


def test_scales_down_out_of_range_peak_to_exactly_ceiling():
    wav = np.array([[0.5, -1.8, 0.9]], dtype=np.float32)
    safe, peak, limited = apply_peak_safety(wav)
    assert limited is True
    assert peak == pytest.approx(1.8)
    assert np.abs(safe).max() == pytest.approx(1.0)


def test_preserves_relative_levels_between_samples():
    wav = np.array([[1.0, -2.0, 0.5]], dtype=np.float32)
    safe, _, _ = apply_peak_safety(wav)
    # scaling is uniform, so ratios between samples survive
    assert np.isclose(safe[0, 1] / safe[0, 0], -2.0)
    assert np.isclose(safe[0, 2] / safe[0, 0], 0.5)


def test_boundary_peak_of_exactly_one_is_not_limited():
    wav = np.array([[1.0, -1.0, 0.3]], dtype=np.float32)
    safe, peak, limited = apply_peak_safety(wav)
    assert limited is False
    assert peak == 1.0
    assert np.array_equal(safe, wav)


def test_silence_is_not_limited():
    wav = np.zeros((2, 100), dtype=np.float32)
    safe, peak, limited = apply_peak_safety(wav)
    assert limited is False
    assert peak == 0.0
    assert np.array_equal(safe, wav)


def test_writing_limited_output_no_longer_clips_as_pcm16():
    """End-to-end: what used to hard-clip on write now round-trips cleanly."""
    import soundfile as sf

    wav = np.array([[0.5, 1.5, -1.8, 0.9]], dtype=np.float32)
    safe, _, limited = apply_peak_safety(wav)
    assert limited is True

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "stem.wav"
        sf.write(path, safe.T, 44100)
        back, _ = sf.read(path, dtype="float32", always_2d=True)
        # no sample sits at the PCM_16 rail from clipping — the loudest
        # original sample (1.8) maps to exactly full-scale, cleanly
        assert np.abs(back).max() <= 1.0
        assert np.isclose(np.abs(back).max(), 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# apply_peak_safety_global: multi-stem limiting must not destroy inter-stem
# balance. Found in review: runner.py was calling apply_peak_safety per stem
# in a loop, so a stem peaking at 4.32 (drums) and one peaking at 1.38
# (vocals) were each independently pulled to the same 1.0 ceiling — drums
# ended up ~10 dB too quiet relative to vocals in any mix reconstructed from
# the downloaded stems.
# ---------------------------------------------------------------------------


def test_global_leaves_stems_untouched_when_none_clip():
    stems = {
        "vocals": np.array([[0.3, -0.2]], dtype=np.float32),
        "drums": np.array([[0.5, -0.4]], dtype=np.float32),
    }
    safe, peaks, limited = apply_peak_safety_global(stems)
    assert limited is False
    assert peaks["vocals"] == pytest.approx(0.3)
    assert peaks["drums"] == pytest.approx(0.5)
    for name in stems:
        assert np.array_equal(safe[name], stems[name])


def test_global_scales_all_stems_by_the_loudest_stems_factor():
    stems = {
        "vocals": np.array([[1.0, -1.0]], dtype=np.float32),  # peak 1.0
        "drums": np.array([[2.0, -2.0]], dtype=np.float32),  # peak 2.0 -- the one that clips
    }
    safe, peaks, limited = apply_peak_safety_global(stems)
    assert limited is True
    assert peaks == {"vocals": pytest.approx(1.0), "drums": pytest.approx(2.0)}
    # both stems scaled by the SAME factor (ceiling / global_peak = 1/2)
    assert np.abs(safe["drums"]).max() == pytest.approx(1.0)
    assert np.abs(safe["vocals"]).max() == pytest.approx(0.5)
    # inter-stem balance preserved: drums was 2x vocals before, still 2x after
    assert np.abs(safe["drums"]).max() / np.abs(safe["vocals"]).max() == pytest.approx(2.0)


def test_global_preserves_balance_that_naive_per_stem_limiting_would_destroy():
    """Regression guard for the exact bug found in review: scaling each
    stem independently by its own peak (apply_peak_safety called per stem)
    forces every stem to the same ceiling regardless of how loud it started,
    destroying the original balance. apply_peak_safety_global must not do
    that — it applies one factor, derived from the loudest stem, to all."""
    stems = {
        "vocals": np.array([[1.38, -1.38]], dtype=np.float32),
        "drums": np.array([[4.32, -4.32]], dtype=np.float32),
    }
    safe, _, _ = apply_peak_safety_global(stems)
    ratio_before = 4.32 / 1.38
    ratio_after = np.abs(safe["drums"]).max() / np.abs(safe["vocals"]).max()
    assert ratio_after == pytest.approx(ratio_before)

    # contrast with the bug: naive per-stem limiting collapses both stems
    # to the same ceiling, destroying that ratio.
    naive_vocals, _, _ = apply_peak_safety(stems["vocals"])
    naive_drums, _, _ = apply_peak_safety(stems["drums"])
    assert np.abs(naive_vocals).max() == pytest.approx(1.0)
    assert np.abs(naive_drums).max() == pytest.approx(1.0)
    naive_ratio_after = np.abs(naive_drums).max() / np.abs(naive_vocals).max()
    assert naive_ratio_after == pytest.approx(1.0)  # bug: ratio destroyed


def test_global_limiting_reconstructs_original_mix_within_30db():
    """Confirmation requested during review: summing the safety-limited
    stems back together must still reconstruct the original mixture to
    within about -30 dB. Global scaling is a single overall gain change
    (it factors out of the sum), so the residual error should in practice
    land far below that bar, near float32 precision — this pins down that
    it isn't degraded to something closer to -30 dB or worse."""
    rng = np.random.default_rng(0)
    n = 44100
    vocals = (0.3 * rng.standard_normal((2, n))).astype(np.float32)
    drums = (4.32 * rng.standard_normal((2, n))).astype(np.float32)  # the loud, clipping stem
    bass = (0.2 * rng.standard_normal((2, n))).astype(np.float32)
    other = (0.15 * rng.standard_normal((2, n))).astype(np.float32)
    stems = {"vocals": vocals, "drums": drums, "bass": bass, "other": other}
    original_mix = vocals + drums + bass + other

    safe_stems, _, limited = apply_peak_safety_global(stems)
    assert limited is True  # drums peaks well above 1.0 here, so this exercises the scaling path

    scale = np.abs(safe_stems["drums"]).max() / np.abs(drums).max()
    reconstructed = sum(safe_stems.values())
    residual = reconstructed.astype(np.float64) - (original_mix * scale).astype(np.float64)

    residual_rms = np.sqrt(np.mean(residual**2))
    signal_rms = np.sqrt(np.mean((original_mix.astype(np.float64) * scale) ** 2))
    db_error = 20 * np.log10(residual_rms / signal_rms) if residual_rms > 0 else float("-inf")

    assert db_error <= -30
