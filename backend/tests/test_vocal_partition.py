"""Unit tests for the Wiener-style vocal-bus partition + frame arbitration
(backend/separators/_vocal_partition.py) that fixes Speech-vs-Singing mode's
spoken_speech/sung_vocals cross-stem bleed (see that module's docstring and
singing_sep.py's). Real speech+singing quality is measured on the cached
Thriller job by scripts/eval_singing_bleed_thriller.py -- these tests check
the partition's own invariants (no double-counting, graceful degradation,
hysteresis) with cheap synthetic signals, not separation quality.
"""

from __future__ import annotations

import numpy as np

from backend.separators import _vocal_partition as vp

SR = 44100


def _tone(freq: float, duration: float, sr: int = SR, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono], axis=0)


def test_partition_reconstructs_the_vocal_bus_within_tight_tolerance():
    # Two independent, overlapping "voices" -- exactly the hard case this
    # fixes: both are nonzero over the whole clip.
    spoken_raw = _tone(180.0, duration=3.0, amplitude=0.3)
    sung_raw = _tone(440.0, duration=3.0, amplitude=0.6)

    spoken, sung = vp.partition_vocal_bus(spoken_raw, sung_raw, SR)

    combined_in = (spoken_raw.astype(np.float64) + sung_raw.astype(np.float64))
    combined_out = spoken.astype(np.float64) + sung.astype(np.float64)
    # Perfect-reconstruction STFT/ISTFT round trip -- tight, not exact,
    # tolerance (float32 stems, windowed overlap-add).
    np.testing.assert_allclose(combined_out, combined_in, atol=1e-3)


def test_neither_stem_exceeds_the_combined_bus_energy():
    """The double-counting bug: before this fix, spoken_speech and
    sung_vocals could each independently retain close to the *full* combined
    energy at once. After partitioning, each stem's RMS must be no larger
    than the combined bus's RMS -- it can only take a share, never all of it
    twice over."""
    spoken_raw = _tone(180.0, duration=3.0, amplitude=0.3)
    sung_raw = _tone(440.0, duration=3.0, amplitude=0.6)

    spoken, sung = vp.partition_vocal_bus(spoken_raw, sung_raw, SR)

    bus_rms = np.sqrt(np.mean((spoken_raw.astype(np.float64) + sung_raw.astype(np.float64)) ** 2))
    spoken_rms = np.sqrt(np.mean(spoken.astype(np.float64) ** 2))
    sung_rms = np.sqrt(np.mean(sung.astype(np.float64) ** 2))
    assert spoken_rms <= bus_rms * 1.01
    assert sung_rms <= bus_rms * 1.01


def test_biased_mask_reduces_to_plain_ratio_mask_when_bias_is_neutral():
    rng = np.random.default_rng(0)
    raw_mask = rng.uniform(0.0, 1.0, size=(2, 10, 7))
    neutral_bias = np.full(7, 0.5)
    biased = vp._biased_mask(raw_mask, neutral_bias)
    np.testing.assert_allclose(biased, raw_mask, atol=1e-6)


def test_biased_mask_always_stays_in_zero_one_and_complements_exactly():
    rng = np.random.default_rng(1)
    raw_mask = rng.uniform(0.0, 1.0, size=(2, 10, 50))
    bias = rng.uniform(0.0, 1.0, size=50)
    mask_sung = vp._biased_mask(raw_mask, bias)
    assert np.all(mask_sung > 0.0) and np.all(mask_sung < 1.0)
    mask_spoken = 1.0 - mask_sung
    np.testing.assert_allclose(mask_sung + mask_spoken, 1.0)


def test_biased_mask_moves_toward_the_committed_decision_but_does_not_override_it():
    # A committed decision nudges a disagreeing raw ratio toward the decided
    # side, but does not fully override it at the current
    # _ARBITRATION_STRENGTH=3.0 -- raising it to force "nearly all" of a
    # disagreeing bin's energy to the decided stem was tried and reverted
    # (see that constant's docstring): a held-out real-audio window caught
    # it regressing a verified baseline, worse the higher the strength. This
    # documents the current, known-partial behavior rather than the
    # aspirational "dominates" one.
    raw_mask_sung = np.array([[[0.9]]])  # raw ratio favors "sung"
    neutral = vp._biased_mask(raw_mask_sung, np.array([0.5]))[0, 0, 0]
    committed_spoken = vp._biased_mask(raw_mask_sung, np.array([0.0]))[0, 0, 0]
    assert committed_spoken < neutral  # moved toward spoken...
    assert committed_spoken > 0.1  # ...but a strongly disagreeing raw ratio still keeps a real share


def test_short_audio_defaults_to_a_single_stem_not_a_neutral_split():
    # Shorter than _PITCH_FRAME_LENGTH -- must skip librosa.pyin entirely
    # rather than crash on too-short input (this is exactly what the fast
    # test_singing_sep.py fakes exercise: 100-sample arrays). The fallback
    # must default to a single stem (spoken), not the neutral 0.5 that
    # reproduces the plain, unarbitrated magnitude-ratio mask -- that
    # neutral default is exactly the "50/50 split of one voice" bug.
    spoken_raw = np.full((2, 100), 0.1, dtype=np.float32)
    sung_raw = np.full((2, 100), 1.0, dtype=np.float32)

    bias = vp._frame_sung_bias(spoken_raw, sung_raw, SR, n_frames=5)
    np.testing.assert_array_equal(bias, np.zeros(5))

    spoken, sung = vp.partition_vocal_bus(spoken_raw, sung_raw, SR)
    assert spoken.shape == (2, 100)
    assert sung.shape == (2, 100)
    assert np.all(np.isfinite(spoken)) and np.all(np.isfinite(sung))

    # The single-stem ("spoken") default should shift *some* energy toward
    # spoken relative to the neutral-0.5 fallback it replaced -- even though
    # sung_raw's raw amplitude is ~10x spoken_raw's here, so the raw
    # magnitude ratio alone still wins this bin at the current, reverted
    # _ARBITRATION_STRENGTH (see that constant's docstring: a stronger
    # override was tried and rolled back after regressing real audio).
    raw_mask_sung = np.array([[[0.99]]])  # close to what this DC-ish input actually produces
    mask_sung_neutral = vp._biased_mask(raw_mask_sung, np.array([0.5]))[0, 0, 0]
    mask_sung_spoken_default = vp._biased_mask(raw_mask_sung, np.array([0.0]))[0, 0, 0]
    assert mask_sung_spoken_default < mask_sung_neutral


def test_frame_sung_bias_with_accompaniment_does_not_crash_and_stays_in_range():
    duration = 3.0
    spoken_raw = _tone(180.0, duration=duration, amplitude=0.3)
    sung_raw = _tone(440.0, duration=duration, amplitude=0.6)
    accompaniment = _tone(220.0, duration=duration, amplitude=0.4)  # stand-in instrument bed

    n_frames = 50
    bias = vp._frame_sung_bias(spoken_raw, sung_raw, SR, n_frames=n_frames, accompaniment=accompaniment)
    assert bias.shape == (n_frames,)
    assert np.all(np.isfinite(bias))
    assert np.all((bias >= 0.0) & (bias <= 1.0))


def test_rolling_trailing_max_holds_a_spike_forward_not_backward():
    x = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    held = vp._rolling_trailing_max(x, window=3)
    np.testing.assert_array_equal(held, [0.0, 0.0, 1.0, 1.0, 1.0, 0.0])


def test_match_length_axis_pads_and_truncates_along_given_axis():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(vp._match_length_axis(arr, 5, axis=1), [[1.0, 2.0, 3.0, 3.0, 3.0], [4.0, 5.0, 6.0, 6.0, 6.0]])
    np.testing.assert_array_equal(vp._match_length_axis(arr, 2, axis=1), [[1.0, 2.0], [4.0, 5.0]])


def test_hysteresis_smooth_does_not_chatter_on_noisy_borderline_score():
    rng = np.random.default_rng(2)
    midpoint = (vp._HYSTERESIS_HIGH + vp._HYSTERESIS_LOW) / 2
    # A score that hovers right at the naive midpoint -- without hysteresis
    # this would flip almost every frame. All frames "voiced" (informative)
    # so hysteresis logic, not the voicing gate, is what's under test.
    score = midpoint + 0.05 * rng.standard_normal(500)
    voiced = np.ones(500, dtype=bool)
    bias = vp._hysteresis_smooth(score, voiced)

    naive_flips = np.sum(np.diff((score > midpoint).astype(int)) != 0)
    smoothed_flips = np.sum(np.diff((bias > midpoint).astype(int)) != 0)
    assert smoothed_flips < naive_flips


def test_hysteresis_smooth_ignores_unvoiced_frames_mid_note():
    # A held note (score above HIGH) with a brief unvoiced dip well after
    # entry has already committed (a consonant, a breath, a real
    # pitch-tracker dropout) must not flip the decision back to spoken --
    # that flicker was a real, measured regression on real audio (see
    # _HYSTERESIS_HIGH/_LOW's docstring comment).
    n = 80
    score = np.full(n, vp._HYSTERESIS_HIGH + 0.1)
    voiced = np.ones(n, dtype=bool)
    voiced[40:45] = False  # dip, well after the note should already be committed
    bias = vp._hysteresis_smooth(score, voiced)
    assert np.all(bias[30:] > 0.5)  # entry needs _ENTER_DEBOUNCE_FRAMES consecutive frames to commit


def test_hysteresis_smooth_does_not_enter_sung_on_a_single_frame_spike():
    # One frame above HIGH surrounded by low score must not be enough to
    # commit to "sung" -- _ENTER_DEBOUNCE_FRAMES of consecutive evidence is
    # required, exactly the debounce that fixed the real-audio regression.
    score = np.full(60, vp._HYSTERESIS_LOW - 0.05)
    score[30] = vp._HYSTERESIS_HIGH + 0.1
    voiced = np.ones(60, dtype=bool)
    bias = vp._hysteresis_smooth(score, voiced)
    assert np.all(bias < 0.5)


def test_hysteresis_smooth_tracks_a_real_transition():
    score = np.concatenate([np.full(100, 0.9), np.full(100, 0.1)])
    voiced = np.ones(200, dtype=bool)
    bias = vp._hysteresis_smooth(score, voiced)
    # Entry/exit each need their own debounce run of consecutive evidence, so
    # the very first few frames of each run are still settling -- check well
    # into each run instead of right at the boundary.
    assert bias[50] > 0.9
    assert bias[-1] < 0.1


def test_match_length_pads_and_truncates():
    arr = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(vp._match_length(arr, 3), arr)
    np.testing.assert_array_equal(vp._match_length(arr, 5), [1.0, 2.0, 3.0, 3.0, 3.0])
    np.testing.assert_array_equal(vp._match_length(arr, 2), [1.0, 2.0])


def test_fill_nan_1d_interpolates_gaps():
    x = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    filled = vp._fill_nan_1d(x)
    assert not np.any(np.isnan(filled))
    np.testing.assert_allclose(filled, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
