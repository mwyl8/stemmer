"""Wiener-style energy partition + frame-level arbitration for Speech-vs-
Singing mode's two vocal stems (singing_sep.py).

The bug this fixes: Bandit's "speech" stem and Demucs's "vocals" stem are
each estimated independently — Bandit runs on the whole mixture; Demucs runs
on whatever's left of the *music* bus after Bandit's own estimate is
subtracted out (_two_stage_chain.run_chain). Neither model was ever trained
on "is this frame spoken or sung", only on its own source-type question, so
a belted or half-spoken line routinely gets claimed by *both* models at
once, each at close to full strength — nothing enforced that spoken_speech
and sung_vocals partition one shared vocal budget, so the same energy shows
up in both stems simultaneously (observed directly in the cached Thriller
job, 5:50-6:30).

Fixed in two layers:

1. `partition_vocal_bus` — treat spoken_raw + sung_raw's complex STFT sum as
   the one "vocal bus" both stems draw from, and split every time-frequency
   bin between them with a soft mask (squared-magnitude ratio, i.e. a Wiener
   filter) that always sums to exactly 1. By construction,
   spoken_stft + sung_stft == vocal_bus in every bin, so neither stem can
   ever claim more than its share — no double-counting, whatever the two
   upstream models originally guessed.

2. `_frame_sung_bias` — the raw per-bin ratio above is model-blind: it only
   says which raw estimate was louder in that bin, not which *voice* it
   actually is. A cheap classical pitch/harmonicity classifier (librosa's
   pYIN autocorrelation-based F0 tracker + a handful of hand features — no
   trained model, see module docstring of `_sung_score`) nudges the split
   frame-by-frame toward "sung" when a frame looks like a held, vibrato'd,
   in-scale note, and toward "spoken" otherwise. The nudge is applied in
   logit space (`_biased_mask`) so layer 1's sum-to-1 guarantee holds no
   matter what the classifier decides — arbitration can shift *where* the
   line falls, never break the partition.

Second bug, found by listening to a real job past this fix (job
c226c717..., 5:20-6:30): the two stems can each carry the *same* vocal
performance at once, split by frequency content rather than routed to one
of them — spoken_speech keeps the consonant/formant energy, sung_vocals
keeps the tonal/harmonic energy, same words, same timing, one voice
smeared across both outputs. Confirmed before changing anything:
`_frame_sung_bias`'s raw per-frame score is noisy frame-to-frame (big
swings between confidently-low and confidently-high even inside a single
verified-pure window), and — the real culprit — `_ARBITRATION_STRENGTH`'s
old value (3.0) was too weak to override a bin where the raw magnitude
ratio already disagreed with the frame's committed decision: a bin at
raw_mask_sung=0.9 under a fully-committed "spoken" shift of -3.0 logits
still ended up ~31% sung, not "nearly all spoken" — so even a *correct,
confident, stable* decision left a substantial, audible fraction of the
vocal in the wrong stem, every frame, which is what the ear picks up as
duplication. Separately, the classifier's committed hysteresis state was
measurably *wrong* (opposite of the verified label) for over 40% of
frame-time in both of the previously "clean" baseline windows — the
15-20x RMS ratios those windows passed at were propped up by the raw
models' own energy imbalance, not by the classifier being right, which is
why raising `_ARBITRATION_STRENGTH` alone isn't the whole fix (see point 3
below).

Two changes address parts of this; a third was tried and reverted:
- `_frame_sung_bias`'s too-short-audio fallback no longer returns a neutral
  0.5 (which reproduces the plain, unarbitrated magnitude-ratio mask —
  exactly the "50/50 split of one voice" failure mode) — it now defaults
  to a single stem (spoken), matching `_hysteresis_smooth`'s own
  before-any-evidence default.
- `_sung_score` gets two new features that don't depend on pYIN's voiced-
  probability confidence (which collapses on a dense mix, per the
  original bug's root cause): beat-grid onset alignment and harmonic fit
  against the accompaniment's chroma, both computed from the
  `accompaniment` stem (the non-vocal Demucs sources) when the caller
  supplies one — sung phrasing tends to lock to both the beat and the
  underlying chord, spoken delivery doesn't. Measured effect on real audio
  is modest and mixed (better on the Price-monologue window, roughly a
  wash on the sung-verse one) — kept because it's not harmful and CPU-cheap
  (~7% of pYIN's own cost), not because it decisively fixes the bug.
- Raising `_ARBITRATION_STRENGTH` (from 3.0 toward the 15.0 the arithmetic
  above calls for) was tried and reverted: a held-out validation window
  (never used to tune anything, see scripts/eval_singing_bleed_thriller.py)
  caught it *regressing* the Price-monologue floor, and a controlled sweep
  confirmed the regression is monotonic across the whole 1-15 range on
  both verified windows, with no corresponding improvement to the bleed
  region's stem-envelope correlation at any strength tested. See
  `_ARBITRATION_STRENGTH`'s own comment for why: a committed hysteresis
  state signals *stability*, not *correctness*, and this classifier is
  still wrong 33-42% of frame-time on unambiguous ground truth — trusting
  it harder amplifies its mistakes faster than its correct calls. The
  original bleed report (5:20-6:30, correlation ~-0.17) is therefore
  **not resolved** by this round of changes; it needs a materially more
  accurate frame classifier before hard routing is safe to enable.

Uses its own small STFT/ISTFT pair rather than `_stft_numpy.py`'s (the
htdemucs-oracle-matching one): that one reflect-pads to center each frame,
which requires the pad width (n_fft // 2) to be <= the signal length — true
for any real audio clip, but not for this module's own fast unit tests
(hundred-sample fakes). This module isn't checked against a PyTorch oracle,
so there's no reason to inherit that constraint; zero-padding instead keeps
the same windowed-overlap-add math correct at any length. Only the forward
STFT differs — `_istft` (imported) has no such restriction and is reused
as-is.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.signal.windows import hann

from backend.separators._stft_numpy import _istft

N_FFT = 2048
HOP_LENGTH = 512  # ~11.6ms @ 44.1kHz: fine enough to resolve vibrato (4-8Hz)
# and syllable rate (~4Hz), coarse enough to keep the pitch tracker cheap over a whole song.

_FMIN_HZ = 65.41  # C2 -- low end of chest voice / spoken fundamental
_FMAX_HZ = 1046.50  # C6 -- top of soprano belting range
_PITCH_FRAME_LENGTH = 4096  # long enough for pYIN to resolve _FMIN_HZ's ~674-sample period

_STABILITY_WINDOW_FRAMES = 9  # ~104ms of context, for pitch-drift/discreteness features
_STABILITY_SLOPE_SCALE = 0.05  # semitones/frame; drift much faster than this reads as speech, not a held note
_DISCRETENESS_CENTS_SCALE = 35.0  # cents of tolerance around the nearest equal-tempered semitone

_VIBRATO_WINDOW_FRAMES = 25  # ~290ms -- long enough to resolve the 4-8Hz vibrato band at ~3.4Hz bins
_VIBRATO_MIN_HZ, _VIBRATO_MAX_HZ = 4.0, 8.0  # typical singing vibrato rate
_VIBRATO_RATIO_SCALE = 0.5

_MIN_NOTE_FRAMES = 13  # ~150ms -- below this, a "stable pitch" run reads as a fast speech inflection

# Calibrated against real (not synthetic) audio: on a cached real take with
# known-good spoken/sung windows, _sung_score's *voiced* frames (unvoiced
# ones are excluded from hysteresis entirely -- see _hysteresis_smooth)
# median'd 0.38 in a verified sung passage vs. 0.27 in a verified spoken one
# -- real pYIN confidence on a dense, produced mix runs much lower than on
# clean synthetic tones, so thresholds tuned against synthetic fixtures
# alone (originally 0.6/0.4) never fired on real content and let raw
# magnitude-ratio evidence get overridden by a near-constant "spoken" bias
# instead of a genuine per-frame decision -- a real, measured regression on
# the verified sung-verse window this module exists to protect
# (scripts/eval_singing_bleed_thriller.py) until this recalibration.
_HYSTERESIS_HIGH, _HYSTERESIS_LOW = 0.33, 0.28

# Also measured on that same real take: a single voiced frame crossing
# _HYSTERESIS_HIGH is common mid-phrase on real audio (the per-frame score is
# noisy), so entering "sung" needs a longer run of sustained evidence
# (_ENTER_DEBOUNCE_FRAMES) than exiting it needs of sustained
# counter-evidence (_EXIT_DEBOUNCE_FRAMES) -- a *symmetric* debounce here
# measurably overcorrected: it fixed chatter inside genuine held notes, but
# then let a rare false "sung" entry during real speech (a theatrically
# sustained vowel, e.g. the Vincent Price monologue) get locked in for just
# as long before it could exit, which is a worse failure mode for a stem
# that's supposed to default to "spoken". Grid-searched against the same
# real take's Price-monologue/sung-verse windows for the enter/exit pair
# that both cleared their required floors (scripts/eval_singing_bleed_thriller.py's
# 15.2x/12x) with the most margin.
_ENTER_DEBOUNCE_FRAMES = 20  # ~232ms of sustained above-HIGH evidence to commit to "sung"
_EXIT_DEBOUNCE_FRAMES = 9  # ~104ms of sustained below-LOW evidence to fall back to "spoken"

_CROSSFADE_FRAMES = 5  # ~58ms ramp at a spoken/sung transition -- short enough not to blur real transitions

# Logit-domain shift magnitude, see _biased_mask. Tried raising this to 15.0
# (from 3.0) to make a *committed* decision route "nearly all" of a bin's
# energy to the decided stem even when the raw magnitude ratio disagrees --
# in isolation that arithmetic is correct (a bin at raw_mask_sung=0.9 under a
# committed "spoken" shift of -3.0 only reaches mask_sung=0.31; at -15.0 it
# reaches 0.0007). But a controlled sweep against real audio (job
# c226c717..., held-out-validated per this module's own history of
# regressing on real audio after synthetic-only tuning) showed raising this
# constant *monotonically worsens* both verified windows (Price monologue
# 4.94x -> 2.88x, sung verse 2.65x -> 1.93x at strength 3 -> 15) while the
# bleed region's stem-envelope correlation barely moves either way (-0.176
# to -0.190 across the whole 1-15 range) -- because `_hysteresis_smooth`
# committing to a state measures *stability*, not *correctness*: the
# classifier is still wrong 33-42% of frame-time even on these unambiguous
# ground-truth windows (see _sung_score's docstring), so trusting a
# committed decision harder amplifies its mistakes faster than it rewards
# its correct calls. Raising this constant is only safe once the frame
# classifier's real-world accuracy is much higher than that -- left at its
# original value pending that, with the mistake (and the sweep that caught
# it) documented here rather than silently dropped.
_ARBITRATION_STRENGTH = 3.0
_EPS = 1e-8

# Beat-grid alignment feature (see _rolling_beat_alignment): sung phrasing
# tends to place syllable onsets close to the beat; spoken delivery doesn't.
# pYIN's own voiced-probability confidence collapses on a dense mix (the
# original bug's root cause), so this feature is deliberately independent of
# it -- it only needs the accompaniment's rhythm and the vocal's own onset
# envelope, neither of which touches pitch tracking.
_BEAT_ALIGNMENT_TOLERANCE_SECONDS = 0.07  # ~70ms -- typical onset-detection jitter around a true beat
_BEAT_ALIGNMENT_HOLD_SECONDS = 0.2  # sustain an onset's alignment credit through a held note, not just its attack

# Harmony-fit feature (see _rolling_harmony_fit): a sung note's F0 usually
# lands on a chord tone of whatever the accompaniment is playing; a spoken
# line's incidental pitch inflections don't track harmony at all, so its
# fit against the underlying chroma should hover near chance (1/12 of the
# accompaniment's chroma energy, if pitch class were unrelated to harmony).
_HARMONY_FIT_SCALE = 0.30  # rescales the raw chroma-fraction-at-pitch-class into a saturating [0, 1] score


def partition_vocal_bus(
    spoken_raw: np.ndarray, sung_raw: np.ndarray, sample_rate: int, accompaniment: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Splits (spoken_raw + sung_raw)'s combined STFT energy between the two
    stems with a mask that sums to 1 per time-frequency bin, biased
    per-frame by `_frame_sung_bias`. Returns (spoken, sung), each shaped and
    dtyped like the inputs.

    `accompaniment` (the non-vocal instrument bus, same shape/rate as the two
    vocal estimates) is optional and only powers the beat-alignment/harmony-
    fit features in `_sung_score` -- omitting it (as the unit tests do)
    falls back to the original pitch-only feature set, just with no rhythm
    or harmony evidence available."""
    length = spoken_raw.shape[-1]
    spoken_stft = _stft_zero_padded(spoken_raw, N_FFT, HOP_LENGTH)
    sung_stft = _stft_zero_padded(sung_raw, N_FFT, HOP_LENGTH)
    vocal_bus = spoken_stft + sung_stft

    power_spoken = np.abs(spoken_stft) ** 2
    power_sung = np.abs(sung_stft) ** 2
    raw_mask_sung = power_sung / np.maximum(power_spoken + power_sung, _EPS)

    bias = _frame_sung_bias(spoken_raw, sung_raw, sample_rate, n_frames=raw_mask_sung.shape[-1], accompaniment=accompaniment)
    mask_sung = _biased_mask(raw_mask_sung, bias)
    mask_spoken = 1.0 - mask_sung

    spoken = _istft(mask_spoken * vocal_bus, N_FFT, HOP_LENGTH, length).astype(np.float32)
    sung = _istft(mask_sung * vocal_bus, N_FFT, HOP_LENGTH, length).astype(np.float32)
    return spoken, sung


def _biased_mask(raw_mask_sung: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Nudges `raw_mask_sung` (per-bin power ratio, shape (C, F, T)) toward
    `bias` (per-frame sung-likelihood in [0, 1], shape (T,)) in logit space,
    so the result always stays in (0, 1) and `1 - mask` is its exact
    complement -- the partition-of-unity guarantee in `partition_vocal_bus`
    holds regardless of what the classifier says. `bias == 0.5` everywhere
    (the neutral case, e.g. audio too short to pitch-track) reproduces the
    plain magnitude-ratio mask exactly."""
    clipped = np.clip(raw_mask_sung, _EPS, 1.0 - _EPS)
    logit = np.log(clipped / (1.0 - clipped))
    shift = _ARBITRATION_STRENGTH * (2.0 * bias - 1.0)
    biased_logit = logit + shift[None, None, :]
    return 1.0 / (1.0 + np.exp(-biased_logit))


def _frame_sung_bias(
    spoken_raw: np.ndarray,
    sung_raw: np.ndarray,
    sample_rate: int,
    n_frames: int,
    accompaniment: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame sung-likelihood in [0, 1], aligned to `partition_vocal_bus`'s
    STFT frame grid, smoothed with hysteresis so the decision doesn't
    chatter frame to frame. Defaults to a single stem (spoken, i.e. 0.0
    everywhere) when the clip is too short for the pitch tracker's analysis
    window -- real jobs are always many seconds long, but this module's unit
    tests exercise it with few-hundred-sample fakes. (Not 0.5: a neutral
    bias reproduces the plain, unarbitrated magnitude-ratio mask, which is
    exactly the "50/50 split of one voice" bug this module exists to avoid
    -- an uncertain default should pick a stem, not divide the bin.)

    `accompaniment`, if given, feeds `_sung_score`'s beat-alignment/harmony-
    fit features (see that function) -- both computed from the non-vocal
    instrument bus, not from the vocal itself, so they stay informative even
    where pYIN's own confidence collapses on a dense mix.

    Diagnostic finding (job c226c717..., not yet acted on): the
    `mono = spoken_raw + sung_raw` sum below is itself a contamination
    source, independent of any instrumental bleed. Instrumental bleed into
    this signal was checked and ruled out on both verified windows -- only
    2.0-2.4% of its STFT power sits in time-frequency bins where the
    accompaniment is comparably loud or louder, and its envelope barely
    correlates with the accompaniment's (+0.035 Price monologue, -0.094
    sung verse). But summing the two raw vocal estimates measurably
    degrades pYIN's pitch tracking even when the secondary one is quiet:
    on the Price monologue, running pYIN on Bandit's speech estimate alone
    scores 0.901 frame accuracy against the verified spoken label, vs. 0.645
    on this function's actual sum -- Demucs's vocal estimate there is ~14dB
    down (mostly silence, correctly) but still corrupts the shared F0
    estimate once mixed in. The indicated next step is to run pYIN
    separately on each raw estimate and feed both resulting feature streams
    into `_sung_score`, not to pick a single "presumptive source" input --
    which one is right is exactly the thing being classified, so choosing
    one upfront would be circular. Separately, the sung verse is NOT
    explained by this: every candidate input (the sum, Bandit alone, Demucs
    alone) scores below chance-for-sung there (best is 0.434) -- a
    different, still-open calibration/feature problem."""
    mono = np.mean(spoken_raw + sung_raw, axis=0).astype(np.float64)
    if mono.shape[-1] < _PITCH_FRAME_LENGTH:
        return np.zeros(n_frames, dtype=np.float64)

    f0, _voiced_flag, voiced_prob = librosa.pyin(
        mono,
        fmin=_FMIN_HZ,
        fmax=_FMAX_HZ,
        sr=sample_rate,
        frame_length=_PITCH_FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        fill_na=np.nan,
    )
    frame_rate_hz = sample_rate / HOP_LENGTH

    beat_alignment = harmony_fit = None
    if accompaniment is not None and accompaniment.shape[-1] >= _PITCH_FRAME_LENGTH:
        accompaniment_mono = np.mean(accompaniment, axis=0).astype(np.float64)
        beat_alignment = _rolling_beat_alignment(mono, accompaniment_mono, sample_rate, frame_rate_hz, len(f0))
        harmony_fit = _rolling_harmony_fit(accompaniment_mono, sample_rate, f0, len(f0))

    score = _sung_score(f0, voiced_prob, frame_rate_hz, beat_alignment, harmony_fit)
    score = _match_length(score, n_frames)
    voiced = _match_length(~np.isnan(f0), n_frames)
    return _hysteresis_smooth(score, voiced)


def _sung_score(
    f0: np.ndarray,
    voiced_prob: np.ndarray,
    frame_rate_hz: float,
    beat_alignment: np.ndarray | None = None,
    harmony_fit: np.ndarray | None = None,
) -> np.ndarray:
    """Combines pYIN's F0 track (plus, when available, musical-context
    evidence from the accompaniment) into a per-frame [0, 1] "does this look
    sung" score from classical, cheap features -- no trained classifier:
    - harmonicity: pYIN's own voiced-probability (periodicity confidence).
      Weakest feature on a dense, produced mix -- pYIN's confidence
      collapses exactly where arbitration matters most, which is why it's
      de-weighted (not dropped) once the two features below are available.
    - stability: how little the pitch *drifts* over ~100ms (a sustained note
      still wobbles with vibrato, but doesn't glissando the way speech does).
    - discreteness: how close the local mean pitch sits to an equal-tempered
      semitone (singing targets scale steps; speech doesn't).
    - vibrato: 4-8Hz oscillation energy in the local pitch contour.
    - duration: length of the current stable-and-voiced run (a briefly
      in-tune speech syllable isn't a held note).
    - beat_alignment (optional): does the vocal's onset land on the
      accompaniment's beat grid (`_rolling_beat_alignment`)? Sung phrasing
      locks to the beat; speech doesn't. Independent of pitch tracking
      entirely, so it doesn't degrade on a dense mix the way harmonicity does.
    - harmony_fit (optional): does the vocal's F0 land on a chord tone of the
      accompaniment (`_rolling_harmony_fit`)? Same rationale, independent
      evidence source.
    Unvoiced frames (no detectable pitch at all) lean spoken, not neutral --
    that's consonants and breath, not ambiguous singing."""
    voiced = ~np.isnan(f0)
    midi = 69.0 + 12.0 * np.log2(np.maximum(f0, _EPS) / 440.0)
    midi_filled = _fill_nan_1d(np.where(voiced, midi, np.nan))

    stability = _rolling_stability(midi_filled)
    discreteness = _rolling_discreteness(midi_filled)
    vibrato = _rolling_vibrato(midi_filled, frame_rate_hz)
    duration = _stable_run_score(stability > 0.5, voiced)
    harmonicity = np.nan_to_num(voiced_prob, nan=0.0)

    if beat_alignment is not None and harmony_fit is not None:
        score = (
            0.15 * harmonicity
            + 0.20 * stability
            + 0.15 * discreteness
            + 0.10 * vibrato
            + 0.10 * duration
            + 0.15 * beat_alignment
            + 0.15 * harmony_fit
        )
    else:
        score = 0.30 * harmonicity + 0.25 * stability + 0.20 * discreteness + 0.15 * vibrato + 0.10 * duration
    return np.clip(np.where(voiced, score, 0.15), 0.0, 1.0)


def _rolling_beat_alignment(
    vocal_mono: np.ndarray, accompaniment_mono: np.ndarray, sample_rate: int, frame_rate_hz: float, n_frames: int
) -> np.ndarray:
    """1.0 where the vocal has a strong onset landing on (or near) the
    accompaniment's beat grid, decaying with distance from the nearest beat
    and held through a note's sustain rather than just its attack -- a sung
    phrase's syllable onsets tend to land on the beat, a spoken line's don't.
    Uses the accompaniment (not the vocal) to find the beat grid itself, so
    this doesn't depend on the vocal having any detectable pitch at all."""
    onset_env_accompaniment = librosa.onset.onset_strength(y=accompaniment_mono, sr=sample_rate, hop_length=HOP_LENGTH)
    _tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env_accompaniment, sr=sample_rate, hop_length=HOP_LENGTH)
    if len(beat_frames) == 0:
        return np.zeros(n_frames, dtype=np.float64)

    onset_env_vocal = librosa.onset.onset_strength(y=vocal_mono, sr=sample_rate, hop_length=HOP_LENGTH)
    onset_env_vocal = _match_length(onset_env_vocal, n_frames)

    frame_idx = np.arange(n_frames)
    distance_frames = np.abs(frame_idx[:, None] - np.asarray(beat_frames)[None, :]).min(axis=1)
    beat_proximity = np.exp(-(distance_frames / frame_rate_hz) / _BEAT_ALIGNMENT_TOLERANCE_SECONDS)

    normalized_onset = onset_env_vocal / (onset_env_vocal.max() + _EPS)
    aligned = normalized_onset * beat_proximity
    hold_frames = max(int(round(_BEAT_ALIGNMENT_HOLD_SECONDS * frame_rate_hz)), 1)
    return _rolling_trailing_max(aligned, hold_frames)


def _rolling_harmony_fit(accompaniment_mono: np.ndarray, sample_rate: int, f0: np.ndarray, n_frames: int) -> np.ndarray:
    """How much of the accompaniment's chroma energy, at each frame, sits at
    the vocal's own pitch class -- chance level is 1/12 (pitch class
    unrelated to the chord being played); a sung note that's actually a
    chord tone should sit well above it, a spoken syllable's incidental
    pitch shouldn't track harmony at all. Uses chroma_stft (not chroma_cqt)
    to stay on the same cheap STFT machinery the rest of this module already
    pays for, rather than adding a second, pricier transform."""
    chroma = librosa.feature.chroma_stft(y=accompaniment_mono, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma = chroma / (chroma.sum(axis=0, keepdims=True) + _EPS)
    chroma = _match_length_axis(chroma, n_frames, axis=1)

    voiced = ~np.isnan(f0)
    # np.where evaluates both branches eagerly, so unvoiced frames' NaN midi
    # (masked out below anyway) must be filled before the int cast, or it warns.
    midi = np.nan_to_num(69.0 + 12.0 * np.log2(np.maximum(f0, _EPS) / 440.0), nan=69.0)
    pitch_class = np.where(voiced, np.round(midi).astype(np.int64) % 12, 0)

    fit = chroma[pitch_class, np.arange(n_frames)]
    return np.clip(np.where(voiced, fit / _HARMONY_FIT_SCALE, 0.0), 0.0, 1.0)


def _rolling_windows(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(x, (half, window - 1 - half), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, window)


def _rolling_trailing_max(x: np.ndarray, window: int) -> np.ndarray:
    """Max over `x[i - window + 1 : i + 1]` at each `i` -- a *trailing* max
    (unlike `_rolling_windows`'s centered one), so a spike at frame `i`
    (e.g. an onset) holds forward through the next `window` frames instead
    of leaking backward before it happened."""
    if window <= 1:
        return x
    padded = np.pad(x, (window - 1, 0), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, window).max(axis=1)


def _match_length_axis(arr: np.ndarray, target_len: int, axis: int) -> np.ndarray:
    """Like `_match_length`, but along an arbitrary axis of an n-d array
    (used for chroma's (12, T) shape instead of a 1-D per-frame score)."""
    n = arr.shape[axis]
    if n == target_len:
        return arr
    if n > target_len:
        index = [slice(None)] * arr.ndim
        index[axis] = slice(0, target_len)
        return arr[tuple(index)]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target_len - n)
    return np.pad(arr, pad_width, mode="edge")


def _rolling_stability(midi_filled: np.ndarray) -> np.ndarray:
    """1.0 where the local least-squares pitch slope is ~flat, decaying
    toward 0.0 as the pitch glides -- vibrato's fast wobble averages out of a
    linear-slope fit, so this responds to drift, not vibrato."""
    windows = _rolling_windows(midi_filled, _STABILITY_WINDOW_FRAMES)
    t = np.arange(_STABILITY_WINDOW_FRAMES, dtype=np.float64)
    t = t - t.mean()
    denom = np.sum(t**2)
    slopes = (windows * t).sum(axis=1) / denom
    return np.exp(-np.abs(slopes) / _STABILITY_SLOPE_SCALE)


def _rolling_discreteness(midi_filled: np.ndarray) -> np.ndarray:
    windows = _rolling_windows(midi_filled, _STABILITY_WINDOW_FRAMES)
    mean_midi = windows.mean(axis=1)
    cents_dev = np.abs(mean_midi - np.round(mean_midi)) * 100.0
    return np.exp(-cents_dev / _DISCRETENESS_CENTS_SCALE)


def _rolling_vibrato(midi_filled: np.ndarray, frame_rate_hz: float) -> np.ndarray:
    windows = _rolling_windows(midi_filled, _VIBRATO_WINDOW_FRAMES)
    detrended = windows - windows.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(detrended, axis=1))
    freqs = np.fft.rfftfreq(_VIBRATO_WINDOW_FRAMES, d=1.0 / frame_rate_hz)
    band = (freqs >= _VIBRATO_MIN_HZ) & (freqs <= _VIBRATO_MAX_HZ)
    total_energy = spectrum.sum(axis=1) + _EPS
    if not band.any():
        return np.zeros(len(detrended))
    band_energy = spectrum[:, band].max(axis=1)
    return np.clip((band_energy / total_energy) / _VIBRATO_RATIO_SCALE, 0.0, 1.0)


def _stable_run_score(stable_mask: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Normalized (capped at 2x _MIN_NOTE_FRAMES) length of the contiguous
    voiced+stable run each frame belongs to, credited to every frame in the
    run (not just its tail) so a long held note scores high throughout."""
    active = stable_mask & voiced
    run_len = np.zeros(len(active), dtype=np.float64)
    count = 0
    for i, is_active in enumerate(active):
        count = count + 1 if is_active else 0
        run_len[i] = count
    for i in range(len(run_len) - 2, -1, -1):
        if active[i] and active[i + 1]:
            run_len[i] = run_len[i + 1]
    return np.clip(run_len / (_MIN_NOTE_FRAMES * 2), 0.0, 1.0)


def _hysteresis_smooth(score: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Binary spoken(0)/sung(1) state machine with separate enter/exit
    thresholds (hysteresis) so noise around one threshold can't flip the
    decision every frame, then a short raised-cosine crossfade at each
    transition so the bias itself doesn't step discontinuously.

    Unvoiced frames (`voiced[i]` False) never drive a transition -- they
    just hold whatever state was last decided. A held note routinely
    contains brief unvoiced dips (a consonant inside a lyric, a breath, a
    pitch-tracker dropout on a real, densely-mixed recording), and treating
    "no pitch detected this frame" as *evidence of speech* was exactly what
    made a single held note flicker in and out of the sung state on real
    audio -- absence of a pitch estimate isn't evidence either way, so it
    shouldn't get to move the decision.

    A transition also requires a run of *consecutive* voiced frames past the
    threshold, not just one -- on real audio the per-frame score is noisy
    enough that a single crossing is common even mid-phrase (measured
    directly against a real cached take: without this debounce, the
    decision flipped roughly every 0.25-0.5s inside a single held note).
    Entering "sung" needs a longer run than exiting it -- see
    _ENTER_DEBOUNCE_FRAMES/_EXIT_DEBOUNCE_FRAMES's comment for why a
    symmetric debounce measurably overcorrected."""
    state = np.zeros(len(score), dtype=np.float64)
    current = 0.0
    enter_run = 0
    exit_run = 0
    for i, s in enumerate(score):
        if voiced[i]:
            if current == 0.0:
                enter_run = enter_run + 1 if s > _HYSTERESIS_HIGH else 0
                if enter_run >= _ENTER_DEBOUNCE_FRAMES:
                    current = 1.0
                    enter_run = 0
            else:
                exit_run = exit_run + 1 if s < _HYSTERESIS_LOW else 0
                if exit_run >= _EXIT_DEBOUNCE_FRAMES:
                    current = 0.0
                    exit_run = 0
        state[i] = current
    if _CROSSFADE_FRAMES <= 0 or len(state) == 0:
        return state
    kernel = np.hanning(2 * _CROSSFADE_FRAMES + 1)
    kernel /= kernel.sum()
    padded = np.pad(state, _CROSSFADE_FRAMES, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _fill_nan_1d(x: np.ndarray) -> np.ndarray:
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if not good.any():
        return np.zeros_like(x)
    return np.interp(idx, idx[good], x[good])


def _match_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode="edge")


def _stft_zero_padded(x: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Centered, Hann-windowed STFT -- same math as `_stft_numpy.py`'s
    private `_stft`, except zero-padded instead of reflect-padded for
    centering (see module docstring for why). Window matches `_istft`
    (imported unchanged) for exact overlap-add reconstruction."""
    win = hann(n_fft, sym=False).astype(np.float64)
    pad = n_fft // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="constant")
    length = xp.shape[-1]
    n_frames = max(1 + (length - n_fft) // hop_length, 1)
    needed = (n_frames - 1) * hop_length + n_fft
    if needed > length:
        xp = np.pad(xp, [(0, 0)] * (x.ndim - 1) + [(0, needed - length)], mode="constant")
    shape = x.shape[:-1] + (n_frames, n_fft)
    strides = xp.strides[:-1] + (hop_length * xp.strides[-1], xp.strides[-1])
    frames = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    frames = frames * win
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    spec = spec / np.sqrt(n_fft)  # matches _istft's `normalized=True` convention (imported unchanged)
    return np.moveaxis(spec, -1, -2)
