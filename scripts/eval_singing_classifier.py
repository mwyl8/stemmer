"""Frame-level evaluation harness for the Speech-vs-Singing hysteresis
classifier (`_vocal_partition.py`'s `_frame_sung_bias`, built from
`_frame_score_and_voiced` + `_hysteresis_smooth`) — measures the classifier's
own spoken/sung decisions against hand-labeled, whole-window ground truth.

This is deliberately separate from scripts/eval_singing_bleed_thriller.py's
RMS-ratio / cross-stem-leakage numbers: those measure output *audio* quality,
which conflates the classifier's own accuracy with the two upstream models'
(Bandit/Demucs) energy imbalance — a stem can pass a loud RMS-ratio floor
even while the classifier is wrong most of the time, if the wrong model
happened to be quiet there. This script judges the classifier on its own
per-frame decisions instead.

Measurement only: reads `_vocal_partition.py`'s internals (including the
`_frame_score_and_voiced` split factored out for exactly this purpose — see
that function's docstring) but does not change any constant, threshold, or
default routing behavior.

Ground truth: hand-picked windows across 3 real tracks, each window labeled
as a whole ("spoken" or "sung" for its entire span), not per-frame — see
WINDOWS below, split explicitly into DEV_WINDOWS (used to look at the
classifier while iterating) and TEST_WINDOWS (only for final confirmation,
never for tuning). Thriller's source.wav was reused from an existing cached
job (already fetched for earlier bleed-fix work); Queen's "Somebody to Love"
was reused from an existing cached karaoke-mode job's source.wav; Chaplin's
"The Great Dictator" speech was fetched fresh via
`backend.ingest.ingest(url=...)`. All three end up as plain 44.1kHz stereo
WAVs under data/eval_cache/sources/ (gitignored, not redistributed, not
reproducible via a single script yet — see require_sources() below) — see
WINDOWS' comments for the significant caveat on the Queen timestamps.

Usage:
    uv run --group speech python scripts/eval_singing_classifier.py [--recompute]

Each track's raw Bandit+Demucs pass is cached under data/eval_cache/raw/ —
it costs on the order of the track's own duration on CPU (an ~14-minute
track takes close to 15 minutes), so a bare re-run reuses the cache.
`--recompute` forces regenerating everything (raw stems + classifier scores)
from scratch, e.g. after a change to the classifier's own feature code.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.separators._two_stage_chain import run_chain
from backend.separators._vocal_partition import (
    HOP_LENGTH,
    N_FFT,
    _HYSTERESIS_HIGH,
    _HYSTERESIS_LOW,
    _frame_score_and_voiced,
    _hysteresis_smooth,
    _stft_zero_padded,
)
from backend.separators.chained_sep import MUSIC_STEM_NAME
from backend.separators.router import select_separator
from backend.separators.singing_sep import _DEMUCS_VOCAL_SOURCE

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = REPO_ROOT / "data" / "eval_cache" / "sources"
RAW_CACHE_DIR = REPO_ROOT / "data" / "eval_cache" / "raw"
CLASSIFIER_CACHE_DIR = REPO_ROOT / "data" / "eval_cache" / "classifier"

Label = Literal["spoken", "sung"]
Split = Literal["dev", "test"]


@dataclass(frozen=True)
class Window:
    track_id: str
    label: Label
    start_seconds: float
    end_seconds: float
    split: Split
    note: str = ""


TRACKS: dict[str, Path] = {
    "thriller": SOURCES_DIR / "thriller.wav",
    "queen_somebody_to_love": SOURCES_DIR / "queen_somebody_to_love.wav",
    "chaplin_great_dictator_speech": SOURCES_DIR / "chaplin_great_dictator_speech.wav",
}

# Whole-window labels, hand-picked for unambiguous content -- no frame-level
# hand labeling needed, every frame in a window inherits that window's label.
#
# --- Thriller (Michael Jackson) --- a real cached job's source audio
# (originally fetched for the Speech-vs-Singing bleed-fix work). The three
# "dev" windows and their timestamps are carried over unchanged from
# scripts/eval_singing_bleed_thriller.py's WINDOWS, which already documents
# Price-monologue/sung-verse as having been used to tune the hysteresis
# thresholds and enter/exit debounce -- they belong in dev, not test, for
# exactly that reason. The outro was explicitly never used to tune anything
# (that script's own docstring: "never used to tune _ARBITRATION_STRENGTH,
# the hysteresis thresholds, or the beat/harmony feature weights") -- it is
# the one genuinely held-out Thriller window, so it goes in test.
#
# --- Queen -- "Somebody to Love" --- fetched fresh for this harness
# (data/eval_cache/sources/queen_somebody_to_love.wav). IMPORTANT CAVEAT:
# these timestamps are placed from general/memorized knowledge of the song's
# structure (intro a cappella -> verses -> gospel call-and-response bridge ->
# sustained high-note outro), NOT verified by listening to this specific
# audio -- nothing in this environment can play audio. Windows are placed
# well inside each section (not at transition edges) and made wide (30s+)
# specifically to tolerate a plausible ±10-15s error in that recollection.
# If these numbers look wrong, the first thing to check is whether these
# boundaries actually land where intended -- listen to the source file and
# adjust before trusting the Queen-track numbers in isolation.
#
# --- Chaplin -- "The Great Dictator" final speech --- also fetched fresh
# (data/eval_cache/sources/chaplin_great_dictator_speech.wav). Lower
# timestamp risk than the Queen track: it's a single continuous spoken
# monologue (with an orchestral swell under parts of it, deliberately
# similar in character to the Price monologue above) for its whole ~3:36
# runtime, so "is this whole clip spoken" doesn't depend on hitting a
# specific section.
WINDOWS: list[Window] = [
    Window("thriller", "spoken", 40, 120, "dev", "car-scene dialogue"),
    Window("thriller", "spoken", 4 * 60 + 15, 4 * 60 + 52, "dev", "Vincent Price monologue (used to tune debounce)"),
    Window("thriller", "sung", 2 * 60 + 20, 2 * 60 + 50, "dev", "sung verse (used to tune debounce)"),
    Window("thriller", "sung", 12 * 60 + 20, 12 * 60 + 55, "test", "outro (held-out, never used to tune anything)"),
    Window("queen_somebody_to_love", "sung", 20, 55, "dev", "verse 1 (approx. timestamp, see caveat above)"),
    Window("queen_somebody_to_love", "sung", 2 * 60 + 50, 3 * 60 + 20, "dev", "gospel call-and-response bridge (approx.)"),
    Window("queen_somebody_to_love", "sung", 3 * 60 + 55, 4 * 60 + 25, "test", "sustained high-note outro (held-out, approx.)"),
    Window("chaplin_great_dictator_speech", "spoken", 10, 60, "dev", "opening"),
    Window("chaplin_great_dictator_speech", "spoken", 90, 150, "dev", "middle"),
    Window("chaplin_great_dictator_speech", "spoken", 170, 210, "test", "closing (held-out)"),
]

DEV_WINDOWS = [w for w in WINDOWS if w.split == "dev"]
TEST_WINDOWS = [w for w in WINDOWS if w.split == "test"]
# Explicit, checked split invariant -- the whole point of separating these
# into two module-level lists is that dev-only code (baselines, any future
# tuning) can only ever be handed DEV_WINDOWS, never WINDOWS itself.
assert {w.split for w in WINDOWS} == {"dev", "test"}, "every window must be assigned dev or test"
assert len(DEV_WINDOWS) + len(TEST_WINDOWS) == len(WINDOWS)


@dataclass(frozen=True)
class TransitionWindow:
    """A window that genuinely spans a spoken<->sung switch, unlike every
    entry in WINDOWS above (each of which is homogeneous by construction --
    good for balanced-accuracy/AUC, useless for measuring how quickly the
    classifier reacts to a real transition). Kept as its own list, separate
    from WINDOWS, so it never gets pulled into the homogeneous-window
    accuracy/baseline machinery above by accident.

    `estimated_switch_seconds` is a best guess, not a verified instant (see
    each entry's note) -- `switch_uncertainty_seconds` records how much slack
    that guess needs, which is usually much larger than any plausible
    debounce-induced lag (a few hundred ms). Consumers should treat any
    "lag vs. estimated switch" number as heavily caveated, and prefer
    comparing the committed-transition time *between two debounce configs*
    on the same window instead -- that comparison doesn't depend on knowing
    the true switch instant at all."""

    track_id: str
    direction: Literal["spoken_to_sung", "sung_to_spoken"]
    search_start_seconds: float
    search_end_seconds: float
    estimated_switch_seconds: float
    switch_uncertainty_seconds: float
    split: Split
    note: str = ""


# Both transitions come from Thriller -- the only one of the 3 tracks with
# both spoken and sung content, so it's the only place a transition exists
# to measure. Both are "dev" (Thriller's homogeneous windows above are all
# dev too, since Price-monologue/sung-verse were used to tune the hysteresis
# thresholds) -- there is currently no held-out transition in the corpus;
# see the calibration script's report for that gap.
TRANSITION_WINDOWS: list[TransitionWindow] = [
    TransitionWindow(
        "thriller",
        "spoken_to_sung",
        100,
        160,
        130,
        10,
        "dev",
        "car-scene dialogue (WINDOWS: ends 120s) into the sung verse (WINDOWS: starts 140s) -- "
        "the true switch is known to fall in [120,140] from those two already-verified windows, "
        "so 130 (the midpoint) is a reasonable guess, not a verified instant.",
    ),
    TransitionWindow(
        "thriller",
        "sung_to_spoken",
        200,
        260,
        250,
        20,
        "dev",
        "singing into the Vincent Price monologue (WINDOWS: starts 255s). LOWER CONFIDENCE than "
        "the transition above: the 170-255s gap between the sung-verse window's end and the "
        "monologue window's start was never independently verified as continuously sung, so an "
        "extra, unknown transition inside this search range is possible -- check n_crossings.",
    ),
]


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio.T, sr  # (samples, channels) -> (channels, samples)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(path, audio.T, sample_rate, subtype="FLOAT")  # lossless float roundtrip, unlike default int16


def compute_raw_stems(track_id: str, force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Bandit's raw "speech" estimate + Demucs's raw "vocals" estimate +
    the summed non-vocal Demucs sources ("instruments") -- the exact three
    arrays `SpeechVsSingingSeparator.separate()` feeds into
    `partition_vocal_bus`/`_frame_sung_bias` in production (singing_sep.py),
    captured here *before* partitioning so the classifier can be measured on
    its actual real-world input. Cached to disk (heavy: a real Bandit+Demucs
    pass on a full track costs close to the track's own duration on CPU)."""
    cache_dir = RAW_CACHE_DIR / track_id
    paths = {name: cache_dir / f"{name}.wav" for name in ("bandit_speech", "demucs_vocals", "instruments")}
    if not force and all(p.exists() for p in paths.values()):
        bandit_speech, sr = _read_wav(paths["bandit_speech"])
        demucs_vocals, _sr2 = _read_wav(paths["demucs_vocals"])
        instruments, _sr3 = _read_wav(paths["instruments"])
        return bandit_speech, demucs_vocals, instruments, sr

    print(f"[{track_id}] no cached raw stems (or --recompute) -- running Bandit+Demucs for real...", file=sys.stderr)
    audio, sr = _read_wav(TRACKS[track_id])
    separator = select_separator("singing", "balanced", 4)  # same tier/stem_count as production default
    bandit_stems, demucs_stems = run_chain(separator.bandit, separator.demucs, audio, MUSIC_STEM_NAME)
    instrument_sources = [demucs_stems[name] for name in demucs_stems if name != _DEMUCS_VOCAL_SOURCE]
    instruments = np.sum(instrument_sources, axis=0).astype(np.float32)
    bandit_speech = bandit_stems["speech"].astype(np.float32)
    demucs_vocals = demucs_stems[_DEMUCS_VOCAL_SOURCE].astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(paths["bandit_speech"], bandit_speech, sr)
    _write_wav(paths["demucs_vocals"], demucs_vocals, sr)
    _write_wav(paths["instruments"], instruments, sr)
    return bandit_speech, demucs_vocals, instruments, sr


def compute_classifier_outputs(track_id: str, force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(raw_score, voiced, hysteresis_state, sample_rate) for the whole
    track, on `partition_vocal_bus`'s STFT frame grid -- `raw_score` is
    `_frame_score_and_voiced`'s pre-hysteresis per-frame [0,1] sung-
    likelihood (what `_HYSTERESIS_HIGH`/`_HYSTERESIS_LOW` are compared
    against), `hysteresis_state` is what production actually acts on
    (`_frame_sung_bias`'s return value, reproduced here as
    `_hysteresis_smooth(raw_score, voiced)` -- exactly the same call, since
    `_frame_sung_bias` is nothing but that composition). Computed once over
    the whole track (not per-window) so the hysteresis state machine carries
    real history into every window, same as a real job."""
    cache_dir = CLASSIFIER_CACHE_DIR / track_id
    score_path, voiced_path, state_path, sr_path = (cache_dir / f"{n}" for n in ("score.npy", "voiced.npy", "state.npy", "sr.txt"))
    if not force and all(p.exists() for p in (score_path, voiced_path, state_path, sr_path)):
        return np.load(score_path), np.load(voiced_path), np.load(state_path), int(sr_path.read_text())

    bandit_speech, demucs_vocals, instruments, sr = compute_raw_stems(track_id, force=force)
    spoken_stft = _stft_zero_padded(bandit_speech, N_FFT, HOP_LENGTH)
    n_frames = spoken_stft.shape[-1]
    score, voiced = _frame_score_and_voiced(bandit_speech, demucs_vocals, sr, n_frames, accompaniment=instruments)
    state = _hysteresis_smooth(score, voiced)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(score_path, score)
    np.save(voiced_path, voiced)
    np.save(state_path, state)
    sr_path.write_text(str(sr))
    return score, voiced, state, sr


def _window_frame_slice(window: Window, sample_rate: int, n_frames: int) -> slice:
    frame_rate_hz = sample_rate / HOP_LENGTH
    start = max(int(window.start_seconds * frame_rate_hz), 0)
    end = min(int(window.end_seconds * frame_rate_hz), n_frames)
    return slice(start, end)


@dataclass
class WindowFrames:
    window: Window
    ground_truth_sung: np.ndarray  # bool, one per frame in the window
    predicted_sung: np.ndarray  # bool, thresholded hysteresis state
    raw_score: np.ndarray
    voiced: np.ndarray


def gather_window_frames(windows: list[Window], force: bool = False) -> list[WindowFrames]:
    out = []
    for w in windows:
        score, voiced, state, sr = compute_classifier_outputs(w.track_id, force=force)
        sl = _window_frame_slice(w, sr, len(state))
        predicted_sung = state[sl] > 0.5
        ground_truth_sung = np.full(predicted_sung.shape, w.label == "sung", dtype=bool)
        out.append(WindowFrames(w, ground_truth_sung, predicted_sung, score[sl], voiced[sl]))
    return out


def _recalls(ground_truth_sung: np.ndarray, predicted_sung: np.ndarray) -> dict[str, float]:
    spoken_mask = ~ground_truth_sung
    sung_mask = ground_truth_sung
    spoken_recall = float(np.mean(~predicted_sung[spoken_mask])) if spoken_mask.any() else float("nan")
    sung_recall = float(np.mean(predicted_sung[sung_mask])) if sung_mask.any() else float("nan")
    balanced_accuracy = float(np.nanmean([spoken_recall, sung_recall]))
    accuracy = float(np.mean(predicted_sung == ground_truth_sung))
    return {
        "balanced_accuracy": balanced_accuracy,
        "spoken_recall": spoken_recall,
        "sung_recall": sung_recall,
        "accuracy": accuracy,
        "n_frames": int(ground_truth_sung.size),
        "n_spoken": int(spoken_mask.sum()),
        "n_sung": int(sung_mask.sum()),
    }


def baseline_metrics(frames: list[WindowFrames], seed: int = 0) -> dict[str, dict[str, float]]:
    gt = np.concatenate([f.ground_truth_sung for f in frames])
    rng = np.random.default_rng(seed)
    majority_predicts_sung = gt.mean() > 0.5
    return {
        "always-spoken": _recalls(gt, np.zeros_like(gt, dtype=bool)),
        "always-sung": _recalls(gt, np.ones_like(gt, dtype=bool)),
        "random (50/50)": _recalls(gt, rng.random(gt.shape) < 0.5),
        "majority-class": _recalls(gt, np.full_like(gt, majority_predicts_sung, dtype=bool)),
    }


def classifier_metrics(frames: list[WindowFrames]) -> dict[str, float]:
    gt = np.concatenate([f.ground_truth_sung for f in frames])
    pred = np.concatenate([f.predicted_sung for f in frames])
    return _recalls(gt, pred)


def classifier_metrics_voiced_only(frames: list[WindowFrames]) -> dict[str, float]:
    """Same as classifier_metrics, but restricted to voiced frames only --
    comparable to the "33-42% wrong on voiced frames of unambiguous ground
    truth" figure `_vocal_partition.py`'s own docstring already reports for
    the Price-monologue/sung-verse windows, as a cross-check against that
    prior finding rather than a replacement for the all-frames number above
    (which is what actually determines stem assignment in production,
    including unvoiced consonants/silence)."""
    gt = np.concatenate([f.ground_truth_sung[f.voiced] for f in frames])
    pred = np.concatenate([f.predicted_sung[f.voiced] for f in frames])
    return _recalls(gt, pred)


def debounce_diagnostic(frames: list[WindowFrames]) -> None:
    """Direct check of the enter/exit debounce asymmetry
    (_ENTER_DEBOUNCE_FRAMES=20 vs _EXIT_DEBOUNCE_FRAMES=9): for sung-labeled
    windows, how often does the raw score already clear the "enter" bar while
    the committed state is still spoken (the cost of the *longer* debounce),
    versus how often does the raw score already drop below the "exit" bar
    while the committed state is still sung (the cost of the *shorter* one)?
    A large gap between these two fractions, in the direction of more
    stuck-spoken frames, is exactly the bias the debounce asymmetry would be
    expected to produce."""
    sung_frames = [f for f in frames if f.window.label == "sung"]
    if not sung_frames:
        print("  (no sung-labeled windows in this split)")
        return
    score = np.concatenate([f.raw_score for f in sung_frames])
    voiced = np.concatenate([f.voiced for f in sung_frames])
    predicted_sung = np.concatenate([f.predicted_sung for f in sung_frames])

    stuck_spoken = voiced & (score > _HYSTERESIS_HIGH) & ~predicted_sung
    stuck_sung = voiced & (score < _HYSTERESIS_LOW) & predicted_sung
    n_voiced = int(voiced.sum())
    print(f"  voiced frames in sung-labeled windows: {n_voiced}")
    if n_voiced == 0:
        return
    print(
        f"  score > HIGH({_HYSTERESIS_HIGH}) but state still spoken (enter-debounce cost): "
        f"{stuck_spoken.sum()} frames ({100 * stuck_spoken.sum() / n_voiced:.1f}%)"
    )
    print(
        f"  score < LOW({_HYSTERESIS_LOW}) but state still sung (exit-debounce cost):      "
        f"{stuck_sung.sum()} frames ({100 * stuck_sung.sum() / n_voiced:.1f}%)"
    )


def _print_metrics_table(rows: dict[str, dict[str, float]]) -> None:
    header = f"  {'':<20}{'bal. acc.':<12}{'spoken-recall':<16}{'sung-recall':<14}{'accuracy':<11}{'n frames'}"
    print(header)
    for name, m in rows.items():
        print(
            f"  {name:<20}{m['balanced_accuracy']:<12.3f}{m['spoken_recall']:<16.3f}"
            f"{m['sung_recall']:<14.3f}{m['accuracy']:<11.3f}{m['n_frames']}"
        )


def require_sources() -> None:
    """Exits with a clear message (not a raw traceback from deep inside
    soundfile/librosa) if any of TRACKS' source WAVs are missing -- the
    normal state right after a fresh clone, since data/eval_cache/ is
    gitignored and not shipped in the repo (CLAUDE.md: don't redistribute
    fetched third-party audio). Called by both this script's main() and
    eval_singing_classifier_calibration.py's, since the calibration script
    hits the same missing-file case through several layers of caching
    helpers before ever printing anything of its own."""
    missing = [(track_id, path) for track_id, path in TRACKS.items() if not path.exists()]
    if not missing:
        return
    print("Missing source audio -- data/eval_cache/ is gitignored, not included in the repo, and", file=sys.stderr)
    print("not yet reproducible via a single script (tracked as a known gap). To populate it:", file=sys.stderr)
    for track_id, path in missing:
        print(f"  {track_id}: expected at {path}", file=sys.stderr)
    print(
        "Each file must be a 44.1kHz stereo WAV. See this file's module docstring for how the three\n"
        "tracks used so far were originally fetched (backend.ingest.ingest(url=...), or copied from a\n"
        "cached job's source.wav).",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="force re-running Bandit+Demucs and the classifier, ignoring any cache")
    args = parser.parse_args()

    require_sources()

    print("=" * 100)
    print("Computing classifier outputs (this is the slow part on a cache miss)...")
    dev_frames = gather_window_frames(DEV_WINDOWS, force=args.recompute)
    test_frames = gather_window_frames(TEST_WINDOWS, force=args.recompute)

    print("\n" + "=" * 100)
    print(f"DEV SET ({len(DEV_WINDOWS)} windows) -- trivial baselines")
    print("=" * 100)
    _print_metrics_table(baseline_metrics(dev_frames))

    print("\n" + "=" * 100)
    print("CURRENT CLASSIFIER -- dev vs. test")
    print("=" * 100)
    _print_metrics_table(
        {
            "dev (all frames)": classifier_metrics(dev_frames),
            "dev (voiced only)": classifier_metrics_voiced_only(dev_frames),
            "test (all frames)": classifier_metrics(test_frames),
            "test (voiced only)": classifier_metrics_voiced_only(test_frames),
        }
    )

    print("\n" + "=" * 100)
    print("PER-WINDOW BREAKDOWN (all frames) -- is sung-recall low everywhere, or only on Thriller's sung verse?")
    print("=" * 100)
    per_window_rows = {}
    for f in dev_frames + test_frames:
        w = f.window
        label = f"[{w.split}] {w.track_id} {w.label} ({w.note})"
        per_window_rows[label] = _recalls(f.ground_truth_sung, f.predicted_sung)
    _print_metrics_table(per_window_rows)

    print("\n" + "=" * 100)
    print("ENTER/EXIT DEBOUNCE DIAGNOSTIC (dev sung windows)")
    print("=" * 100)
    debounce_diagnostic(dev_frames)
    print("\nENTER/EXIT DEBOUNCE DIAGNOSTIC (test sung windows)")
    debounce_diagnostic(test_frames)


if __name__ == "__main__":
    main()
