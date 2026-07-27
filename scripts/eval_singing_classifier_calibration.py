"""Calibration exploration for the Speech-vs-Singing classifier
(_vocal_partition.py) -- built on scripts/eval_singing_classifier.py's
cached dev/test windows and raw Bandit+Demucs stems (run that script first;
this one reuses its cache, it does not re-fetch or re-separate anything).

Three questions, each its own section:

1. ROC/AUC on the raw pre-hysteresis score (`_frame_score_and_voiced`),
   dev set, voiced frames only -- how much signal the score carries,
   independent of where the threshold sits. Reported overall and per
   sung-track (Thriller vs. Queen), against a shared pooled-spoken negative
   set so the two are directly comparable.

2. Two independent sweeps against dev's balanced accuracy:
   (a) a static single threshold on the raw score (no hysteresis at all) --
       finds the best possible fixed cut point, compared against the
       current _HYSTERESIS_HIGH=0.33/_LOW=0.28 evaluated the same way.
   (b) the hysteresis enter/exit debounce, holding HIGH/LOW fixed at their
       current values -- current (20, 9), two symmetric points, and the
       literal reversal (9, 20).

3. The ablation that matters most: output audio quality (RMS ratio /
   cross-stem leakage / envelope correlation, same metrics as
   scripts/eval_singing_bleed_thriller.py) with arbitration OFF (the plain
   Wiener magnitude-ratio mask, bias=0.5 everywhere) vs. ON (today's shipped
   classifier) -- reusing partition_vocal_bus's new bias_override parameter
   so both conditions run through the exact same production DSP path.

Discipline: sections 1-3 above run on DEV_WINDOWS only; (20,20) is chosen
from dev alone, then confirmed once against TEST_WINDOWS in main() -- no
threshold, debounce combo, or on/off decision is chosen by looking at test.

Three further checks on that (20,20) choice, added after the first pass
(reported dev-and-test side by side throughout, since they're validity/
robustness checks on an already-chosen configuration, not a new search that
could overfit to test):

- run_extended_debounce_validity_check(): does balanced accuracy keep
  climbing past (20,20)? Every WINDOWS entry is homogeneous, so it
  structurally *can* keep climbing without that meaning anything -- this
  either confirms (20,20) is a real peak or exposes that the first pass's
  finding was partly an artifact of homogeneous eval windows.
- run_transition_lag_analysis(): TRANSITION_WINDOWS (eval_singing_
  classifier.py) span real spoken<->sung boundaries, unlike every other
  window here -- measures how much extra lag (20,20) adds at a genuine
  transition, both against a (heavily caveated) estimated switch time and,
  more reliably, against (20,9) directly on the same boundary.
- run_debounce_output_quality_ablation(): off vs. on at each exit-debounce
  value in a given list (enter held fixed at 20) -- the first pass's
  ablation only ever compared off against today's shipped classifier, so it
  couldn't say whether widening exit debounce changes the on/off answer, or
  whether output quality moves monotonically with how long the classifier
  is trusted once committed.

Measurement/calibration only: recommends a configuration but changes no
shipped default (_ARBITRATION_STRENGTH, _HYSTERESIS_HIGH/_LOW,
_ENTER/_EXIT_DEBOUNCE_FRAMES all untouched).

Usage:
    uv run --group speech python scripts/eval_singing_classifier_calibration.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.eval.metrics import cross_stem_leakage_fraction, stem_envelope_correlation, window_rms_ratio
from backend.separators._vocal_partition import HOP_LENGTH, _HYSTERESIS_HIGH, _HYSTERESIS_LOW, partition_vocal_bus
from scripts.eval_singing_classifier import (
    DEV_WINDOWS,
    TEST_WINDOWS,
    TRANSITION_WINDOWS,
    Window,
    _recalls,
    _window_frame_slice,
    compute_classifier_outputs,
    compute_raw_stems,
)

_ENTER_EXIT_COMBOS: list[tuple[str, int, int]] = [
    ("current (20, 9)", 20, 9),
    ("symmetric-tight (9, 9)", 9, 9),
    ("symmetric-mid (14, 14)", 14, 14),
    ("symmetric-loose (20, 20)", 20, 20),
    ("reversed (9, 20)", 9, 20),
]
_CROSSFADE_FRAMES = 5  # matches _vocal_partition.py's own constant; not swept here


# --------------------------------------------------------------------------
# Section 1: ROC / AUC on the raw score
# --------------------------------------------------------------------------


def roc_curve(y_sung: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hand-rolled (no sklearn dependency -- same posture as
    backend/eval/metrics.py's own SDR/SIR/SAR: a few lines beats a new
    dependency for one curve). Returns (fpr, tpr), monotonic from (0,0) to
    (1,1), one point per distinct score value plus the two endpoints."""
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y_sung[order]
    score_sorted = score[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(~y_sorted)
    distinct = np.where(np.diff(score_sorted) != 0)[0]
    idx = np.r_[distinct, len(score_sorted) - 1]
    tpr = np.r_[0.0, tps[idx] / n_pos]
    fpr = np.r_[0.0, fps[idx] / n_neg]
    return fpr, tpr


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapezoid(tpr, fpr))


def compute_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(pos_scores), dtype=bool), np.zeros(len(neg_scores), dtype=bool)])
    s = np.concatenate([pos_scores, neg_scores])
    fpr, tpr = roc_curve(y, s)
    return auc(fpr, tpr)


@dataclass
class TrackVoicedScores:
    track_id: str
    label: str  # this window's ground-truth label ("spoken" or "sung")
    score: np.ndarray  # raw pre-hysteresis score, voiced frames only


def _dev_voiced_scores(windows: list[Window]) -> list[TrackVoicedScores]:
    cache: dict[str, tuple] = {}
    out = []
    for w in windows:
        if w.track_id not in cache:
            cache[w.track_id] = compute_classifier_outputs(w.track_id)
        score, voiced, _state, sr = cache[w.track_id]
        sl = _window_frame_slice(w, sr, len(score))
        out.append(TrackVoicedScores(w.track_id, w.label, score[sl][voiced[sl]]))
    return out


def run_auc_section() -> None:
    print("=" * 100)
    print("SECTION 1: ROC-AUC on the raw pre-hysteresis score (dev, voiced frames only)")
    print("=" * 100)
    records = _dev_voiced_scores(DEV_WINDOWS)
    all_sung = np.concatenate([r.score for r in records if r.label == "sung"])
    all_spoken = np.concatenate([r.score for r in records if r.label == "spoken"])

    rows = {
        "all dev (pooled)": compute_auc(all_sung, all_spoken),
    }
    for track_id in sorted({r.track_id for r in records if r.label == "sung"}):
        pos = np.concatenate([r.score for r in records if r.track_id == track_id and r.label == "sung"])
        rows[f"{track_id} sung vs. pooled spoken"] = compute_auc(pos, all_spoken)
        same_track_neg_parts = [r.score for r in records if r.track_id == track_id and r.label == "spoken"]
        if same_track_neg_parts:
            rows[f"{track_id} sung vs. {track_id} spoken (same-track)"] = compute_auc(pos, np.concatenate(same_track_neg_parts))

    for name, value in rows.items():
        n_str = "" if np.isnan(value) else ""
        print(f"  {name:<48}AUC = {value:.3f}{n_str}")
    print(
        "\n  (0.5 = no separation, 1.0 = perfect. Same-track rows use that track's own spoken\n"
        "  frames as the negative class -- cleanest comparison, but only exists where a track has\n"
        "  both labels in dev. Cross-track rows use ALL dev spoken frames pooled as a shared\n"
        "  negative set so different sung tracks are directly comparable to each other.)"
    )


# --------------------------------------------------------------------------
# Section 2a: static single-threshold sweep on the raw score
# --------------------------------------------------------------------------


def run_static_threshold_sweep() -> float:
    print("\n" + "=" * 100)
    print("SECTION 2a: static single-threshold sweep on raw score (dev, all frames incl. unvoiced)")
    print("=" * 100)
    records = _dev_voiced_scores_all_frames(DEV_WINDOWS)
    y = np.concatenate([r["gt_sung"] for r in records])
    score = np.concatenate([r["score"] for r in records])

    grid = np.linspace(0.0, 1.0, 101)
    best_t, best_bal_acc = None, -1.0
    rows: dict[str, dict] = {}
    for t in grid:
        m = _recalls(y, score > t)
        if m["balanced_accuracy"] > best_bal_acc:
            best_bal_acc, best_t = m["balanced_accuracy"], t
    rows[f"optimal static threshold t*={best_t:.2f}"] = _recalls(y, score > best_t)
    rows["current HIGH=0.33 (static, no hysteresis)"] = _recalls(y, score > _HYSTERESIS_HIGH)
    rows["current LOW=0.28 (static, no hysteresis)"] = _recalls(y, score > _HYSTERESIS_LOW)

    header = f"  {'':<42}{'bal. acc.':<12}{'spoken-recall':<16}{'sung-recall':<14}{'accuracy'}"
    print(header)
    for name, m in rows.items():
        print(f"  {name:<42}{m['balanced_accuracy']:<12.3f}{m['spoken_recall']:<16.3f}{m['sung_recall']:<14.3f}{m['accuracy']:.3f}")
    print(
        "\n  This is a *static* single-threshold classifier on the raw score -- no hysteresis, no\n"
        "  debounce, no crossfade. It isolates threshold placement from timing effects; it is not a\n"
        "  proposal to remove hysteresis (a per-frame flicker-prone decision is its own problem)."
    )
    return best_t


def _dev_voiced_scores_all_frames(windows: list[Window]) -> list[dict]:
    """Like _dev_voiced_scores, but keeps unvoiced frames too (score/voiced
    both defined for every frame; unvoiced frames' score is fixed at 0.15 by
    _sung_score, see that function's docstring) -- the static-threshold sweep
    should be judged on the same all-frames basis the main harness's headline
    numbers use, not a voiced-only subset."""
    cache: dict[str, tuple] = {}
    out = []
    for w in windows:
        if w.track_id not in cache:
            cache[w.track_id] = compute_classifier_outputs(w.track_id)
        score, _voiced, _state, sr = cache[w.track_id]
        sl = _window_frame_slice(w, sr, len(score))
        s = score[sl]
        out.append({"score": s, "gt_sung": np.full(len(s), w.label == "sung", dtype=bool)})
    return out


# --------------------------------------------------------------------------
# Section 2b: enter/exit debounce sweep
# --------------------------------------------------------------------------


def _hysteresis_smooth_configurable(
    score: np.ndarray, voiced: np.ndarray, high: float, low: float, enter_frames: int, exit_frames: int, crossfade_frames: int = _CROSSFADE_FRAMES
) -> np.ndarray:
    """Exact reimplementation of _vocal_partition.py's _hysteresis_smooth,
    with the enter/exit debounce (and, for completeness, high/low) exposed as
    parameters instead of hardcoded module constants -- that function itself
    isn't parameterized, so a sweep needs its own copy. Never imported by
    production code; kept in lockstep with _hysteresis_smooth by inspection,
    not by import, since the whole point is to try values production never
    uses."""
    state = np.zeros(len(score), dtype=np.float64)
    current = 0.0
    enter_run = 0
    exit_run = 0
    for i, s in enumerate(score):
        if voiced[i]:
            if current == 0.0:
                enter_run = enter_run + 1 if s > high else 0
                if enter_run >= enter_frames:
                    current = 1.0
                    enter_run = 0
            else:
                exit_run = exit_run + 1 if s < low else 0
                if exit_run >= exit_frames:
                    current = 0.0
                    exit_run = 0
        state[i] = current
    if crossfade_frames <= 0 or len(state) == 0:
        return state
    kernel = np.hanning(2 * crossfade_frames + 1)
    kernel /= kernel.sum()
    padded = np.pad(state, crossfade_frames, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _debounce_metrics_for_combo(windows: list[Window], enter_frames: int, exit_frames: int) -> dict:
    cache: dict[str, tuple] = {}
    gt_all, pred_all = [], []
    for w in windows:
        if w.track_id not in cache:
            score, voiced, _state, sr = compute_classifier_outputs(w.track_id)
            state = _hysteresis_smooth_configurable(score, voiced, _HYSTERESIS_HIGH, _HYSTERESIS_LOW, enter_frames, exit_frames)
            cache[w.track_id] = (state, sr)
        state, sr = cache[w.track_id]
        sl = _window_frame_slice(w, sr, len(state))
        pred = state[sl] > 0.5
        gt_all.append(np.full(len(pred), w.label == "sung", dtype=bool))
        pred_all.append(pred)
    return _recalls(np.concatenate(gt_all), np.concatenate(pred_all))


def run_debounce_sweep() -> tuple[str, int, int]:
    print("\n" + "=" * 100)
    print("SECTION 2b: enter/exit debounce sweep (dev, HIGH/LOW held at current 0.33/0.28)")
    print("=" * 100)
    header = f"  {'':<28}{'bal. acc.':<12}{'spoken-recall':<16}{'sung-recall':<14}{'accuracy'}"
    print(header)
    best_name, best_enter, best_exit, best_bal_acc = None, None, None, -1.0
    for name, enter_frames, exit_frames in _ENTER_EXIT_COMBOS:
        m = _debounce_metrics_for_combo(DEV_WINDOWS, enter_frames, exit_frames)
        print(f"  {name:<28}{m['balanced_accuracy']:<12.3f}{m['spoken_recall']:<16.3f}{m['sung_recall']:<14.3f}{m['accuracy']:.3f}")
        if m["balanced_accuracy"] > best_bal_acc:
            best_bal_acc, best_name, best_enter, best_exit = m["balanced_accuracy"], name, enter_frames, exit_frames
    print(f"\n  Best on dev balanced accuracy: {best_name}")
    return best_name, best_enter, best_exit


# --------------------------------------------------------------------------
# Section 3: arbitration on/off output-quality ablation
# --------------------------------------------------------------------------


def geomean(xs) -> float:
    xs = np.asarray(list(xs), dtype=np.float64)
    return float(np.exp(np.mean(np.log(xs))))


def _partition_both_conditions(track_id: str, enter_frames: int, exit_frames: int) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Runs partition_vocal_bus twice on cached raw stems: ON (bias = the
    hysteresis state for the given enter/exit debounce, recomputed from the
    cached raw score/voiced via _hysteresis_smooth_configurable -- passing
    (20, 9) reproduces bit-for-bit what production ships today) and OFF
    (bias = 0.5 everywhere, the plain unarbitrated Wiener mask, per
    _biased_mask's own docstring). Both reuse the exact same production DSP
    path via bias_override -- no reimplementation of the mask/ISTFT math."""
    bandit_speech, demucs_vocals, instruments, sr = compute_raw_stems(track_id)
    score, voiced, _state, sr2 = compute_classifier_outputs(track_id)
    assert sr == sr2
    state = _hysteresis_smooth_configurable(score, voiced, _HYSTERESIS_HIGH, _HYSTERESIS_LOW, enter_frames, exit_frames)
    spoken_on, sung_on = partition_vocal_bus(bandit_speech, demucs_vocals, sr, accompaniment=instruments, bias_override=state)
    neutral = np.full_like(state, 0.5)
    spoken_off, sung_off = partition_vocal_bus(bandit_speech, demucs_vocals, sr, accompaniment=instruments, bias_override=neutral)
    return {"on": (spoken_on, sung_on, sr), "off": (spoken_off, sung_off, sr)}


def _window_quality_metrics(spoken: np.ndarray, sung: np.ndarray, sr: int, w: Window) -> tuple[float, float, float]:
    target = sung if w.label == "sung" else spoken
    other = spoken if w.label == "sung" else sung
    ratio = window_rms_ratio(target, other, sr, w.start_seconds, w.end_seconds)
    leak = cross_stem_leakage_fraction(spoken, sung, sr, start_seconds=w.start_seconds, end_seconds=w.end_seconds)
    corr = stem_envelope_correlation(spoken, sung, sr, w.start_seconds, w.end_seconds)
    return ratio, leak, corr


def run_debounce_output_quality_ablation(windows: list[Window], label: str, exit_combos: list[tuple[str, int, int]]) -> None:
    """off vs. on at each (enter, exit) combo in `exit_combos` -- all run
    through the same production partition_vocal_bus path via bias_override,
    so this directly answers both "should arbitration exist at all" (off
    vs. any on) and "does exit-debounce length change that answer" (on at
    one exit value vs. another). `off` is computed once (bias is a flat 0.5
    regardless of debounce, so it doesn't depend on `exit_combos` at all)."""
    print("\n" + "=" * 100)
    combo_desc = ", ".join(name for name, _e, _x in exit_combos)
    print(f"TASK: off vs. on at {combo_desc} -- output audio quality ({label})")
    print("=" * 100)
    tracks = sorted({w.track_id for w in windows})
    partitions_by_combo = {name: {t: _partition_both_conditions(t, enter, exit_) for t in tracks} for name, enter, exit_ in exit_combos}
    off_track = next(iter(partitions_by_combo.values()))  # any combo's "off" works -- identical across all of them

    conditions: dict[str, object] = {"off": lambda t: off_track[t]["off"]}
    for name, _enter, _exit in exit_combos:
        conditions[name] = (lambda t, _name=name: partitions_by_combo[_name][t]["on"])

    summary: dict[str, dict[str, list[float]]] = {c: {"ratio": [], "leak": [], "corr": []} for c in conditions}
    header = f"  {'window':<46}{'cond':<12}{'rms ratio':<11}{'leakage frac':<14}{'correlation'}"
    print(header)
    for w in windows:
        for cond_name, getter in conditions.items():
            spoken, sung, sr = getter(w.track_id)
            ratio, leak, corr = _window_quality_metrics(spoken, sung, sr, w)
            label_str = f"{w.track_id} {w.label} ({w.note})"
            print(f"  {label_str:<46}{cond_name:<12}{ratio:<11.2f}{leak:<14.3f}{corr:+.3f}")
            summary[cond_name]["ratio"].append(ratio)
            summary[cond_name]["leak"].append(leak)
            summary[cond_name]["corr"].append(corr)

    print(f"\n  {'':<14}{'geomean rms ratio':<20}{'mean leakage':<16}{'mean |correlation|'}")
    for cond_name in conditions:
        geo_ratio = geomean(summary[cond_name]["ratio"])
        mean_leak = float(np.mean(summary[cond_name]["leak"]))
        mean_abs_corr = float(np.mean(np.abs(summary[cond_name]["corr"])))
        print(f"  {cond_name:<14}{geo_ratio:<20.2f}{mean_leak:<16.3f}{mean_abs_corr:.3f}")
    print(
        "\n  Geometric mean for rms ratio (an arithmetic mean is swamped by whichever window happens\n"
        "  to have the largest absolute ratio). Higher ratio = better (target louder than the stem\n"
        "  it shouldn't be in). Lower leakage/|correlation| = better (less of one performance smeared\n"
        "  across both stems)."
    )


# --------------------------------------------------------------------------
# Task 1: extended symmetric-debounce validity check
# --------------------------------------------------------------------------


def run_extended_debounce_validity_check() -> None:
    """Not a search for a better value -- a validity check on the (20,20)
    finding above. Every WINDOWS entry is homogeneous (entirely spoken or
    entirely sung), so a long enough debounce can only help *by
    construction*: it just filters out within-window noise, since there's no
    real transition inside these windows it ever needs to react to. If
    balanced accuracy keeps climbing well past (20,20), that's this eval
    set's homogeneity showing through, not evidence that long debounce is
    good policy in general -- see run_transition_lag_analysis for the check
    that actually stresses reaction time. Values grow geometrically (x1.5)
    from 60 while dev balanced accuracy keeps improving, capped at 800
    frames (~9.3s) -- already far past anything plausible to ship."""
    print("\n" + "=" * 100)
    print("TASK 1: extended symmetric-debounce validity check (is the (20,20) gain just homogeneity?)")
    print("=" * 100)
    frame_rate_hz = 44100 / HOP_LENGTH
    header = f"  {'frames (sec)':<16}{'dev bal.acc':<13}{'dev sung-r':<12}{'test bal.acc':<13}{'test sung-r'}"
    print(header)

    values = [9, 14, 20, 30, 40, 60]
    cap = 800
    rows: list[tuple[int, dict, dict]] = []
    while True:
        v = values[len(rows)] if len(rows) < len(values) else values[-1]
        dev_m = _debounce_metrics_for_combo(DEV_WINDOWS, v, v)
        test_m = _debounce_metrics_for_combo(TEST_WINDOWS, v, v)
        rows.append((v, dev_m, test_m))
        print(f"  {f'{v} ({v / frame_rate_hz:.2f}s)':<16}{dev_m['balanced_accuracy']:<13.3f}{dev_m['sung_recall']:<12.3f}{test_m['balanced_accuracy']:<13.3f}{test_m['sung_recall']:.3f}")
        if len(rows) < len(values):
            continue
        if v >= cap or rows[-1][1]["balanced_accuracy"] <= rows[-2][1]["balanced_accuracy"]:
            break
        values.append(min(int(v * 1.5), cap))

    peak = max(rows, key=lambda r: r[1]["balanced_accuracy"])
    still_climbing = rows[-1][1]["balanced_accuracy"] >= rows[-2][1]["balanced_accuracy"] and rows[-1][0] < cap
    print(f"\n  Peak dev balanced accuracy: {peak[1]['balanced_accuracy']:.3f} at ({peak[0]}, {peak[0]}) frames")
    if still_climbing:
        print(
            "  STILL CLIMBING at the cap -- consistent with the eval set's homogeneity inflating the\n"
            "  score for arbitrarily long debounce, not with (20,20) being a genuinely good value.\n"
            "  Treat the (20,20) recommendation as unconfirmed by this check."
        )
    else:
        print("  Peaks and declines after (20,20)-ish -- clean result, not just homogeneity.")


# --------------------------------------------------------------------------
# Task 2: transition lag
# --------------------------------------------------------------------------


def find_transition_frame(state: np.ndarray, sample_rate: int, search_start_seconds: float, search_end_seconds: float, direction: str) -> tuple[int | None, int]:
    """First frame index (into the full-track `state` array) where the
    thresholded (>0.5) state crosses in `direction` within the search
    window, plus the total number of crossings found there (>1 means the
    window wasn't as clean as hoped -- see TransitionWindow's docstring).
    Returns (None, 0) if no crossing is found at all."""
    frame_rate_hz = sample_rate / HOP_LENGTH
    start = max(int(search_start_seconds * frame_rate_hz), 0)
    end = min(int(search_end_seconds * frame_rate_hz), len(state))
    predicted_sung = (state[start:end] > 0.5).astype(int)
    diff = np.diff(predicted_sung)
    target_diff = 1 if direction == "spoken_to_sung" else -1
    crossings = np.where(diff == target_diff)[0]
    if len(crossings) == 0:
        return None, 0
    return int(crossings[0]) + 1 + start, len(crossings)


def run_transition_lag_analysis() -> None:
    print("\n" + "=" * 100)
    print("TASK 2: transition lag at real spoken<->sung boundaries (dev only -- no test-side transition exists, see below)")
    print("=" * 100)
    frame_rate_hz = 44100 / HOP_LENGTH
    for t in TRANSITION_WINDOWS:
        score, voiced, _state, sr = compute_classifier_outputs(t.track_id)
        print(f"\n  {t.track_id} {t.direction} [{t.split}]: {t.note}")
        results = {}
        for name, enter_frames, exit_frames in (("current (20,9)", 20, 9), ("candidate (20,20)", 20, 20)):
            state = _hysteresis_smooth_configurable(score, voiced, _HYSTERESIS_HIGH, _HYSTERESIS_LOW, enter_frames, exit_frames)
            frame, n_crossings = find_transition_frame(state, sr, t.search_start_seconds, t.search_end_seconds, t.direction)
            results[name] = (frame, n_crossings)
            if frame is None:
                print(f"    {name:<20} no crossing found in [{t.search_start_seconds:.0f}s, {t.search_end_seconds:.0f}s] -- state never flips there")
                continue
            transition_seconds = frame / frame_rate_hz
            lag_vs_estimate = transition_seconds - t.estimated_switch_seconds
            flag = "" if n_crossings == 1 else f"  ({n_crossings} crossings found -- not a clean single transition, treat with suspicion)"
            print(
                f"    {name:<20} committed at {transition_seconds:.2f}s "
                f"(vs. estimated switch {t.estimated_switch_seconds:.0f}s +/-{t.switch_uncertainty_seconds:.0f}s: "
                f"lag {lag_vs_estimate:+.2f}s, caveated){flag}"
            )
        f9, _ = results["current (20,9)"]
        f20, _ = results["candidate (20,20)"]
        if f9 is not None and f20 is not None:
            extra_lag = (f20 - f9) / frame_rate_hz
            print(
                f"    Extra lag from widening exit to 20 (this pair only, doesn't depend on the estimated\n"
                f"    switch time at all): {extra_lag:+.3f}s"
            )
    print(
        "\n  No held-out (test) transition exists in the current corpus: Queen and Chaplin each contain\n"
        "  only one label, and Thriller's only unambiguous transition boundaries sit inside its\n"
        "  already-dev windows. Confirming this on a genuinely held-out transition would need a new\n"
        "  source with both spoken and sung content outside what's already been used to tune anything."
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    run_auc_section()
    run_static_threshold_sweep()
    best_name, best_enter, best_exit = run_debounce_sweep()

    print("\n" + "=" * 100)
    print("CONFIRMING DEBOUNCE CHOICE ON TEST (touched once -- not used to choose anything above)")
    print("=" * 100)
    current = _debounce_metrics_for_combo(DEV_WINDOWS, 20, 9)
    chosen_dev = _debounce_metrics_for_combo(DEV_WINDOWS, best_enter, best_exit)
    current_test = _debounce_metrics_for_combo(TEST_WINDOWS, 20, 9)
    chosen_test = _debounce_metrics_for_combo(TEST_WINDOWS, best_enter, best_exit)
    print(f"\n  Debounce: current (20,9) vs. chosen {best_name}")
    header = f"  {'':<28}{'bal. acc.':<12}{'spoken-recall':<16}{'sung-recall'}"
    print(header)
    print(f"  {'current, dev':<28}{current['balanced_accuracy']:<12.3f}{current['spoken_recall']:<16.3f}{current['sung_recall']:.3f}")
    print(f"  {'current, test':<28}{current_test['balanced_accuracy']:<12.3f}{current_test['spoken_recall']:<16.3f}{current_test['sung_recall']:.3f}")
    print(f"  {'chosen, dev':<28}{chosen_dev['balanced_accuracy']:<12.3f}{chosen_dev['spoken_recall']:<16.3f}{chosen_dev['sung_recall']:.3f}")
    print(f"  {'chosen, test':<28}{chosen_test['balanced_accuracy']:<12.3f}{chosen_test['spoken_recall']:<16.3f}{chosen_test['sung_recall']:.3f}")

    # This round's additions -- all reported dev-and-test side by side per
    # this round's instructions, but none of them choose a NEW configuration
    # by looking at test: (20,20) was already chosen above from dev alone;
    # these are validity/robustness checks on that choice, not a fresh
    # search.
    run_extended_debounce_validity_check()
    run_transition_lag_analysis()

    # exit=9 (shipped) / exit=14 (midpoint) / exit=20 (candidate), enter
    # held fixed at 20 throughout -- isolates exit-debounce length as the
    # one axis being varied. Tests whether output quality degrades
    # monotonically as exit debounce lengthens (i.e. as the classifier is
    # trusted for longer once committed), the same shape of result the
    # earlier _ARBITRATION_STRENGTH sweep found for trusting it *harder*.
    exit_axis = [("exit=9 (shipped)", 20, 9), ("exit=14", 20, 14), ("exit=20 (candidate)", 20, 20)]
    run_debounce_output_quality_ablation(DEV_WINDOWS, "dev", exit_axis)
    run_debounce_output_quality_ablation(TEST_WINDOWS, "test", exit_axis)


if __name__ == "__main__":
    main()
