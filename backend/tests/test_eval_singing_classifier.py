"""Sanity checks for scripts/eval_singing_classifier.py's dataset invariants
and metric functions — not a claim about the classifier's real-world
accuracy (that requires real Bandit+Demucs inference on real audio, see the
script's own module docstring), just that the harness measures what it
claims to measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.eval_singing_classifier import (
    DEV_WINDOWS,
    TEST_WINDOWS,
    TRACKS,
    WINDOWS,
    Window,
    WindowFrames,
    _recalls,
    _window_frame_slice,
    baseline_metrics,
    classifier_metrics,
    classifier_metrics_voiced_only,
)


def test_every_window_is_assigned_to_exactly_one_split():
    assert set(DEV_WINDOWS) | set(TEST_WINDOWS) == set(WINDOWS)
    assert set(DEV_WINDOWS).isdisjoint(TEST_WINDOWS)
    assert len(DEV_WINDOWS) + len(TEST_WINDOWS) == len(WINDOWS)


def test_dev_and_test_both_nonempty_and_cover_both_labels():
    # A test set that only had one class, or was empty, couldn't do what
    # CLAUDE.md's task asks of it -- final confirmation on unseen, unbiased
    # ground truth.
    assert len(DEV_WINDOWS) > 0
    assert len(TEST_WINDOWS) > 0
    assert {w.label for w in TEST_WINDOWS} == {"spoken", "sung"}


def test_at_least_three_distinct_tracks_and_ten_windows():
    assert len({w.track_id for w in WINDOWS}) >= 3
    assert len(WINDOWS) >= 8


def test_every_window_references_a_known_track():
    for w in WINDOWS:
        assert w.track_id in TRACKS


def test_windows_are_positive_duration_and_ordered():
    for w in WINDOWS:
        assert w.end_seconds > w.start_seconds


def test_window_frame_slice_converts_seconds_to_hop_aligned_frames():
    # HOP_LENGTH=512 @ 44100Hz -> ~86.13 frames/sec.
    w = Window("t", "spoken", 1.0, 2.0, "dev")
    sl = _window_frame_slice(w, sample_rate=44100, n_frames=1000)
    assert sl.start == pytest.approx(86, abs=1)
    assert sl.stop == pytest.approx(172, abs=1)
    assert sl.stop > sl.start


def test_window_frame_slice_clips_to_available_frames():
    w = Window("t", "spoken", 0.0, 100.0, "dev")
    sl = _window_frame_slice(w, sample_rate=44100, n_frames=50)
    assert sl.stop == 50


def _frames(label: str, predicted_sung: np.ndarray, voiced: np.ndarray | None = None) -> WindowFrames:
    n = len(predicted_sung)
    w = Window("t", label, 0, 1, "dev")
    ground_truth_sung = np.full(n, label == "sung", dtype=bool)
    voiced = np.ones(n, dtype=bool) if voiced is None else voiced
    return WindowFrames(w, ground_truth_sung, predicted_sung, raw_score=np.zeros(n), voiced=voiced)


def test_recalls_perfect_predictor_scores_one_everywhere():
    gt = np.array([False, False, True, True])
    m = _recalls(gt, gt.copy())
    assert m["balanced_accuracy"] == 1.0
    assert m["spoken_recall"] == 1.0
    assert m["sung_recall"] == 1.0
    assert m["accuracy"] == 1.0


def test_recalls_inverted_predictor_scores_zero_recall_both_classes():
    gt = np.array([False, False, True, True])
    m = _recalls(gt, ~gt)
    assert m["balanced_accuracy"] == 0.0
    assert m["spoken_recall"] == 0.0
    assert m["sung_recall"] == 0.0


def test_recalls_accuracy_can_diverge_from_balanced_accuracy_under_class_imbalance():
    # 9 spoken frames, 1 sung frame; always-predict-spoken gets 90% raw
    # accuracy but 50% balanced accuracy -- the exact trap task item 3 warns
    # about, and the reason this harness reports both.
    gt = np.array([False] * 9 + [True])
    always_spoken = np.zeros(10, dtype=bool)
    m = _recalls(gt, always_spoken)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["spoken_recall"] == 1.0
    assert m["sung_recall"] == 0.0


def test_baseline_metrics_always_spoken_and_always_sung_are_perfect_complements():
    frames = [
        _frames("spoken", np.array([False, False, False])),
        _frames("sung", np.array([True, True])),
    ]
    baselines = baseline_metrics(frames)
    assert baselines["always-spoken"]["spoken_recall"] == 1.0
    assert baselines["always-spoken"]["sung_recall"] == 0.0
    assert baselines["always-sung"]["spoken_recall"] == 0.0
    assert baselines["always-sung"]["sung_recall"] == 1.0
    # Balanced accuracy treats both degenerate baselines identically, unlike
    # raw accuracy which would favor whichever class has more frames here.
    assert baselines["always-spoken"]["balanced_accuracy"] == pytest.approx(0.5)
    assert baselines["always-sung"]["balanced_accuracy"] == pytest.approx(0.5)


def test_baseline_metrics_majority_class_matches_the_larger_class():
    # 3 spoken frames, 1 sung frame -> majority baseline should predict
    # spoken for everything, same as always-spoken here.
    frames = [
        _frames("spoken", np.array([False, False, False])),
        _frames("sung", np.array([True])),
    ]
    baselines = baseline_metrics(frames)
    assert baselines["majority-class"]["spoken_recall"] == 1.0
    assert baselines["majority-class"]["sung_recall"] == 0.0


def test_classifier_metrics_aggregates_across_windows():
    frames = [
        _frames("spoken", np.array([False, False, True])),  # 2/3 correct
        _frames("sung", np.array([True, False])),  # 1/2 correct
    ]
    m = classifier_metrics(frames)
    assert m["spoken_recall"] == pytest.approx(2 / 3)
    assert m["sung_recall"] == pytest.approx(1 / 2)
    assert m["n_frames"] == 5


def test_classifier_metrics_voiced_only_excludes_unvoiced_frames():
    # One spoken window: 2 voiced frames (both correctly predicted spoken),
    # 1 unvoiced frame wrongly predicted sung. voiced-only recall should
    # ignore the unvoiced mistake entirely; all-frames recall should not.
    frames = [_frames("spoken", np.array([False, False, True]), voiced=np.array([True, True, False]))]
    all_frames = classifier_metrics(frames)
    voiced_only = classifier_metrics_voiced_only(frames)
    assert all_frames["spoken_recall"] == pytest.approx(2 / 3)
    assert voiced_only["spoken_recall"] == 1.0
    assert voiced_only["n_frames"] == 2
