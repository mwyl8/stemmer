"""peaks.py: server-side waveform peaks precompute (PRD Addendum §2.5) so the
player never has to decode a full file client-side just to draw a waveform.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from backend.peaks import compute_peaks


def _write_wav(path, data, sr=44100):
    sf.write(path, data.T, sr)  # data is (channels, samples) -> soundfile wants (samples, channels)


def test_returns_one_series_per_channel(tmp_path):
    sr = 44100
    n = sr * 2
    data = np.stack([np.sin(np.linspace(0, 40 * np.pi, n)), np.sin(np.linspace(0, 20 * np.pi, n))]).astype(np.float32)
    path = tmp_path / "stem.wav"
    _write_wav(path, data, sr)

    peaks = compute_peaks(path, num_buckets=100)
    assert len(peaks) == 2
    assert len(peaks[0]) == 200  # 2 * num_buckets: interleaved [min, max] per bucket
    assert len(peaks[1]) == 200


def test_peaks_stay_within_source_amplitude_bounds(tmp_path):
    sr = 44100
    n = sr
    rng = np.random.default_rng(0)
    data = (0.4 * rng.standard_normal((1, n))).astype(np.float32)
    path = tmp_path / "stem.wav"
    _write_wav(path, data, sr)

    peaks = compute_peaks(path, num_buckets=50)
    true_max = float(np.max(np.abs(data)))
    assert max(abs(v) for v in peaks[0]) <= true_max + 1e-4


def test_captures_a_transient_that_naive_striding_would_miss(tmp_path):
    """A single one-sample spike buried in near-silence must still show up
    in the downsampled peaks — this is the whole point of per-bucket
    min/max instead of picking every Nth sample."""
    sr = 44100
    n = sr
    data = np.zeros((1, n), dtype=np.float32)
    spike_index = 12345
    data[0, spike_index] = 0.9
    path = tmp_path / "stem.wav"
    _write_wav(path, data, sr)

    peaks = compute_peaks(path, num_buckets=200)
    assert max(peaks[0]) == pytest.approx(0.9, abs=1e-4)


def test_symmetric_signal_yields_roughly_symmetric_peaks(tmp_path):
    """A waveform that's genuinely symmetric about zero should downsample to
    a peaks series that's still roughly symmetric — i.e. we keep both the
    min AND max per bucket, not just one-sided magnitude (which would render
    as a one-sided, not mirrored, waveform in the player)."""
    sr = 44100
    n = sr
    data = (0.6 * np.sin(np.linspace(0, 200 * np.pi, n))).astype(np.float32).reshape(1, -1)
    path = tmp_path / "stem.wav"
    _write_wav(path, data, sr)

    peaks = compute_peaks(path, num_buckets=100)[0]
    assert min(peaks) == pytest.approx(-max(peaks), abs=0.05)


def test_empty_audio_returns_empty_series_per_channel(tmp_path):
    data = np.zeros((2, 0), dtype=np.float32)
    path = tmp_path / "empty.wav"
    sf.write(path, data.T, 44100)

    peaks = compute_peaks(path, num_buckets=100)
    assert peaks == [[], []]


def test_num_buckets_larger_than_sample_count_is_clamped(tmp_path):
    data = np.array([[0.1, 0.2, -0.3, 0.4, -0.5]], dtype=np.float32)
    path = tmp_path / "tiny.wav"
    _write_wav(path, data, 44100)

    peaks = compute_peaks(path, num_buckets=10_000)  # far more buckets than samples
    assert len(peaks) == 1
    assert len(peaks[0]) > 0  # didn't blow up / didn't return garbage
