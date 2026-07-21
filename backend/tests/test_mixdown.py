"""mixdown.py: server-side "export a custom mix" (PRD Addendum §2.3) —
reproduces the player's effective-volume/exclusive-solo rules against the
full-quality wav stems instead of a Web Audio graph.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from backend.mixdown import TrackAdjustment, mix_stems


def _write_stem(tmp_path, name, value, n=1000, sr=44100):
    data = np.full((n, 2), value, dtype=np.float32)
    path = tmp_path / f"{name}.wav"
    sf.write(path, data, sr, subtype="FLOAT")
    return str(path)


def test_unity_mix_sums_all_stems(tmp_path):
    paths = {
        "vocals": _write_stem(tmp_path, "vocals", 0.1),
        "drums": _write_stem(tmp_path, "drums", 0.2),
    }
    result = mix_stems(paths, adjustments=[])
    # both at unity volume/pan=0 -> straight sum
    np.testing.assert_allclose(result.audio, np.full((2, 1000), 0.3, dtype=np.float32), atol=1e-5)


def test_muted_stem_is_excluded_from_the_mix(tmp_path):
    paths = {
        "vocals": _write_stem(tmp_path, "vocals", 0.1),
        "drums": _write_stem(tmp_path, "drums", 0.2),
    }
    adjustments = [TrackAdjustment(name="drums", muted=True)]
    result = mix_stems(paths, adjustments)
    np.testing.assert_allclose(result.audio, np.full((2, 1000), 0.1, dtype=np.float32), atol=1e-5)


def test_solo_excludes_every_non_soloed_stem(tmp_path):
    paths = {
        "vocals": _write_stem(tmp_path, "vocals", 0.1),
        "drums": _write_stem(tmp_path, "drums", 0.2),
        "bass": _write_stem(tmp_path, "bass", 0.3),
    }
    adjustments = [TrackAdjustment(name="vocals", solo=True)]
    result = mix_stems(paths, adjustments)
    np.testing.assert_allclose(result.audio, np.full((2, 1000), 0.1, dtype=np.float32), atol=1e-5)


def test_volume_scales_a_stems_contribution(tmp_path):
    paths = {"vocals": _write_stem(tmp_path, "vocals", 0.2)}
    adjustments = [TrackAdjustment(name="vocals", volume=0.5)]
    result = mix_stems(paths, adjustments)
    np.testing.assert_allclose(result.audio, np.full((2, 1000), 0.1, dtype=np.float32), atol=1e-5)


def test_master_volume_scales_the_whole_mix(tmp_path):
    paths = {"vocals": _write_stem(tmp_path, "vocals", 0.2)}
    result = mix_stems(paths, adjustments=[], master_volume=0.5)
    np.testing.assert_allclose(result.audio, np.full((2, 1000), 0.1, dtype=np.float32), atol=1e-5)


def test_master_muted_silences_everything_regardless_of_per_track_state(tmp_path):
    paths = {"vocals": _write_stem(tmp_path, "vocals", 0.2)}
    adjustments = [TrackAdjustment(name="vocals", solo=True)]  # would otherwise definitely play
    with pytest.raises(ValueError):
        mix_stems(paths, adjustments, master_muted=True)


def test_hard_left_pan_zeroes_the_right_channel(tmp_path):
    paths = {"vocals": _write_stem(tmp_path, "vocals", 0.2)}
    adjustments = [TrackAdjustment(name="vocals", pan=-1.0)]
    result = mix_stems(paths, adjustments)
    assert result.audio[1].max() == pytest.approx(0.0, abs=1e-5)
    assert result.audio[0].max() > 0.1  # left channel keeps (boosted-for-equal-power) signal


def test_center_pan_is_unity_on_both_channels(tmp_path):
    paths = {"vocals": _write_stem(tmp_path, "vocals", 0.2)}
    result = mix_stems(paths, adjustments=[TrackAdjustment(name="vocals", pan=0.0)])
    np.testing.assert_allclose(result.audio[0], result.audio[1], atol=1e-5)
    np.testing.assert_allclose(result.audio[0], np.full(1000, 0.2, dtype=np.float32), atol=1e-5)


def test_everything_muted_raises_value_error(tmp_path):
    paths = {"vocals": _write_stem(tmp_path, "vocals", 0.2)}
    adjustments = [TrackAdjustment(name="vocals", muted=True)]
    with pytest.raises(ValueError):
        mix_stems(paths, adjustments)


def test_mismatched_stem_lengths_are_padded_not_truncated(tmp_path):
    short = tmp_path / "vocals.wav"
    sf.write(short, np.full((500, 2), 0.1, dtype=np.float32), 44100, subtype="FLOAT")
    long = tmp_path / "drums.wav"
    sf.write(long, np.full((1000, 2), 0.2, dtype=np.float32), 44100, subtype="FLOAT")

    result = mix_stems({"vocals": str(short), "drums": str(long)}, adjustments=[])
    assert result.audio.shape[1] == 1000  # padded to the longer stem, not truncated to the shorter
    np.testing.assert_allclose(result.audio[:, :500], np.full((2, 500), 0.3, dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(result.audio[:, 500:], np.full((2, 500), 0.2, dtype=np.float32), atol=1e-5)


def test_no_matching_stems_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        mix_stems({}, adjustments=[])
