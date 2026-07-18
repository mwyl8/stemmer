"""Routing + tier selection: music mode + fast/balanced tier -> DemucsONNXSeparator.
Video/full mode and the "best" tier raise NotImplementedError until Phase 4/6.
"""

import pytest

from backend.separators.base import Separator
from backend.separators.demucs_onnx import DemucsONNXSeparator
from backend.separators.router import select_separator


@pytest.mark.parametrize("tier", ["fast", "balanced"])
def test_music_mode_routes_to_demucs_onnx(tier):
    sep = select_separator("music", tier)
    assert isinstance(sep, DemucsONNXSeparator)


def test_best_tier_not_yet_wired():
    with pytest.raises(NotImplementedError):
        select_separator("music", "best")


@pytest.mark.parametrize("mode", ["video", "full"])
def test_non_music_modes_not_yet_wired(mode):
    with pytest.raises(NotImplementedError):
        select_separator(mode, "fast")


def test_separator_is_abstract():
    with pytest.raises(TypeError):
        Separator()
