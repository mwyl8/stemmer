"""Routing + tier + stem_count selection: music -> DemucsONNXSeparator
(4-stem default, 6-stem opt-in via htdemucs_6s); video -> Bandit; full ->
ChainedSeparator(bandit, demucs). Bandit construction itself is mocked here
(router._select_bandit) so these stay fast/offline regardless of whether the
`speech` extras + downloaded weights are present — that's exactly what
test_chained_sep.py and the bandit_sep smoke path cover separately.
"""

from __future__ import annotations

import pytest

from backend.config import MUSIC_MODELS
from backend.separators import router
from backend.separators.base import Separator
from backend.separators.chained_sep import ChainedSeparator
from backend.separators.demucs_onnx import DemucsONNXSeparator
from backend.separators.router import select_separator


@pytest.mark.parametrize("tier", ["fast", "balanced"])
def test_music_mode_routes_to_demucs_onnx(tier):
    sep = select_separator("music", tier)
    assert isinstance(sep, DemucsONNXSeparator)
    assert set(sep.sources) == {"vocals", "drums", "bass", "other"}  # 4-stem default


@pytest.mark.parametrize("tier", ["fast", "balanced"])
def test_music_mode_six_stem_routes_to_htdemucs_6s(tier):
    sep = select_separator("music", tier, stem_count=6)
    assert isinstance(sep, DemucsONNXSeparator)
    assert set(sep.sources) == {"vocals", "drums", "bass", "other", "guitar", "piano"}


def test_music_mode_defaults_to_four_stem():
    default_sep = select_separator("music", "balanced")
    four_stem_sep = select_separator("music", "balanced", stem_count=4)
    assert default_sep.sources == four_stem_sep.sources


def test_unsupported_stem_count_raises():
    with pytest.raises(NotImplementedError):
        select_separator("music", "balanced", stem_count=5)


def test_best_tier_not_yet_wired():
    with pytest.raises(NotImplementedError):
        select_separator("music", "best")


def test_video_mode_routes_to_bandit(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(router, "_select_bandit", lambda: sentinel)
    assert select_separator("video", "balanced") is sentinel


def test_full_mode_chains_bandit_and_demucs_on_configured_tier_and_stems(monkeypatch):
    bandit_sentinel = object()
    demucs_calls = []

    def fake_select_demucs(tier, stem_count=4):
        demucs_calls.append((tier, stem_count))
        return f"demucs-for-{tier}-{stem_count}"

    monkeypatch.setattr(router, "_select_bandit", lambda: bandit_sentinel)
    monkeypatch.setattr(router, "_select_demucs", fake_select_demucs)

    sep = select_separator("full", "balanced")

    assert isinstance(sep, ChainedSeparator)
    assert sep.bandit is bandit_sentinel
    assert sep.demucs == "demucs-for-balanced-4"
    assert demucs_calls == [("balanced", 4)]  # full mode's tier+stem_count drive the Demucs stage

    demucs_calls.clear()
    sep6 = select_separator("full", "balanced", stem_count=6)
    assert sep6.demucs == "demucs-for-balanced-6"
    assert demucs_calls == [("balanced", 6)]


def test_full_mode_propagates_bad_tier(monkeypatch):
    monkeypatch.setattr(router, "_select_bandit", lambda: object())
    with pytest.raises(NotImplementedError):
        select_separator("full", "best")


def test_unknown_mode_raises():
    with pytest.raises(NotImplementedError):
        select_separator("bogus", "fast")


def test_separator_is_abstract():
    with pytest.raises(TypeError):
        Separator()


def test_music_models_config_has_four_and_six_stem():
    assert MUSIC_MODELS == {4: "htdemucs", 6: "htdemucs_6s"}
