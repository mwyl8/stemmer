"""Routing + tier selection: music -> DemucsONNXSeparator; video -> Bandit;
full -> ChainedSeparator(bandit, demucs). Bandit construction itself is
mocked here (router._select_bandit) so these stay fast/offline regardless of
whether the `speech` extras + downloaded weights are present — that's
exactly what test_chained_sep.py and the bandit_sep smoke path cover
separately.
"""

from __future__ import annotations

import pytest

from backend.separators import router
from backend.separators.base import Separator
from backend.separators.chained_sep import ChainedSeparator
from backend.separators.demucs_onnx import DemucsONNXSeparator
from backend.separators.router import select_separator


@pytest.mark.parametrize("tier", ["fast", "balanced"])
def test_music_mode_routes_to_demucs_onnx(tier):
    sep = select_separator("music", tier)
    assert isinstance(sep, DemucsONNXSeparator)


def test_best_tier_not_yet_wired():
    with pytest.raises(NotImplementedError):
        select_separator("music", "best")


def test_video_mode_routes_to_bandit(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(router, "_select_bandit", lambda: sentinel)
    assert select_separator("video", "balanced") is sentinel


def test_full_mode_chains_bandit_and_demucs_on_configured_tier(monkeypatch):
    bandit_sentinel = object()
    demucs_calls = []

    def fake_select_demucs(tier):
        demucs_calls.append(tier)
        return "demucs-for-" + tier

    monkeypatch.setattr(router, "_select_bandit", lambda: bandit_sentinel)
    monkeypatch.setattr(router, "_select_demucs", fake_select_demucs)

    sep = select_separator("full", "balanced")

    assert isinstance(sep, ChainedSeparator)
    assert sep.bandit is bandit_sentinel
    assert sep.demucs == "demucs-for-balanced"
    assert demucs_calls == ["balanced"]  # full mode's tier drives the Demucs stage


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
