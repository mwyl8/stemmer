"""Mode + tier -> pick/chain separators.

- Music mode  -> htdemucs (ONNX, the product path).
- Video mode  -> Bandit alone: speech/music/effects (PyTorch fallback — see
                 bandit_sep.py for why there's no ONNX export yet).
- Full mode   -> chained: Bandit first, then Demucs on Bandit's "music" stem,
                 merged into speech/vocals/drums/bass/other (chained_sep.py).

Bandit's import (and its `speech` dependency group: torch/torchaudio/
pytorch-lightning/spafe) is deferred into the video/full branches, never at
module level — so `import router` and music-mode routing never require it,
same footprint-isolation principle as demucs_onnx.py keeping torch out of
the music path entirely.

The PyTorch oracle (`_oracle_torch.py`) is never reachable from here either
— it is a reference used directly by the eval path (tests, eval harness),
never by mode+tier routing. See CLAUDE.md guardrail: PyTorch is not the
product path for Demucs; Bandit is the one explicit, documented exception to
that (PRD §5), until it gets an ONNX export.
"""

from __future__ import annotations

from backend.config import MODELS_DIR, TIERS
from backend.separators.base import Separator
from backend.separators.demucs_onnx import DemucsONNXSeparator

# fast/balanced both run the single-model htdemucs graph; fast points at the
# int8-quantized weights (scripts/quantize.py), balanced at full precision.
# "best" (htdemucs_ft, a 4-model ensemble) isn't exported yet — PRD guardrail
# is to never default to it anyway; it's opt-in, later work.
_TIER_MODEL_PATH = {
    "fast": MODELS_DIR / "htdemucs_core_int8.onnx",
    "balanced": MODELS_DIR / "htdemucs_core.onnx",
}


def select_separator(mode: str, tier: str) -> Separator:
    if mode == "music":
        return _select_demucs(tier)
    if mode == "video":
        return _select_bandit()
    if mode == "full":
        from backend.separators.chained_sep import ChainedSeparator

        return ChainedSeparator(bandit=_select_bandit(), demucs=_select_demucs(tier))
    raise NotImplementedError(f"unknown mode {mode!r}")


def _select_demucs(tier: str) -> DemucsONNXSeparator:
    if tier not in _TIER_MODEL_PATH:
        raise NotImplementedError(f"tier {tier!r} not wired yet (htdemucs_ft 'best' tier is later work)")
    return DemucsONNXSeparator(model_path=_TIER_MODEL_PATH[tier], segment=TIERS[tier]["segment"])


def _select_bandit() -> Separator:
    # Bandit currently ships one checkpoint — tier doesn't affect it (yet).
    from backend.separators.bandit_sep import BanditSeparator

    return BanditSeparator()
