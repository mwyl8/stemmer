"""Mode + tier -> pick/chain separators. Music mode = htdemucs (ONNX, the
product path); Video mode = Bandit and Full mode = chained Bandit->Demucs land
in Phase 4.

The PyTorch oracle (`_oracle_torch.py`) is never reachable from here — it is a
reference used directly by the eval path (tests, eval harness), never by
mode+tier routing. See CLAUDE.md guardrail: PyTorch is not the product path.
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
    if mode != "music":
        raise NotImplementedError(f"mode {mode!r} lands in Phase 4 (Bandit speech/music/effects)")
    if tier not in _TIER_MODEL_PATH:
        raise NotImplementedError(f"tier {tier!r} not wired yet (htdemucs_ft 'best' tier is later work)")
    return DemucsONNXSeparator(model_path=_TIER_MODEL_PATH[tier], segment=TIERS[tier]["segment"])
