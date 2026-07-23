"""Mode + tier + stem_count -> pick/chain separators.

- Music mode  -> htdemucs (ONNX, the product path). 4-stem (vocals/drums/
                 bass/other) is the default; stem_count=6 opts into
                 htdemucs_6s (+guitar/piano) — same ONNX export path
                 (scripts/export_onnx.py --model), just a different
                 pretrained checkpoint, so it's a model-path swap here, not
                 new code.
- Video mode  -> Bandit alone: speech/music/effects (PyTorch fallback — see
                 bandit_sep.py for why there's no ONNX export yet). Bandit
                 has no stem-count concept; stem_count is ignored.
- Full mode   -> chained: Bandit first, then Demucs (at the configured
                 stem_count) on Bandit's "music" stem, merged into
                 speech + the Demucs source set (chained_sep.py).
- Singing mode-> the same Bandit-then-Demucs chain as Full mode, but merged
                 into spoken_speech / sung_vocals / instruments instead —
                 "does the talking voice land in a different stem than the
                 singing voice" (singing_sep.py).

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

from backend.config import DEFAULT_STEM_COUNT, MODELS_DIR, MUSIC_MODELS, TIERS
from backend.separators.base import Separator
from backend.separators.demucs_onnx import DemucsONNXSeparator

# fast/balanced both run the single-model htdemucs graph; fast points at the
# int8-quantized weights (scripts/quantize.py), balanced at full precision.
# "best" (htdemucs_ft, a 4-model ensemble) isn't exported yet — PRD guardrail
# is to never default to it anyway; it's opt-in, later work.
_TIER_SUFFIX = {"fast": "_int8", "balanced": ""}


def select_separator(mode: str, tier: str, stem_count: int = DEFAULT_STEM_COUNT) -> Separator:
    if mode == "music":
        return _select_demucs(tier, stem_count)
    if mode == "video":
        return _select_bandit()
    if mode == "full":
        from backend.separators.chained_sep import ChainedSeparator

        return ChainedSeparator(bandit=_select_bandit(), demucs=_select_demucs(tier, stem_count))
    if mode == "singing":
        from backend.separators.singing_sep import SpeechVsSingingSeparator

        return SpeechVsSingingSeparator(bandit=_select_bandit(), demucs=_select_demucs(tier, stem_count))
    raise NotImplementedError(f"unknown mode {mode!r}")


def _select_demucs(tier: str, stem_count: int = DEFAULT_STEM_COUNT) -> DemucsONNXSeparator:
    if tier not in _TIER_SUFFIX:
        raise NotImplementedError(f"tier {tier!r} not wired yet (htdemucs_ft 'best' tier is later work)")
    if stem_count not in MUSIC_MODELS:
        raise NotImplementedError(f"stem_count {stem_count!r} not supported (expected one of {tuple(MUSIC_MODELS)})")
    model_name = MUSIC_MODELS[stem_count]
    model_path = MODELS_DIR / f"{model_name}_core{_TIER_SUFFIX[tier]}.onnx"
    metadata_path = MODELS_DIR / f"{model_name}_core.json"
    if not model_path.exists():
        raise NotImplementedError(
            f"{model_path} not exported yet — run: uv run --group eval python scripts/export_onnx.py --model {model_name}"
        )
    return DemucsONNXSeparator(model_path=model_path, metadata_path=metadata_path, segment=TIERS[tier]["segment"])


def _select_bandit() -> Separator:
    # Bandit currently ships one checkpoint — tier doesn't affect it (yet).
    from backend.separators.bandit_sep import BanditSeparator

    return BanditSeparator()
