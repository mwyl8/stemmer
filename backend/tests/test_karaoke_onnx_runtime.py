"""KaraokeONNXSeparator construction, provider selection, and runtime_info()
— same coverage as test_demucs_onnx_runtime.py for the Demucs path. Needs
the exported karaoke ONNX model — skips cleanly if
scripts/export_roformer_onnx.py hasn't been run yet.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.arch import HostArch
from backend.config import MODELS_DIR
from backend.separators.karaoke_onnx import KaraokeONNXSeparator

ONNX_PATH = MODELS_DIR / "karaoke_core.onnx"
METADATA_PATH = MODELS_DIR / "karaoke_core.json"

pytestmark = pytest.mark.skipif(not ONNX_PATH.exists(), reason=f"{ONNX_PATH} not exported")


def test_default_providers_resolve_through_arch_for_arm64():
    host = HostArch(arch="arm64", has_avx512=False, has_vnni=False)
    sep = KaraokeONNXSeparator(model_path=ONNX_PATH, metadata_path=METADATA_PATH, host_arch=host)
    info = sep.runtime_info()
    assert info["arch"] == "arm64"
    assert info["provider"] == "CPUExecutionProvider"
    assert info["model"] == "karaoke_core"


def test_sources_are_relabeled_lead_and_backing():
    sep = KaraokeONNXSeparator(model_path=ONNX_PATH, metadata_path=METADATA_PATH)
    assert sep.sources == ["lead_vocal", "backing_vocals"]


def test_separate_on_a_short_silent_clip_produces_correct_shapes_and_no_nans():
    """Not a quality check (silence in, silence-ish out) — just that the
    chunking/overlap-add plumbing runs end to end without shape errors or
    NaNs, on a clip shorter than one training_length segment (the boundary
    case DemucsONNXSeparator's own docstring calls out)."""
    sep = KaraokeONNXSeparator(model_path=ONNX_PATH, metadata_path=METADATA_PATH)
    audio = np.zeros((2, 44100 * 2), dtype=np.float32)  # 2s, well under training_length
    stems = sep.separate(audio)
    assert set(stems.keys()) == {"lead_vocal", "backing_vocals"}
    for name, wav in stems.items():
        assert wav.shape == (2, 44100 * 2)
        assert not np.isnan(wav).any()
        assert not np.isinf(wav).any()


def test_num_chunks_matches_actual_progress_callback_total():
    sep = KaraokeONNXSeparator(model_path=ONNX_PATH, metadata_path=METADATA_PATH)
    audio = np.zeros((2, 44100 * 2), dtype=np.float32)
    expected_total = sep.num_chunks(audio.shape[-1])

    seen_totals = set()
    sep.separate(audio, on_chunk=lambda done, total: seen_totals.add(total))
    assert seen_totals == {expected_total}
