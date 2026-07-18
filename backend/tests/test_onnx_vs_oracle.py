"""Diff the ONNX product path (demucs_onnx.py) against the PyTorch oracle
(_oracle_torch.py) on a short clip.

Needs the eval extras (torch/demucs) to load the oracle, and an exported
ONNX model (scripts/export_onnx.py) — both opt-in, so this test skips
cleanly rather than failing a default (product-only) install.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.config import DATA_DIR, MODELS_DIR, MUSIC_SAMPLE_RATE

pytest.importorskip("torch")
pytest.importorskip("demucs")

ONNX_MODEL_PATH = MODELS_DIR / "htdemucs_core.onnx"
TEST_AUDIO_PATH = DATA_DIR / "test.mp3"
CLIP_SECONDS = 5  # well under the ~7.8s training window: exactly one chunk on
# both paths, so the diff isolates model/STFT numerics from chunking behavior.
MAX_ABS_DIFF_TOLERANCE = 0.01  # measured ~5e-4 on random noise; real audio gets more headroom

pytestmark = pytest.mark.skipif(
    not ONNX_MODEL_PATH.exists(),
    reason="models/htdemucs_core.onnx not exported — run: uv run --group eval python scripts/export_onnx.py",
)


@pytest.fixture(scope="module")
def short_clip() -> np.ndarray:
    if not TEST_AUDIO_PATH.exists():
        pytest.skip(f"{TEST_AUDIO_PATH} not present")
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "clip.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(TEST_AUDIO_PATH),
                "-t",
                str(CLIP_SECONDS),
                "-ac",
                "2",
                "-ar",
                str(MUSIC_SAMPLE_RATE),
                str(wav_path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        audio, _sr = sf.read(wav_path, dtype="float32", always_2d=True)
    return audio.T  # (channels, samples)


def test_onnx_matches_oracle_on_short_clip(short_clip):
    from backend.separators._oracle_torch import OracleTorchSeparator
    from backend.separators.demucs_onnx import DemucsONNXSeparator

    oracle_stems = OracleTorchSeparator().separate(short_clip)
    onnx_stems = DemucsONNXSeparator().separate(short_clip)

    assert set(oracle_stems) == set(onnx_stems)
    for name in oracle_stems:
        diff = np.abs(oracle_stems[name] - onnx_stems[name]).max()
        assert diff < MAX_ABS_DIFF_TOLERANCE, f"{name}: max-abs-diff {diff} exceeds tolerance"
