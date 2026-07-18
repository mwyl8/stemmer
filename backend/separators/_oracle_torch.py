"""PyTorch Demucs oracle. REFERENCE ONLY.

This is the eval harness's ground truth for the ONNX/quantized product path
(`demucs_onnx.py`) to be diffed against. It is never reachable from the normal
mode+tier routing in `router.py` — only from the eval path. Do not wire this
into request handling; PyTorch is not the product inference path (PRD §5,
CLAUDE.md §6).
"""

from __future__ import annotations

import numpy as np
import torch

from backend.separators.base import Separator

MODEL_NAME = "htdemucs"


class OracleTorchSeparator(Separator):
    """Loads htdemucs via the demucs Python API and separates on CPU.

    audio: float32 ndarray, shape (channels, samples), stereo @ 44.1 kHz.
    separate() -> {"vocals" | "drums" | "bass" | "other": float32 (2, samples)}.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu", shifts: int = 0):
        from demucs.api import Separator as DemucsApiSeparator

        self._api = DemucsApiSeparator(model=model_name, device=device, shifts=shifts, progress=False)

    @property
    def samplerate(self) -> int:
        return self._api.samplerate

    def separate(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        _, stems = self._api.separate_tensor(wav, sr=self._api.samplerate)
        return {name: source.numpy().astype(np.float32) for name, source in stems.items()}
