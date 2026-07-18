"""htdemucs via ONNX Runtime — the product music separator.

router.py wires music mode + balanced/fast tier here. No `torch` or `demucs`
import: this module and its STFT helper (`_stft_numpy.py`) are pure
numpy + onnxruntime, so the product path never pulls in the eval-only
PyTorch dependency (PRD §5). `_oracle_torch.py` / `scripts/export_onnx.py`
are the only places that import `torch`/`demucs`.

The exported ONNX graph (`models/htdemucs_core.onnx`) has a fixed input shape
of exactly `training_length` samples (the model's native ~7.8s segment) — see
`_demucs_core.py` for why. Long audio is handled here with the same
segment/overlap-add scheme demucs itself uses (demucs/apply.py + utils.py's
`center_trim`): chunk the input at `config.segment` stride with 25% overlap.
Full-length chunks feed the graph as-is; a chunk shorter than
`training_length` (only possible at the tail of the file, or when the whole
clip is shorter than one segment) is *centered* within a training_length
window pulled from the surrounding real audio (falling back to zero-padding
only past the actual start/end of the file) — matching `TensorChunk.padded()`
— and the model's output is centered back down to the chunk's real length
with the same delta//2 crop `center_trim` uses. This isn't just style: getting
it wrong measurably changes the output at chunk boundaries (verified against
the PyTorch oracle in test_onnx_vs_oracle.py), since zero-padding shoves real
audio away from the position the STFT/model expects it in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from backend.config import INTRA_OP_THREADS, MODELS_DIR
from backend.separators._stft_numpy import cac_to_complex, ispec, magnitude_cac, spec
from backend.separators.base import Separator

DEFAULT_MODEL_PATH = MODELS_DIR / "htdemucs_core.onnx"
DEFAULT_METADATA_PATH = MODELS_DIR / "htdemucs_core.json"


class DemucsONNXSeparator(Separator):
    """audio: float32 ndarray, shape (channels, samples), stereo @ 44.1 kHz.
    separate() -> {"vocals" | "drums" | "bass" | "other": float32 (2, samples)}.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        segment: float | None = None,
        overlap: float = 0.25,
        threads: int | None = None,
    ):
        meta = json.loads(Path(metadata_path).read_text())
        self.sources: list[str] = meta["sources"]
        self.nfft: int = meta["nfft"]
        self.hop_length: int = meta["hop_length"]
        self.samplerate: int = meta["samplerate"]
        self.audio_channels: int = meta["audio_channels"]
        self.training_length: int = meta["training_length"]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or INTRA_OP_THREADS
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])

        # The ONNX graph's input shape is fixed at training_length; a smaller
        # configured segment only controls how much of each chunk's output we
        # keep (see _run_chunk), not the shape fed into the graph.
        requested = int((segment or self.training_length / self.samplerate) * self.samplerate)
        self.segment_length = min(requested, self.training_length)
        self.overlap = overlap

    def separate(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        mix = np.ascontiguousarray(audio, dtype=np.float32)
        if mix.ndim != 2:
            raise ValueError(f"expected (channels, samples), got shape {mix.shape}")
        length = mix.shape[-1]

        stride = max(int((1 - self.overlap) * self.segment_length), 1)
        weight = _crossfade_weight(self.segment_length)

        n_sources = len(self.sources)
        out = np.zeros((n_sources, self.audio_channels, length), dtype=np.float64)
        sum_weight = np.zeros(length, dtype=np.float64)

        for offset in range(0, length, stride):
            chunk_len = min(self.segment_length, length - offset)
            chunk_out = self._run_chunk(mix, offset, chunk_len)  # (S, C, chunk_len)
            w = weight[:chunk_len]
            out[:, :, offset : offset + chunk_len] += chunk_out * w
            sum_weight[offset : offset + chunk_len] += w

        out /= np.maximum(sum_weight, 1e-8)
        return {name: out[i].astype(np.float32) for i, name in enumerate(self.sources)}

    def _run_chunk(self, mix: np.ndarray, offset: int, chunk_len: int) -> np.ndarray:
        """Feed one training_length window to the ONNX graph and return the
        model's output cropped back down to `chunk_len`, matching demucs's
        own TensorChunk.padded()/center_trim (see module docstring)."""
        total_length = mix.shape[-1]
        delta = self.training_length - chunk_len
        start = offset - delta // 2
        end = start + self.training_length
        correct_start = max(0, start)
        correct_end = min(total_length, end)
        pad_left = correct_start - start
        pad_right = end - correct_end

        windowed = mix[:, correct_start:correct_end]
        padded = np.pad(windowed, [(0, 0), (pad_left, pad_right)])
        mix_in = padded[None].astype(np.float32)  # (1, C, training_length)

        z = spec(mix_in, self.nfft, self.hop_length)
        mag = magnitude_cac(z).astype(np.float32)

        x_out, xt_out = self.session.run(["x_out", "xt_out"], {"mag": mag, "mix": mix_in})
        zout = cac_to_complex(x_out)
        x_rec = ispec(zout, self.hop_length, self.training_length)
        full = xt_out + x_rec  # (1, S, C, training_length)

        crop_start = delta // 2
        return full[0, :, :, crop_start : crop_start + chunk_len]


def _crossfade_weight(segment_length: int) -> np.ndarray:
    """Triangle-shaped overlap-add weight, maximal at the segment center —
    the same scheme demucs/apply.py uses so adjacent chunks blend instead of
    clicking at their boundaries."""
    half = segment_length // 2
    ramp = np.concatenate([np.arange(1, half + 1), np.arange(segment_length - half, 0, -1)]).astype(np.float64)
    return ramp / ramp.max()
