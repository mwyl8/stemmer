"""Bandit-family cinematic separator (speech/music/effects) — PyTorch fallback.

No ONNX export exists for this model yet. It's a complex-mask band-split RNN
— the STFT step returns a complex spectrogram consumed directly by the mask
estimator, the same shape of problem that forced Demucs's STFT-split ONNX
rewrite in Phase 1 (`_demucs_core.py`). Doing that rewrite again for a whole
different, less-documented architecture is real work, and PRD §5 explicitly
sanctions the alternative for Bandit specifically: "ONNX where an export
exists; otherwise PyTorch behind the same interface with a note to export in
a follow-up." That's what this is — same caveat as the "fast" tier's int8
quantization quality question in Phase 1: it works, it's correct, it's just
not yet the optimized CPU path. ONNX-exporting this is a concrete Phase 6
item, not a vague one — the model is small (band-split + GRU, no
transformer) and its own STFT/ISTFT could very likely follow the exact same
split-the-graph-at-the-STFT-boundary approach as `_demucs_core.py`, given
another pass of the same effort.

Model: a BandSplitRNN reimplementation of BandIt (Watcharasupat et al.),
trained on DnR v3, vendored from ZFTurbo/Music-Source-Separation-Training
(MIT license) at `_bandit_vendor/` — see that package's docstring for why
vendored rather than depending on the upstream research repos directly.
Native 44.1kHz stereo, same rate as the Demucs path, so audio from ingest()
feeds it directly with no resampling.

Needs the `speech` dependency group (torch/torchaudio/pytorch-lightning/
spafe) — not installed by default, same posture as `eval`'s torch. Imported
lazily by router.py so selecting music mode never requires it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml

from backend.config import MODELS_DIR
from backend.separators._bandit_vendor.model import MultiMaskMultiSourceBandSplitRNNSimple
from backend.separators.base import Separator

DEFAULT_CHECKPOINT_PATH = MODELS_DIR / "bandit" / "model_bandit_plus_dnr_sdr_11.47.chpt"
DEFAULT_CONFIG_PATH = MODELS_DIR / "bandit" / "config_dnr_bandit_bsrnn_multi_mus64.yaml"


class BanditSeparator(Separator):
    """audio: float32 ndarray, shape (channels, samples), stereo @ 44.1 kHz.
    separate() -> {"speech" | "music" | "effects": float32 (2, samples)}.
    """

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
        segment_seconds: float | None = None,
        overlap: float = 0.25,
        device: str = "cpu",
    ):
        config = yaml.safe_load(Path(config_path).read_text())
        self.samplerate: int = config["audio"]["sample_rate"]
        self.sources: list[str] = list(config["model"]["stems"])
        self.chunk_size: int = int(segment_seconds * self.samplerate) if segment_seconds else config["audio"]["chunk_size"]
        self.overlap = overlap
        self.device = device

        self.model = MultiMaskMultiSourceBandSplitRNNSimple(**config["model"])
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)

    def num_chunks(self, length: int) -> int:
        """Chunk count for an input of `length` samples — see
        DemucsONNXSeparator.num_chunks for why this depends only on length,
        not audio content."""
        stride = max(int((1 - self.overlap) * self.chunk_size), 1)
        return len(range(0, length, stride))

    def separate(self, audio: np.ndarray, on_chunk: Callable[[int, int], None] | None = None) -> dict[str, np.ndarray]:
        mix = np.ascontiguousarray(audio, dtype=np.float32)
        if mix.ndim != 2:
            raise ValueError(f"expected (channels, samples), got shape {mix.shape}")
        length = mix.shape[-1]

        stride = max(int((1 - self.overlap) * self.chunk_size), 1)
        weight = _crossfade_weight(self.chunk_size)
        offsets = list(range(0, length, stride))
        total = len(offsets)
        if on_chunk is not None:
            on_chunk(0, total)

        n_sources = len(self.sources)
        n_channels = mix.shape[0]
        out = np.zeros((n_sources, n_channels, length), dtype=np.float64)
        sum_weight = np.zeros(length, dtype=np.float64)

        for i, offset in enumerate(offsets):
            chunk_len = min(self.chunk_size, length - offset)
            chunk_out = self._run_chunk(mix, offset, chunk_len)  # (S, C, chunk_len)
            w = weight[:chunk_len]
            out[:, :, offset : offset + chunk_len] += chunk_out * w
            sum_weight[offset : offset + chunk_len] += w
            if on_chunk is not None:
                on_chunk(i + 1, total)

        out /= np.maximum(sum_weight, 1e-8)
        return {name: out[i].astype(np.float32) for i, name in enumerate(self.sources)}

    def _run_chunk(self, mix: np.ndarray, offset: int, chunk_len: int) -> np.ndarray:
        """Center the chunk in a full chunk_size window pulled from the
        surrounding real audio (falling back to zero-padding only past the
        actual start/end of the file), run the model, then crop the output
        back down with the matching center offset. Same scheme as
        demucs_onnx.py's _run_chunk — see that module's docstring for why
        zero-tail-padding would be wrong here."""
        total_length = mix.shape[-1]
        delta = self.chunk_size - chunk_len
        start = offset - delta // 2
        end = start + self.chunk_size
        correct_start = max(0, start)
        correct_end = min(total_length, end)
        pad_left = correct_start - start
        pad_right = end - correct_end

        windowed = mix[:, correct_start:correct_end]
        padded = np.pad(windowed, [(0, 0), (pad_left, pad_right)])
        batch = torch.from_numpy(padded[None]).to(self.device)  # (1, C, chunk_size)

        with torch.no_grad():
            res = self.model(batch)  # (1, S, C, chunk_size)

        crop_start = delta // 2
        return res[0, :, :, crop_start : crop_start + chunk_len].cpu().numpy()


def _crossfade_weight(chunk_size: int) -> np.ndarray:
    """Triangle-shaped overlap-add weight, maximal at the chunk center —
    same scheme as demucs_onnx.py / demucs's own apply.py."""
    half = chunk_size // 2
    ramp = np.concatenate([np.arange(1, half + 1), np.arange(chunk_size - half, 0, -1)]).astype(np.float64)
    return ramp / ramp.max()
