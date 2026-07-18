"""Pure-NumPy replica of htdemucs's STFT pre/post-processing.

This is the product-path counterpart to `_demucs_core.py`'s torch-based
spec/magnitude_cac/cac_to_complex/ispec: same math, no `torch` or `demucs`
import, so `demucs_onnx.py` never pulls the eval-only PyTorch dependency into
the product path (PRD §5: "ONNX Runtime... drops the heavy PyTorch dependency
at inference"). Every function here is checked against its torch counterpart
in `backend/tests/test_onnx_vs_oracle.py`.

Matches `torch.stft`/`torch.istft` with (center=True, pad_mode="reflect",
normalized=True, window=hann(n_fft, periodic)) exactly to float32 precision:
- forward STFT: reflect-pad by n_fft//2, frame at hop_length stride, window,
  rfft, divide by sqrt(n_fft).
- inverse STFT: multiply by sqrt(n_fft), irfft, window again, overlap-add,
  divide by the summed squared window (COLA normalization), crop the
  n_fft//2 padding back off.

On top of that base STFT/ISTFT, htdemucs's own `_spec`/`_ispec` (demucs/htdemucs.py)
add another layer of reflect-padding and slicing so that spectrogram frame
count divides evenly into hop_length; that layer is replicated here in
`spec`/`ispec` verbatim from the original.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal.windows import hann


def _stft(x: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Matches torch.stft(x, n_fft, hop_length, window=hann(n_fft), win_length=n_fft,
    normalized=True, center=True, return_complex=True, pad_mode="reflect")."""
    win = hann(n_fft, sym=False).astype(np.float64)
    pad = n_fft // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    length = xp.shape[-1]
    n_frames = 1 + (length - n_fft) // hop_length
    shape = x.shape[:-1] + (n_frames, n_fft)
    strides = xp.strides[:-1] + (hop_length * xp.strides[-1], xp.strides[-1])
    frames = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    frames = frames * win
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    spec = spec / np.sqrt(n_fft)
    return np.moveaxis(spec, -1, -2)  # (..., freq, frame)


def _istft(z: np.ndarray, n_fft: int, hop_length: int, length: int) -> np.ndarray:
    """Matches torch.istft(z, n_fft, hop_length, window=hann(n_fft), win_length=n_fft,
    normalized=True, length=length, center=True)."""
    win = hann(n_fft, sym=False).astype(np.float64)
    zt = np.moveaxis(z, -2, -1)  # (..., frame, freq)
    zt = zt * np.sqrt(n_fft)
    frames = np.fft.irfft(zt, n=n_fft, axis=-1)  # (..., frame, n_fft)
    frames = frames * win
    n_frames = frames.shape[-2]
    pad = n_fft // 2
    out_len = (n_frames - 1) * hop_length + n_fft
    out = np.zeros(frames.shape[:-2] + (out_len,), dtype=frames.dtype)
    win_sq_sum = np.zeros(out_len, dtype=np.float64)
    for t in range(n_frames):
        start = t * hop_length
        out[..., start : start + n_fft] += frames[..., t, :]
        win_sq_sum[start : start + n_fft] += win**2
    out = out / np.maximum(win_sq_sum, 1e-11)
    return out[..., pad : pad + length]


def spec(mix: np.ndarray, nfft: int, hop_length: int) -> np.ndarray:
    """Mirrors HTDemucs._spec exactly (see backend/separators/_demucs_core.py)."""
    hl = hop_length
    le = int(math.ceil(mix.shape[-1] / hl))
    pad = hl // 2 * 3
    x = np.pad(mix, [(0, 0)] * (mix.ndim - 1) + [(pad, pad + le * hl - mix.shape[-1])], mode="reflect")
    z = _stft(x, nfft, hl)[..., :-1, :]
    return z[..., 2 : 2 + le]


def magnitude_cac(z: np.ndarray) -> np.ndarray:
    """Mirrors HTDemucs._magnitude for cac=True: complex-as-channels."""
    b, c, fr, t = z.shape
    m = np.stack([z.real, z.imag], axis=-1)  # (b, c, fr, t, 2), matches view_as_real
    m = np.moveaxis(m, -1, 2)  # (b, c, 2, fr, t)
    return m.reshape(b, c * 2, fr, t).astype(np.float32)


def ispec(z: np.ndarray, hop_length: int, length: int) -> np.ndarray:
    """Mirrors HTDemucs._ispec at scale=0 (the only scale htdemucs uses)."""
    hl = hop_length
    z = np.pad(z, [(0, 0)] * (z.ndim - 2) + [(0, 1), (0, 0)])  # F.pad(z, (0,0,0,1)): freq axis
    z = np.pad(z, [(0, 0)] * (z.ndim - 2) + [(0, 0), (2, 2)])  # F.pad(z, (2,2)): time axis
    pad = hl // 2 * 3
    le = hl * int(math.ceil(length / hl)) + 2 * pad
    n_fft = 2 * (z.shape[-2] - 1)
    x = _istft(z, n_fft, hl, length=le)
    return x[..., pad : pad + length]


def cac_to_complex(m: np.ndarray) -> np.ndarray:
    """Mirrors HTDemucs._mask's cac branch. `m` is already split by source,
    shape (B, S, C, Fr, T) with C = 2 * audio_channels (real+imag packed)."""
    b, s, c, fr, t = m.shape
    out = m.reshape(b, s, -1, 2, fr, t)
    out = np.moveaxis(out, 3, -1)  # (b, s, c//2, fr, t, 2), matches .permute(0,1,2,4,5,3)
    return out[..., 0] + 1j * out[..., 1]
