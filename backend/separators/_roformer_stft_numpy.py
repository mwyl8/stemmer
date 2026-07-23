"""Pure-NumPy replica of MelBandRoformer's STFT pre/post-processing and
mel-band frequency gather/scatter — the product-path counterpart to
`_roformer_core.py`'s torch-based equivalents, so `roformer_onnx.py` (once
that lands) never pulls the eval-only PyTorch dependency into the product
path. Same principle as `_stft_numpy.py` for htdemucs, but a different (and
simpler) STFT convention: MelBandRoformer uses a single plain
`torch.stft(..., center=True, pad_mode="reflect")`/`torch.istft(...)`, not
htdemucs's extra frame-alignment padding layer — so this is NOT a drop-in
reuse of `_stft_numpy.py`'s `spec`/`ispec` (those replicate htdemucs-specific
padding); only the underlying `_stft`/`_istft` shapes are analogous, and even
those differ in the `normalized` flag this checkpoint sets false.

Every function here is checked against its torch counterpart in
backend/tests/test_roformer_onnx_vs_oracle.py.
"""

from __future__ import annotations

import numpy as np
from librosa import filters
from scipy.signal.windows import hann


def stft(x: np.ndarray, n_fft: int, hop_length: int, win_length: int, normalized: bool = False) -> np.ndarray:
    """Matches torch.stft(x, n_fft, hop_length, win_length, window=hann(win_length),
    center=True, pad_mode="reflect", normalized=normalized, return_complex=True).
    `x`: (..., samples) real. Returns complex128 (..., freq, frame)."""
    win = hann(win_length, sym=False).astype(np.float64)
    if win_length < n_fft:
        pad_amount = n_fft - win_length
        win = np.pad(win, (pad_amount // 2, pad_amount - pad_amount // 2))
    pad = n_fft // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    length = xp.shape[-1]
    n_frames = 1 + (length - n_fft) // hop_length
    shape = x.shape[:-1] + (n_frames, n_fft)
    strides = xp.strides[:-1] + (hop_length * xp.strides[-1], xp.strides[-1])
    frames = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    frames = frames * win
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    if normalized:
        spec = spec / np.sqrt(n_fft)
    return np.moveaxis(spec, -1, -2)  # (..., freq, frame)


def istft(
    z: np.ndarray, n_fft: int, hop_length: int, win_length: int, length: int, normalized: bool = False
) -> np.ndarray:
    """Matches torch.istft(z, n_fft, hop_length, win_length, window=hann(win_length),
    center=True, normalized=normalized, length=length)."""
    win = hann(win_length, sym=False).astype(np.float64)
    if win_length < n_fft:
        pad_amount = n_fft - win_length
        win = np.pad(win, (pad_amount // 2, pad_amount - pad_amount // 2))
    zt = np.moveaxis(z, -2, -1)  # (..., frame, freq)
    if normalized:
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


def mel_band_indices(sample_rate: int, n_fft: int, num_bands: int, stereo: bool) -> tuple[np.ndarray, np.ndarray]:
    """Replicates MelBandRoformer.__init__'s mel-filterbank-derived band
    gather indices (mel_band_roformer.py lines ~459-497) in plain numpy —
    deterministic given (sample_rate, n_fft, num_bands), so this never needs
    to touch torch/the checkpoint at all. Returns (freq_indices,
    num_bands_per_freq): `freq_indices` gathers stft bins (folded with the
    stereo channel, if stereo) into band order for `band_split`;
    `num_bands_per_freq` is the per-frequency-bin overlap count used to
    average overlapping band mask estimates back down in `combine_masks`."""
    freqs = n_fft // 2 + 1

    mel_filter_bank = filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_bands)
    mel_filter_bank[0][0] = 1.0
    mel_filter_bank[-1, -1] = 1.0

    freqs_per_band = mel_filter_bank > 0  # (num_bands, freqs) bool
    assert freqs_per_band.any(axis=0).all(), "all frequencies need to be covered by all bands"

    repeated_freq_indices = np.tile(np.arange(freqs), (num_bands, 1))
    freq_indices = repeated_freq_indices[freqs_per_band]

    if stereo:
        freq_indices = np.repeat(freq_indices, 2)
        freq_indices = freq_indices * 2 + np.tile(np.arange(2), len(freq_indices) // 2)

    num_bands_per_freq = freqs_per_band.sum(axis=0)  # (freqs,)
    return freq_indices.astype(np.int64), num_bands_per_freq.astype(np.float64)


def gather_bands(stft_repr: np.ndarray, freq_indices: np.ndarray) -> np.ndarray:
    """stft_repr: (b, f_folded, t, 2) real+imag, f_folded = freq*channels
    (stereo-interleaved). Returns (b, t, f*c) real-valued band-gathered
    features ready for `MelBandRoformerCore` — mirrors mel_band_roformer.py's
    `x = stft_repr[batch_arange, self.freq_indices]` followed by
    `rearrange(x, 'b f t c -> b t (f c)')`."""
    x = stft_repr[:, freq_indices]  # (b, f_gathered, t, 2)
    x = np.transpose(x, (0, 2, 1, 3))  # (b, t, f, c) — c must stay last/fastest so (f c) flattens with c fast, f slow
    b, t, f, c = x.shape
    return np.ascontiguousarray(x).reshape(b, t, f * c).astype(np.float32)


def combine_masks(
    masks: np.ndarray, freq_indices: np.ndarray, num_bands_per_freq: np.ndarray, num_freqs: int, num_channels: int
) -> np.ndarray:
    """masks: (b, n, t, f_gathered*2) real-valued Core output (n = num_stems).
    Scatters each band's mask estimate back to its full-frequency-bin
    position and averages overlapping bands — mirrors mel_band_roformer.py's
    `scatter_add_`/`masks_summed / denom` step. Returns (b, n, num_freqs*
    num_channels, t) complex128, ready to multiply with the original STFT."""
    b, n, t, fc = masks.shape
    f = fc // 2
    masks = masks.reshape(b, n, t, f, 2)
    masks_complex = masks[..., 0] + 1j * masks[..., 1]  # (b, n, t, f)
    masks_complex = np.moveaxis(masks_complex, 2, -1)  # (b, n, f, t)

    full_bins = num_freqs * num_channels
    summed = np.zeros((b, n, full_bins, t), dtype=np.complex128)
    for i, bin_idx in enumerate(freq_indices):
        summed[:, :, bin_idx, :] += masks_complex[:, :, i, :]

    denom = np.repeat(num_bands_per_freq, num_channels)
    denom = np.clip(denom, 1e-8, None)
    return summed / denom[None, None, :, None]


def zero_dc(modulated: np.ndarray) -> np.ndarray:
    """Zeroes the DC frequency bin (index 0 along the frequency axis, axis
    -2) — mirrors mel_band_roformer.py's `zero_dc=True` default (this
    checkpoint's actual setting), applied post-modulation, pre-ISTFT.
    `modulated`: (..., freq, frame) complex, per-channel (not band-folded)."""
    out = modulated.copy()
    out[..., 0, :] = 0.0
    return out
