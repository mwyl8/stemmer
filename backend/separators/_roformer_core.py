"""The ONNX-exportable subset of MelBandRoformer — same split strategy as
`_demucs_core.py` (see that module's docstring for the general rationale):
`torch.onnx.export` cannot trace complex-valued ops, and
MelBandRoformer.forward() does `torch.stft(..., return_complex=True)` /
`torch.istft(...)` plus `torch.view_as_complex`/`view_as_real` around the
mask application — the exact same class of blocker that forced htdemucs's
STFT-split rewrite. Investigated directly here rather than assumed: the rest
of the model (BandSplit, the RoPE transformer stack, MaskEstimator) is pure
real-valued tensor math — `rotary_embedding_torch`'s `apply_rotary_emb` is
cos/sin/concat, no complex dtype anywhere — so splitting at the same
STFT/complex boundary Demucs used works here too.

`MelBandRoformerCore` holds the pretrained `band_split`, transformer
`layers`, and `mask_estimators` submodules directly (no copies, so ONNX
weights are bit-identical to the checkpoint). The STFT, mel-band frequency
gather, and mask-to-waveform reconstruction stay in
`_roformer_stft_numpy.py` (product path) / this module's `stft_pre`/
`mask_post` (PyTorch reference, used only by scripts/export_roformer_onnx.py
and the oracle test) outside the traced graph.

Relies on `zero_dc=True` (this checkpoint's default — MelBandRoformer zeroes
the DC frequency bin post-modulation, pre-ISTFT) and `match_input_audio_length=False`
(also this checkpoint's default — the model doesn't force ISTFT to the exact
input sample count; in practice, for this STFT config, the round-trip length
already matches). Both assumptions are exercised by the pre-export exact-diff
check in scripts/export_roformer_onnx.py, same posture as `_demucs_core.py`'s
documented cac/wiener_iters assumptions.
"""

from __future__ import annotations

import torch
from einops import pack, rearrange, unpack
from torch import nn


class MelBandRoformerCore(nn.Module):
    """band-gathered STFT features in -> per-stem masks out. See module
    docstring: everything before/after this is STFT/complex-number work that
    stays outside the ONNX graph."""

    def __init__(self, model):
        super().__init__()
        self.band_split = model.band_split
        self.layers = model.layers
        self.mask_estimators = model.mask_estimators

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (b, t, f*c) real, already mel-band-gathered and complex-folded
        (see `_roformer_stft_numpy.gather_bands`). Returns (b, n, t, f*c)
        real-valued per-stem mask estimates (n = num_stems)."""
        x = self.band_split(x)

        for transformer_block in self.layers:
            time_transformer, freq_transformer = transformer_block

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        return masks


def stft_pre(raw_audio: torch.Tensor, model) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference (eval-only) counterpart of `_roformer_stft_numpy`'s
    `stft` + `gather_bands`, used solely to verify the numpy product path
    against the real model in scripts/export_roformer_onnx.py /
    test_roformer_onnx_vs_oracle.py — never imported by the product
    separator. Returns (core_input, stft_repr) where `stft_repr` (complex,
    (b, f*s, t)) is what `mask_post` needs to reconstruct the waveform.
    """
    stft_window = model.stft_window_fn(device=raw_audio.device)
    stft_repr = torch.stft(raw_audio.reshape(-1, raw_audio.shape[-1]), **model.stft_kwargs, window=stft_window, return_complex=True)
    b, s = raw_audio.shape[0], raw_audio.shape[1]
    stft_repr = stft_repr.reshape(b, s, *stft_repr.shape[-2:])  # b s f t
    stft_repr_real = torch.view_as_real(stft_repr)  # b s f t c
    stft_repr_folded = rearrange(stft_repr_real, "b s f t c -> b (f s) t c")

    batch_arange = torch.arange(b, device=raw_audio.device)[..., None]
    x = stft_repr_folded[batch_arange, model.freq_indices]
    x = rearrange(x, "b f t c -> b t (f c)")
    return x, rearrange(stft_repr, "b s f t -> b (f s) t")


def mask_post(masks: torch.Tensor, stft_repr: torch.Tensor, model, audio_length: int) -> torch.Tensor:
    """PyTorch reference counterpart of `_roformer_stft_numpy.combine_masks`
    + istft — see `stft_pre`'s docstring."""
    masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)
    stft_repr = rearrange(stft_repr, "b f t -> b 1 f t")
    stft_repr = torch.view_as_complex(torch.view_as_real(stft_repr))
    masks = torch.view_as_complex(masks.contiguous()).type(stft_repr.dtype)

    num_stems = masks.shape[1]
    batch = masks.shape[0]
    scatter_indices = model.freq_indices[None, None, :, None].expand(batch, num_stems, -1, stft_repr.shape[-1])
    stft_repr_expanded_stems = stft_repr.expand(-1, num_stems, -1, -1)
    masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

    denom = model.num_bands_per_freq.repeat_interleave(model.audio_channels).reshape(-1, 1)
    masks_averaged = masks_summed / denom.clamp(min=1e-8)

    stft_repr = stft_repr * masks_averaged
    stft_repr = rearrange(stft_repr, "b n (f s) t -> (b n s) f t", s=model.audio_channels)
    if model.zero_dc:
        stft_repr = stft_repr.index_fill(1, torch.tensor(0, device=stft_repr.device), 0.0)
    stft_window = model.stft_window_fn(device=stft_repr.device)
    recon = torch.istft(stft_repr, **model.stft_kwargs, window=stft_window, return_complex=False, length=audio_length)
    return rearrange(recon, "(b n s) t -> b n s t", b=stft_repr.shape[0] // (num_stems * model.audio_channels), s=model.audio_channels, n=num_stems)
