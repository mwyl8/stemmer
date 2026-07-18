"""Shared htdemucs STFT pre/post-processing and the ONNX-exportable "core" net.

`torch.onnx.export` cannot trace complex-valued ops — `aten::stft`/`aten::istft`
with complex output, and `view_as_complex` — so HTDemucs.forward() can't be
exported as-is (confirmed: both the legacy tracer and the dynamo exporter fail
on it directly). The fix, following the approach used by sevagh/demucs.onnx and
the Mixxx GSoC 2025 htdemucs export: split the model at the STFT boundary.
Everything from the magnitude spectrogram through the conv/transformer network
to the pre-ISTFT output is pure real-valued tensor math and exports cleanly
(`DemucsCore` below); the STFT, complex-as-channels reshape, and ISTFT stay in
plain PyTorch/numpy outside the graph (cheap FFT work, not what needs
acceleration). `demucs_onnx.py` calls `spec`/`magnitude_cac` before the ONNX
session and `cac_to_complex`/`ispec` after it; `scripts/export_onnx.py` traces
`DemucsCore` directly against the same pretrained submodules so weights match
the PyTorch oracle exactly.

htdemucs specifics this relies on (verified against the pretrained checkpoint):
cac=True (complex-as-channels, not Wiener filtering) and wiener_iters<=0, so
`_mask`'s cac branch is always taken and the true complex spectrogram `z` is
never needed for reconstruction — only its post-STFT magnitude is.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from demucs.hdemucs import pad1d
from demucs.spec import ispectro, spectro


def spec(mix: torch.Tensor, nfft: int, hop_length: int) -> torch.Tensor:
    """Mirrors HTDemucs._spec: STFT with the padding convention that keeps
    output frame count == input length / hop_length (when divisible)."""
    hl = hop_length
    le = int(math.ceil(mix.shape[-1] / hl))
    pad = hl // 2 * 3
    x = pad1d(mix, (pad, pad + le * hl - mix.shape[-1]), mode="reflect")
    z = spectro(x, nfft, hl)[..., :-1, :]
    z = z[..., 2 : 2 + le]
    return z


def magnitude_cac(z: torch.Tensor) -> torch.Tensor:
    """Mirrors HTDemucs._magnitude for cac=True: complex-as-channels."""
    b, c, fr, t = z.shape
    m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
    return m.reshape(b, c * 2, fr, t)


def ispec(z: torch.Tensor, hop_length: int, length: int) -> torch.Tensor:
    """Mirrors HTDemucs._ispec at scale=0 (the only scale htdemucs uses)."""
    hl = hop_length
    z = F.pad(z, (0, 0, 0, 1))
    z = F.pad(z, (2, 2))
    pad = hl // 2 * 3
    le = hl * int(math.ceil(length / hl)) + 2 * pad
    x = ispectro(z, hl, length=le)
    return x[..., pad : pad + length]


def cac_to_complex(m: torch.Tensor) -> torch.Tensor:
    """Mirrors HTDemucs._mask's cac branch. `m` is already split by source,
    shape (B, S, C, Fr, T) with C = 2 * audio_channels (real+imag packed)."""
    b, s, c, fr, t = m.shape
    out = m.view(b, s, -1, 2, fr, t).permute(0, 1, 2, 4, 5, 3)
    return torch.view_as_complex(out.contiguous())


class DemucsCore(nn.Module):
    """The exportable subset of HTDemucs: magnitude-spectrogram + raw waveform
    in, pre-ISTFT spectrogram + time-branch waveform out. Holds the pretrained
    submodules directly (no copies) so ONNX weights are bit-identical to the
    PyTorch oracle.
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.tencoder = model.tencoder
        self.decoder = model.decoder
        self.tdecoder = model.tdecoder
        self.freq_emb = model.freq_emb
        self.freq_emb_scale = model.freq_emb_scale
        self.crosstransformer = model.crosstransformer
        self.bottom_channels = model.bottom_channels
        if self.bottom_channels:
            self.channel_upsampler = model.channel_upsampler
            self.channel_downsampler = model.channel_downsampler
            self.channel_upsampler_t = model.channel_upsampler_t
            self.channel_downsampler_t = model.channel_downsampler_t
        self.depth = model.depth
        self.n_sources = len(model.sources)

    def forward(self, mag: torch.Tensor, mix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, fq, t = mag.shape
        x = mag
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved = []
        saved_t = []
        lengths = []
        lengths_t = []
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        if self.crosstransformer:
            if self.bottom_channels:
                _, _, f, _ = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_downsampler_t(xt)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        s = self.n_sources
        x = x.view(b, s, -1, fq, t)
        x = x * std[:, None] + mean[:, None]

        training_length = xt.shape[-1]
        xt = xt.view(b, s, -1, training_length)
        xt = xt * stdt[:, None] + meant[:, None]

        return x, xt
