"""MelBandRoformer karaoke model via ONNX Runtime — the product Lead-vs-
Backing-Vocals splitter (Lead-vs-Backing-Vocals mode chains this after
Demucs isolates the vocal bus — see lead_backing_sep.py).

router.py wires "karaoke" mode here. No `torch` import: this module and its
STFT helper (`_roformer_stft_numpy.py`) are pure numpy + onnxruntime, same
principle as demucs_onnx.py keeping the eval-only PyTorch dependency
(rotary_embedding_torch, beartype, the vendored model) out of the product
path entirely — see `_roformer_core.py` / `scripts/export_roformer_onnx.py`
for where those live instead.

This checkpoint was trained on full mixes as "Vocals" (lead) vs.
"Instrumental" (everything else, including backing vocals + music) — the
community calls it a "karaoke" model because that's exactly a karaoke
track's split (remove only the lead). Fed the *isolated vocal bus* instead
of a full mix (this mode's whole point), "Instrumental" becomes "whatever
isn't the lead vocal in a bus that has no real instruments left" — i.e.
backing vocals/harmonies. `sources` below is relabeled ["lead_vocal",
"backing_vocals"] to match that repurposing; see scripts/export_roformer_onnx.py's
metadata for exactly which upstream checkpoint labels those map from.

The exported ONNX graph (`models/karaoke_core.onnx`) has a fixed input shape
of exactly `training_length` samples (see `_roformer_core.py` for why —
same class of STFT/complex-number export blocker htdemucs had, same fix).
Long audio uses the same segment/overlap-add scheme as demucs_onnx.py
(25% overlap, chunks centered in a training_length window pulled from
surrounding real audio, cropped back with the same delta//2 convention) —
see that module's docstring for the rationale.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort

from backend.arch import HostArch, detect_host_arch, resolve_providers
from backend.config import INTRA_OP_THREADS, MODELS_DIR
from backend.separators._roformer_stft_numpy import combine_masks, gather_bands, istft, mel_band_indices, stft, zero_dc
from backend.separators.base import Separator

DEFAULT_MODEL_PATH = MODELS_DIR / "karaoke_core.onnx"
DEFAULT_METADATA_PATH = MODELS_DIR / "karaoke_core.json"


class KaraokeONNXSeparator(Separator):
    """audio: float32 ndarray, shape (2, samples), stereo @ 44.1 kHz — meant
    to be Demucs's isolated "vocals" stem, not a full mix (see module
    docstring). separate() -> {"lead_vocal" | "backing_vocals": float32 (2, samples)}.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        overlap: float = 0.25,
        threads: int | None = None,
        providers: tuple[str, ...] | None = None,
        host_arch: HostArch | None = None,
    ):
        meta = json.loads(Path(metadata_path).read_text())
        self.sources: list[str] = meta["sources"]
        self.n_fft: int = meta["n_fft"]
        self.hop_length: int = meta["hop_length"]
        self.win_length: int = meta["win_length"]
        self.normalized: bool = meta["normalized"]
        self.num_bands: int = meta["num_bands"]
        self.sample_rate: int = meta["sample_rate"]
        self.audio_channels: int = meta["audio_channels"]
        self.training_length: int = meta["training_length"]
        self.overlap = overlap

        self.freq_indices, self.num_bands_per_freq = mel_band_indices(
            self.sample_rate, self.n_fft, self.num_bands, stereo=(self.audio_channels == 2)
        )

        self._host_arch = host_arch or detect_host_arch()
        self._model_name = Path(model_path).stem
        resolved_providers = list(providers) if providers is not None else list(resolve_providers(self._host_arch))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or INTRA_OP_THREADS
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=resolved_providers)

    def num_chunks(self, length: int) -> int:
        """Chunk count for an input of `length` samples — see
        DemucsONNXSeparator.num_chunks for why this depends only on length,
        not audio content."""
        stride = max(int((1 - self.overlap) * self.training_length), 1)
        return len(range(0, length, stride))

    def runtime_info(self) -> dict[str, str]:
        active_providers = self.session.get_providers()
        provider = active_providers[0] if active_providers else "CPUExecutionProvider"
        return {"arch": self._host_arch.profile_key, "provider": provider, "model": self._model_name}

    def separate(self, audio: np.ndarray, on_chunk: Callable[[int, int], None] | None = None) -> dict[str, np.ndarray]:
        mix = np.ascontiguousarray(audio, dtype=np.float32)
        if mix.ndim != 2:
            raise ValueError(f"expected (channels, samples), got shape {mix.shape}")
        length = mix.shape[-1]

        stride = max(int((1 - self.overlap) * self.training_length), 1)
        weight = _crossfade_weight(self.training_length)
        offsets = list(range(0, length, stride))
        total = len(offsets)
        if on_chunk is not None:
            on_chunk(0, total)

        n_sources = len(self.sources)
        out = np.zeros((n_sources, self.audio_channels, length), dtype=np.float64)
        sum_weight = np.zeros(length, dtype=np.float64)

        for i, offset in enumerate(offsets):
            chunk_len = min(self.training_length, length - offset)
            chunk_out = self._run_chunk(mix, offset, chunk_len)  # (S, C, chunk_len)
            w = weight[:chunk_len]
            out[:, :, offset : offset + chunk_len] += chunk_out * w
            sum_weight[offset : offset + chunk_len] += w
            if on_chunk is not None:
                on_chunk(i + 1, total)

        out /= np.maximum(sum_weight, 1e-8)
        return {name: out[i].astype(np.float32) for i, name in enumerate(self.sources)}

    def _run_chunk(self, mix: np.ndarray, offset: int, chunk_len: int) -> np.ndarray:
        """Feed one training_length window to the ONNX graph and return the
        model's output cropped back down to `chunk_len` — same centering
        convention as DemucsONNXSeparator._run_chunk (see that module's
        docstring for why zero-padding a chunk boundary would be wrong)."""
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

        b, s = 1, self.audio_channels
        z = stft(mix_in.reshape(b * s, -1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, normalized=self.normalized)
        freqs, frames = z.shape[-2], z.shape[-1]
        z = z.reshape(b, s, freqs, frames)
        z_folded_complex = np.moveaxis(z, 1, 2).reshape(b, freqs * s, frames)  # (b, f*s, t)
        z_folded_real = np.stack([z_folded_complex.real, z_folded_complex.imag], axis=-1)

        core_input = gather_bands(z_folded_real, self.freq_indices)
        (masks,) = self.session.run(["masks"], {"core_input": core_input.astype(np.float32)})

        combined_masks = combine_masks(masks, self.freq_indices, self.num_bands_per_freq, num_freqs=freqs, num_channels=s)
        n = combined_masks.shape[1]
        modulated = z_folded_complex[:, None] * combined_masks  # (b, n, f*s, t)
        modulated = modulated.reshape(b * n, freqs, s, frames)  # undo the (f s) fold
        modulated = np.moveaxis(modulated, 2, 1).reshape(b * n * s, freqs, frames)  # -> (b*n*s, f, t)
        modulated = zero_dc(modulated)

        recon_flat = istft(
            modulated, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            length=self.training_length, normalized=self.normalized,
        )
        recon = recon_flat.reshape(n, s, self.training_length)  # b=1, dropped

        crop_start = delta // 2
        return recon[:, :, crop_start : crop_start + chunk_len]


def _crossfade_weight(segment_length: int) -> np.ndarray:
    """Triangle-shaped overlap-add weight, maximal at the segment center —
    same scheme as demucs_onnx.py's."""
    half = segment_length // 2
    ramp = np.concatenate([np.arange(1, half + 1), np.arange(segment_length - half, 0, -1)]).astype(np.float64)
    return ramp / ramp.max()
