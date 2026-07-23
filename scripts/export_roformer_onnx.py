"""Export the MelBandRoformer karaoke model (Lead vs. Backing Vocals mode)
to ONNX: models/karaoke_core.onnx.

Only the STFT-split "core" network (backend/separators/_roformer_core.py) is
exportable — see that module's docstring for why. Run from the eval group
(needs torch + the vendored model code to load pretrained weights;
onnxruntime alone can't export, only run):

    uv run --group eval python scripts/export_roformer_onnx.py

Verification ladder, same shape as scripts/export_onnx.py's:
  1. Pre-export, pure PyTorch, no ONNX involved: MelBandRoformerCore(model) +
     the stft_pre/mask_post reconstruction reproduces model(mix) exactly
     (0.0 max-abs-diff). Any nonzero diff here means the Core split itself
     is wrong, independent of anything ONNX-related.
  2. Post-export: the ONNX graph matches the (already-verified) PyTorch core
     to fp32 kernel noise.
  3. Full reconstruction (ONNX core + numpy STFT/ISTFT, the actual product
     path) matches model(mix) to the same tolerance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import MODELS_DIR
from backend.separators._roformer_core import MelBandRoformerCore, mask_post, stft_pre
from backend.separators._roformer_stft_numpy import combine_masks, gather_bands, istft, mel_band_indices, stft, zero_dc
from backend.separators._roformer_vendor.mel_band_roformer import MelBandRoformer

DEFAULT_REPO_ID = "becruily/mel-band-roformer-karaoke"
DEFAULT_CKPT_FILENAME = "mel_band_roformer_karaoke_becruily.ckpt"
DEFAULT_CONFIG_FILENAME = "config_karaoke_becruily.yaml"

OUTPUT_PATH = MODELS_DIR / "karaoke_core.onnx"
METADATA_PATH = MODELS_DIR / "karaoke_core.json"

PRE_EXPORT_TOLERANCE = 0.0
VERIFY_TOLERANCE = 2e-3
FULL_RECON_TOLERANCE = 1e-3  # the gate the task asked for: ~1e-3 or better


def _load_model_and_config(repo_id: str, ckpt_filename: str, config_filename: str):
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_filename)
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)

    cfg = yaml.unsafe_load(Path(config_path).read_text())["model"]
    model = MelBandRoformer(
        dim=cfg["dim"],
        depth=cfg["depth"],
        stereo=cfg["stereo"],
        num_stems=cfg["num_stems"],
        time_transformer_depth=cfg["time_transformer_depth"],
        freq_transformer_depth=cfg["freq_transformer_depth"],
        num_bands=cfg["num_bands"],
        dim_head=cfg["dim_head"],
        heads=cfg["heads"],
        attn_dropout=cfg["attn_dropout"],
        ff_dropout=cfg["ff_dropout"],
        flash_attn=cfg["flash_attn"],
        dim_freqs_in=cfg["dim_freqs_in"],
        sample_rate=cfg["sample_rate"],
        stft_n_fft=cfg["stft_n_fft"],
        stft_hop_length=cfg["stft_hop_length"],
        stft_win_length=cfg["stft_win_length"],
        stft_normalized=cfg["stft_normalized"],
        mask_estimator_depth=cfg["mask_estimator_depth"],
        multi_stft_resolution_loss_weight=cfg["multi_stft_resolution_loss_weight"],
        multi_stft_resolutions_window_sizes=tuple(cfg["multi_stft_resolutions_window_sizes"]),
        multi_stft_hop_size=cfg["multi_stft_hop_size"],
        multi_stft_normalized=cfg["multi_stft_normalized"],
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg


def export(
    repo_id: str = DEFAULT_REPO_ID,
    ckpt_filename: str = DEFAULT_CKPT_FILENAME,
    config_filename: str = DEFAULT_CONFIG_FILENAME,
    output_path: Path = OUTPUT_PATH,
    metadata_path: Path = METADATA_PATH,
    clip_seconds: float = 6.0,
) -> Path:
    model, cfg = _load_model_and_config(repo_id, ckpt_filename, config_filename)

    torch.manual_seed(0)
    sample_rate = cfg["sample_rate"]
    audio_length = int(clip_seconds * sample_rate)
    mix = torch.randn(1, 2, audio_length)

    core = MelBandRoformerCore(model)
    core.eval()

    core_input, stft_repr = stft_pre(mix, model)

    _verify_pre_export(core, core_input, stft_repr, mix, model, audio_length)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        core,
        (core_input,),
        str(output_path),
        input_names=["core_input"],
        output_names=["masks"],
        opset_version=17,
        dynamo=False,
    )

    _verify_onnx(core, core_input, model, mix, audio_length, output_path, cfg)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "ckpt_filename": ckpt_filename,
                "sources": ["lead_vocal", "backing_vocals"],  # this checkpoint's own labels: Vocals, Instrumental
                "sample_rate": sample_rate,
                "n_fft": cfg["stft_n_fft"],
                "hop_length": cfg["stft_hop_length"],
                "win_length": cfg["stft_win_length"],
                "normalized": cfg["stft_normalized"],
                "num_bands": cfg["num_bands"],
                "audio_channels": 2,
                "training_length": audio_length,  # fixed ONNX input length in samples — the exported graph has no dynamic axes
            },
            indent=2,
        )
    )
    return output_path


def _verify_pre_export(core, core_input, stft_repr, mix, model, audio_length: int) -> None:
    with torch.no_grad():
        ref_full = model(mix)
        masks = core(core_input)
        core_full = mask_post(masks, stft_repr, model, audio_length)
    diff = (ref_full - core_full).abs().max().item()
    print(f"verify: pre-export pure-pytorch core-vs-full max-abs-diff={diff:.6g}")
    if diff > PRE_EXPORT_TOLERANCE:
        raise RuntimeError(f"MelBandRoformerCore refactor diverges from model.forward(): max-abs-diff {diff:.6g}")


def _verify_onnx(core, core_input, model, mix, audio_length: int, onnx_path: Path, cfg: dict) -> None:
    """Exercises every function in _roformer_stft_numpy.py (the real product
    path — stft, mel_band_indices, gather_bands, combine_masks, istft)
    against the PyTorch oracle, not just the ONNX graph in isolation."""
    n_fft, hop_length, win_length = cfg["stft_n_fft"], cfg["stft_hop_length"], cfg["stft_win_length"]
    mix_np = mix.numpy()
    b, s, _ = mix_np.shape

    z = stft(mix_np.reshape(b * s, -1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, normalized=cfg["stft_normalized"])
    freqs, frames = z.shape[-2], z.shape[-1]
    z = z.reshape(b, s, freqs, frames)
    z_folded_complex = np.moveaxis(z, 1, 2).reshape(b, freqs * s, frames)  # (b, f*s, t) — f-major/s-minor, matches model's (f s) fold
    z_folded_real = np.stack([z_folded_complex.real, z_folded_complex.imag], axis=-1)  # (b, f*s, t, 2)

    freq_indices, num_bands_per_freq = mel_band_indices(cfg["sample_rate"], n_fft, cfg["num_bands"], stereo=True)
    core_input_numpy = gather_bands(z_folded_real, freq_indices)  # (b, t, f_gathered*2)

    # Numpy STFT + mel-band gather vs. the torch reference input (stft_pre) —
    # verifies _roformer_stft_numpy's input side independently of the ONNX
    # graph itself.
    input_diff = np.abs(core_input_numpy - core_input.numpy()).max()
    print(f"verify: numpy stft+gather_bands vs torch reference input max-abs-diff={input_diff:.6g}")
    if input_diff > VERIFY_TOLERANCE:
        raise RuntimeError(f"_roformer_stft_numpy input diverges from torch reference: max-abs-diff {input_diff:.6g}")

    with torch.no_grad():
        ref_masks = core(core_input)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_masks,) = sess.run(["masks"], {"core_input": core_input_numpy.astype(np.float32)})

    diff = np.abs(ref_masks.numpy() - onnx_masks).max()
    print(f"verify: max-abs-diff masks={diff:.6g}")
    if diff > VERIFY_TOLERANCE:
        raise RuntimeError(f"ONNX export diverges from PyTorch core: max-abs-diff {diff:.6g} > {VERIFY_TOLERANCE}")

    # Full reconstruction via the actual numpy product path end to end (no
    # torch past this point) — the real gate the task asked for.
    with torch.no_grad():
        ref_full = model(mix).numpy()

    combined_masks = combine_masks(onnx_masks, freq_indices, num_bands_per_freq, num_freqs=freqs, num_channels=s)  # (b, n, f*s, t)

    n = combined_masks.shape[1]
    modulated = z_folded_complex[:, None] * combined_masks  # (b, n, f*s, t)
    modulated = modulated.reshape(b * n, freqs, s, frames)  # undo the (f s) fold
    modulated = np.moveaxis(modulated, 2, 1).reshape(b * n * s, freqs, frames)  # -> (b*n*s, f, t) for per-channel istft
    modulated = zero_dc(modulated)

    recon_flat = istft(modulated, n_fft=n_fft, hop_length=hop_length, win_length=win_length, length=audio_length, normalized=cfg["stft_normalized"])
    recon = recon_flat.reshape(b, n, s, audio_length)

    full_diff = np.abs(ref_full - recon).max()
    print(f"verify: full-reconstruction (numpy product path) max-abs-diff={full_diff:.6g}")
    if full_diff > FULL_RECON_TOLERANCE:
        raise RuntimeError(f"Full numpy reconstruction diverges: max-abs-diff {full_diff:.6g} > {FULL_RECON_TOLERANCE}")


def main() -> None:
    path = export()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.1f} MB)")
    print(f"wrote {METADATA_PATH}")


if __name__ == "__main__":
    main()
