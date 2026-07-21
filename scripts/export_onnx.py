"""Export htdemucs-family models to ONNX: models/{model_name}_core.onnx.

Only the STFT-split "core" network (backend/separators/_demucs_core.py) is
exportable — see that module's docstring for why the full HTDemucs.forward()
cannot be traced directly. That split only relies on cac=True/wiener_iters<=0,
which holds for every htdemucs variant, so htdemucs_6s (or any future variant)
goes through the exact same path as htdemucs — no new code, just a different
`--model` name. Run from the eval group (needs torch+demucs to load the
pretrained weights; onnxruntime alone can't export, only run):

    uv run --group eval python scripts/export_onnx.py
    uv run --group eval python scripts/export_onnx.py --model htdemucs_6s

Verification ladder, cheapest/most-certain check first — a model is only
written to models/ once all three pass:
  1. Pre-export, pure PyTorch, no ONNX involved: DemucsCore(model) + the
     spec/ispec reconstruction reproduces model(mix) exactly (0.0 max-abs-
     diff). DemucsCore holds the same submodules as `model` and runs the
     identical op sequence split into two calls instead of one forward(), so
     any nonzero diff here means the refactor itself is wrong — independent
     of anything ONNX-related.
  2. Post-export: the ONNX graph matches the (already-verified) PyTorch core
     to fp32 kernel noise.
  3. Full reconstruction (ONNX core + STFT/ISTFT) matches model(mix) to the
     same tolerance — the same check test_onnx_vs_oracle.py repeats against
     _oracle_torch.py at the higher (~1e-4) real-audio tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import MODELS_DIR
from backend.separators._demucs_core import DemucsCore, cac_to_complex, ispec, magnitude_cac, spec

DEFAULT_MODEL_NAME = "htdemucs"
VERIFY_TOLERANCE = 2e-3  # max-abs-diff vs the traced PyTorch model, fp32 kernel noise
PRE_EXPORT_TOLERANCE = 0.0  # pure-PyTorch refactor check: must be exact, no export involved


def model_paths(model_name: str) -> tuple[Path, Path]:
    return MODELS_DIR / f"{model_name}_core.onnx", MODELS_DIR / f"{model_name}_core.json"


def export(model_name: str = DEFAULT_MODEL_NAME, output_path: Path | None = None, metadata_path: Path | None = None) -> Path:
    from demucs.pretrained import get_model

    default_output, default_metadata = model_paths(model_name)
    output_path = output_path or default_output
    metadata_path = metadata_path or default_metadata

    bag = get_model(model_name)
    assert len(bag.models) == 1, f"expected a single-model {model_name}, not a bag/ft ensemble"
    model = bag.models[0]
    model.eval()
    assert model.cac and model.wiener_iters <= 0, "core split assumes complex-as-channels, not Wiener"

    training_length = int(model.segment * model.samplerate)
    torch.manual_seed(0)
    mix = torch.randn(1, model.audio_channels, training_length)
    z = spec(mix, model.nfft, model.hop_length)
    mag = magnitude_cac(z)

    core = DemucsCore(model)
    core.eval()

    _verify_pre_export(core, mag, mix, model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        core,
        (mag, mix),
        str(output_path),
        input_names=["mag", "mix"],
        output_names=["x_out", "xt_out"],
        opset_version=17,
        dynamo=False,
    )

    _verify_onnx(core, mag, mix, model, output_path)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "sources": model.sources,
                "nfft": model.nfft,
                "hop_length": model.hop_length,
                "samplerate": model.samplerate,
                "audio_channels": model.audio_channels,
                "training_length": training_length,
            },
            indent=2,
        )
    )
    return output_path


def _verify_pre_export(core: DemucsCore, mag: torch.Tensor, mix: torch.Tensor, model) -> None:
    with torch.no_grad():
        ref_full = model(mix)
        core_x, core_xt = core(mag, mix)
    zout = cac_to_complex(core_x)
    x_rec = ispec(zout, model.hop_length, mix.shape[-1])
    core_full = core_xt + x_rec
    diff = (ref_full - core_full).abs().max().item()
    print(f"verify: pre-export pure-pytorch core-vs-full max-abs-diff={diff:.6g}")
    if diff > PRE_EXPORT_TOLERANCE:
        raise RuntimeError(
            f"DemucsCore refactor diverges from model.forward() before any export: "
            f"max-abs-diff {diff:.6g} > {PRE_EXPORT_TOLERANCE}"
        )


def _verify_onnx(core, mag, mix, model, onnx_path: Path) -> None:
    with torch.no_grad():
        ref_x, ref_xt = core(mag, mix)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_x, onnx_xt = sess.run(["x_out", "xt_out"], {"mag": mag.numpy(), "mix": mix.numpy()})

    diff_x = (ref_x.numpy() - onnx_x).__abs__().max()
    diff_xt = (ref_xt.numpy() - onnx_xt).__abs__().max()
    max_diff = max(diff_x, diff_xt)
    print(f"verify: max-abs-diff x_out={diff_x:.6g} xt_out={diff_xt:.6g}")
    if max_diff > VERIFY_TOLERANCE:
        raise RuntimeError(f"ONNX export diverges from PyTorch core: max-abs-diff {max_diff:.6g} > {VERIFY_TOLERANCE}")

    # Also sanity-check the full reconstruction (core + STFT/ISTFT) end to end.
    with torch.no_grad():
        ref_full = model(mix)
    zout = cac_to_complex(torch.from_numpy(onnx_x))
    x_rec = ispec(zout, model.hop_length, mix.shape[-1])
    onnx_full = torch.from_numpy(onnx_xt) + x_rec
    full_diff = (ref_full - onnx_full).abs().max().item()
    print(f"verify: full-reconstruction max-abs-diff={full_diff:.6g}")
    if full_diff > VERIFY_TOLERANCE:
        raise RuntimeError(f"Full ONNX reconstruction diverges: max-abs-diff {full_diff:.6g} > {VERIFY_TOLERANCE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="demucs pretrained model name, e.g. htdemucs or htdemucs_6s")
    args = parser.parse_args()

    path = export(args.model)
    _, metadata_path = model_paths(args.model)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.1f} MB)")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
