"""Export htdemucs to ONNX: models/htdemucs_core.onnx.

Only the STFT-split "core" network (backend/separators/_demucs_core.py) is
exportable — see that module's docstring for why the full HTDemucs.forward()
cannot be traced directly. Run from the eval group (needs torch+demucs to load
the pretrained weights; onnxruntime alone can't export, only run):

    uv run --group eval python scripts/export_onnx.py

Verifies the export against the PyTorch model on a random tensor before
writing, so a broken export never gets silently written to models/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import MODELS_DIR
from backend.separators._demucs_core import DemucsCore, cac_to_complex, ispec, magnitude_cac, spec

MODEL_NAME = "htdemucs"
OUTPUT_PATH = MODELS_DIR / "htdemucs_core.onnx"
METADATA_PATH = MODELS_DIR / "htdemucs_core.json"
VERIFY_TOLERANCE = 2e-3  # max-abs-diff vs the traced PyTorch model, fp32 kernel noise


def export(output_path: Path = OUTPUT_PATH, metadata_path: Path = METADATA_PATH) -> Path:
    from demucs.pretrained import get_model

    bag = get_model(MODEL_NAME)
    assert len(bag.models) == 1, "expected a single-model htdemucs, not a bag/ft ensemble"
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

    _verify(core, mag, mix, model, output_path)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
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


def _verify(core, mag, mix, model, onnx_path: Path) -> None:
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


if __name__ == "__main__":
    path = export()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.1f} MB)")
    print(f"wrote {METADATA_PATH}")
