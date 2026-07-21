"""Int8-quantize an exported htdemucs-family ONNX model: models/{model_name}_core_int8.onnx.

Dynamic quantization only touches weights (no calibration data needed) and is
the "fast" tier's model per config.TIERS. Run after scripts/export_onnx.py:

    uv run python scripts/quantize.py
    uv run python scripts/quantize.py --model htdemucs_6s

No torch/demucs needed — this operates purely on the ONNX graph, so it runs
in the default (non-eval) environment, same as the product path itself.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import MODELS_DIR

DEFAULT_MODEL_NAME = "htdemucs"


def default_paths(model_name: str) -> tuple[Path, Path]:
    return MODELS_DIR / f"{model_name}_core.onnx", MODELS_DIR / f"{model_name}_core_int8.onnx"


def quantize(input_path: Path, output_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found — run scripts/export_onnx.py first")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="demucs pretrained model name, e.g. htdemucs or htdemucs_6s")
    args = parser.parse_args()

    input_path, output_path = default_paths(args.model)
    t0 = time.time()
    path = quantize(input_path, output_path)
    before_mb = input_path.stat().st_size / (1024 * 1024)
    after_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} in {time.time() - t0:.1f}s")
    print(f"size: {before_mb:.1f} MB -> {after_mb:.1f} MB ({100 * after_mb / before_mb:.0f}%)")
