"""Int8-quantize the exported htdemucs ONNX model: models/htdemucs_core_int8.onnx.

Dynamic quantization only touches weights (no calibration data needed) and is
the "fast" tier's model per config.TIERS. Run after scripts/export_onnx.py:

    uv run python scripts/quantize.py

No torch/demucs needed — this operates purely on the ONNX graph, so it runs
in the default (non-eval) environment, same as the product path itself.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import MODELS_DIR

INPUT_PATH = MODELS_DIR / "htdemucs_core.onnx"
OUTPUT_PATH = MODELS_DIR / "htdemucs_core_int8.onnx"


def quantize(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found — run scripts/export_onnx.py first")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    return output_path


if __name__ == "__main__":
    t0 = time.time()
    path = quantize()
    before_mb = INPUT_PATH.stat().st_size / (1024 * 1024)
    after_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} in {time.time() - t0:.1f}s")
    print(f"size: {before_mb:.1f} MB -> {after_mb:.1f} MB ({100 * after_mb / before_mb:.0f}%)")
