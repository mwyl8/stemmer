"""Phase 1 end-to-end smoke test: decode data/test.mp3, separate it via the
isolated-subprocess runner (the ONNX product path), write data/out/{stem}.wav,
print per-stem RMS.

    uv run python scripts/smoke.py

No torch/demucs needed — this exercises exactly the product path a real
request would take (router -> DemucsONNXSeparator -> runner subprocess).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import DATA_DIR, MUSIC_SAMPLE_RATE
from backend.runner import separate_in_subprocess

INPUT_PATH = DATA_DIR / "test.mp3"
OUTPUT_DIR = DATA_DIR / "out"


def ffmpeg_decode(input_path: Path, output_wav: Path) -> None:
    """ffmpeg -> WAV, 44.1 kHz stereo. Sandboxed with a timeout; the full
    URL-validation/caps version of this lands in ingest/decode.py (Phase 2)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "2",
        "-ar",
        str(MUSIC_SAMPLE_RATE),
        str(output_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "input.wav"
        print(f"decoding {INPUT_PATH} -> {wav_path} ({MUSIC_SAMPLE_RATE} Hz stereo)...")
        ffmpeg_decode(INPUT_PATH, wav_path)

        print("separating via the isolated-subprocess runner (music/balanced, ONNX)...")
        stems = separate_in_subprocess(
            input_wav=wav_path,
            output_dir=OUTPUT_DIR,
            mode="music",
            tier="balanced",
            timeout=900,
        )

    print(f"\nwrote stems to {OUTPUT_DIR}:")
    for name, path in sorted(stems.items()):
        wav, _sr = sf.read(path, dtype="float32", always_2d=True)
        rms = float(np.sqrt(np.mean(wav**2)))
        print(f"  {name:8s} rms={rms:.6f}  ({path})")


if __name__ == "__main__":
    main()
