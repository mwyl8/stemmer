"""Isolated-subprocess entrypoint: runs one separation, then exits.

Two halves:
- `separate_in_subprocess()` — parent-process API. Spawns `python -m
  backend.runner` as a child, waits up to `timeout` seconds, kills it on
  timeout (`subprocess.run`'s own `timeout` handling). This is what `pool.py`
  (Phase 3) will call per job; `scripts/smoke.py` calls it directly for now
  since there's no job queue yet.
- `_main()` — child-process entrypoint. Loads the routed Separator exactly
  once (via `router.select_separator`, which loads the ONNX Runtime session)
  and reuses it for every chunk of the one file it separates, then exits —
  the process boundary gives clean memory reclamation and a hard kill point,
  without needing a long-lived worker (that's Phase 3's `pool.py`).

Deliberately does not import onnxruntime/numpy-heavy separator code at
parent-process import time — only the child process (a fresh Python
interpreter) pays that cost, keeping the parent light.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


class SeparationTimeout(RuntimeError):
    """Raised when the child subprocess is killed for exceeding its timeout."""


class SeparationFailed(RuntimeError):
    """Raised when the child subprocess exits non-zero."""


def separate_in_subprocess(
    input_wav: Path,
    output_dir: Path,
    mode: str = "music",
    tier: str = "balanced",
    timeout: float = 300.0,
) -> dict[str, Path]:
    """Run one separation of `input_wav` in an isolated subprocess.

    Returns {stem_name: wav_path} for the stems written under `output_dir`.
    Raises SeparationTimeout if the child doesn't finish in time (it is
    killed), or SeparationFailed if it exits non-zero.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "backend.runner",
        "--input",
        str(input_wav),
        "--output-dir",
        str(output_dir),
        "--mode",
        mode,
        "--tier",
        tier,
    ]
    try:
        proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        raise SeparationTimeout(f"separation exceeded {timeout}s timeout") from exc

    if proc.returncode != 0:
        raise SeparationFailed(f"runner subprocess failed (exit {proc.returncode}):\n{proc.stderr}")

    manifest = json.loads((output_dir / "stems.json").read_text())
    return {name: output_dir / f"{name}.wav" for name in manifest}


def _main() -> None:
    import soundfile as sf

    from backend.config import MUSIC_SAMPLE_RATE
    from backend.separators.router import select_separator

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default="music")
    parser.add_argument("--tier", default="balanced")
    args = parser.parse_args()

    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    if sr != MUSIC_SAMPLE_RATE:
        raise SeparationFailed(f"expected {MUSIC_SAMPLE_RATE} Hz input, got {sr}")
    audio = audio.T  # (samples, channels) -> (channels, samples)

    separator = select_separator(args.mode, args.tier)  # loads the model once
    stems = separator.separate(audio)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name, wav in stems.items():
        sf.write(output_dir / f"{name}.wav", wav.T, MUSIC_SAMPLE_RATE)
        names.append(name)
    (output_dir / "stems.json").write_text(json.dumps(names))


if __name__ == "__main__":
    _main()
