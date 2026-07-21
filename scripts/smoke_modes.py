"""Phase 4 smoke test: run music/video/full modes on the same clip via the
real isolated-subprocess runner (no mocks), writing data/out_modes/{mode}/
{stem}.wav and printing per-stem RMS so you can eyeball — or listen to —
what each mode actually produces.

    uv run --group speech python scripts/smoke_modes.py
    uv run --group speech python scripts/smoke_modes.py --input /tmp/clip.wav --modes full
    uv run python scripts/smoke_modes.py --modes music   # no speech extras needed
    uv run python scripts/smoke_modes.py --modes music --stems 6   # htdemucs_6s

video/full modes need the `speech` extras (torch/torchaudio/pytorch-
lightning/spafe) and the downloaded Bandit checkpoint
(scripts/fetch_bandit_weights.py); music mode alone works from a plain
`uv run`. `--stems 6` needs models/htdemucs_6s_core*.onnx exported first
(scripts/export_onnx.py --model htdemucs_6s) and applies to the Demucs side
of music/full mode only — video mode (Bandit alone) has no stem-count
concept.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import DATA_DIR, MUSIC_SAMPLE_RATE
from backend.runner import separate_in_subprocess

DEFAULT_INPUT_PATH = DATA_DIR / "test.mp3"
OUTPUT_ROOT = DATA_DIR / "out_modes"
MODES = ("music", "video", "full")


def ffmpeg_decode(input_path: Path, output_wav: Path, duration: float | None, offset: float) -> None:
    cmd = ["ffmpeg", "-y", "-ss", str(offset), "-i", str(input_path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-ac", "2", "-ar", str(MUSIC_SAMPLE_RATE), str(output_wav)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


def run_mode(mode: str, wav_path: Path, tier: str, stems: int, timeout: float) -> None:
    output_dir = OUTPUT_ROOT / mode if stems == 4 else OUTPUT_ROOT / f"{mode}_{stems}s"
    print(f"\n=== mode={mode} tier={tier} stems={stems} ===")
    result = separate_in_subprocess(wav_path, output_dir, mode=mode, tier=tier, stem_count=stems, timeout=timeout)
    for name, path in sorted(result.items()):
        wav, _sr = sf.read(path, dtype="float32", always_2d=True)
        rms = float(np.sqrt(np.mean(wav**2)))
        print(f"  {name:8s} rms={rms:.6f}  ({path})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="audio file to decode (default: data/test.mp3)")
    parser.add_argument("--duration", type=float, default=30.0, help="seconds to use (None via --duration -1 for the whole file)")
    parser.add_argument("--offset", type=float, default=0.0, help="start offset in seconds")
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--stems", type=int, default=4, choices=(4, 6), help="Demucs stem count: 4 (default) or 6 (htdemucs_6s)")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found")
    duration = None if args.duration < 0 else args.duration

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "input.wav"
        window = f"[{args.offset}s..{'end' if duration is None else args.offset + duration}s]"
        print(f"decoding {args.input} {window} -> {wav_path} ({MUSIC_SAMPLE_RATE} Hz stereo)")
        ffmpeg_decode(args.input, wav_path, duration, args.offset)

        for mode in args.modes:
            run_mode(mode, wav_path, tier=args.tier, stems=args.stems, timeout=args.timeout)

    print(f"\nall stems written under {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
