"""Measures real-time factor (separation seconds per second of audio) for
each wired tier on *this* host's detected architecture, via the real
isolated-subprocess runner (no mocks) — the source of truth
config.TIER_RTF / config.ARCH_RUNTIME_PROFILES are meant to be tuned from,
not guessed. Run it fresh on any new/changed hardware (a new CPU, a new
onnxruntime build with a different provider set) before trusting those
constants there.

    uv run python scripts/bench_arch.py
    uv run python scripts/bench_arch.py --tiers fast balanced --repeats 3
    uv run python scripts/bench_arch.py --input /tmp/clip.wav --duration 20

Prints one row per tier: host arch bucket, resolved provider, RTF (median of
`--repeats` runs), and how that compares to today's config.TIER_RTF value —
a large gap means the constant is stale for this host and should be updated.
Also writes the raw numbers to --out (JSON) so results from several hosts
(x86_64 VNNI box, Apple Silicon laptop, a Graviton instance, ...) can be
diffed side by side.

This never changes config.py itself — updating ARCH_RUNTIME_PROFILES/
TIER_RTF from what this prints is a deliberate, reviewed edit, same as any
other measured-constant update in this codebase (see TIER_RTF's own
docstring for that norm).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.arch import detect_host_arch, profile_for, resolve_providers
from backend.config import DATA_DIR, MUSIC_SAMPLE_RATE, TIER_RTF, TIERS
from backend.runner import SeparationFailed, SeparationTimeout, separate_in_subprocess

DEFAULT_INPUT_PATH = DATA_DIR / "test.mp3"


def ffmpeg_decode(input_path: Path, output_wav: Path, duration: float) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-t", str(duration), "-ac", "2", "-ar", str(MUSIC_SAMPLE_RATE), str(output_wav)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


def bench_tier(wav_path: Path, tier: str, audio_duration: float, repeats: int, timeout: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        elapsed = []
        for _ in range(repeats):
            output_dir = Path(tmp) / f"{tier}-{len(elapsed)}"
            start = time.monotonic()
            separate_in_subprocess(wav_path, output_dir, mode="music", tier=tier, timeout=timeout)
            elapsed.append(time.monotonic() - start)
    rtf_samples = [e / audio_duration for e in elapsed]
    return {"elapsed_seconds": elapsed, "rtf_samples": rtf_samples, "rtf_median": statistics.median(rtf_samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="audio file to decode (default: data/test.mp3)")
    parser.add_argument("--duration", type=float, default=20.0, help="seconds of audio to measure with")
    parser.add_argument("--tiers", nargs="+", default=[t for t in TIERS if t != "best"], choices=list(TIERS))
    parser.add_argument("--repeats", type=int, default=3, help="runs per tier — RTF reported is the median")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None, help="also write raw results as JSON to this path")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found")

    host = detect_host_arch()
    profile = profile_for(host)
    providers = resolve_providers(host)
    print(f"host arch: {host}")
    print(f"profile_key: {host.profile_key}  (config default tier here: {profile.tier!r})")
    print(f"resolved providers: {providers}\n")

    results = {"host": vars(host), "profile_key": host.profile_key, "providers": [str(p) for p in providers], "tiers": {}}

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "bench_input.wav"
        print(f"decoding {args.duration}s from {args.input} -> {wav_path}")
        ffmpeg_decode(args.input, wav_path, args.duration)

        header = f"{'tier':<10} {'RTF (median)':<14} {'config.TIER_RTF':<16} {'delta':<10}"
        print(header)
        print("-" * len(header))
        for tier in args.tiers:
            try:
                bench = bench_tier(wav_path, tier, args.duration, args.repeats, args.timeout)
            except (SeparationFailed, SeparationTimeout) as exc:
                print(f"{tier:<10} FAILED: {exc}")
                results["tiers"][tier] = {"error": str(exc)}
                continue
            results["tiers"][tier] = bench
            configured = TIER_RTF.get(tier)
            delta = f"{bench['rtf_median'] - configured:+.3f}" if configured is not None else "n/a"
            configured_str = f"{configured:.3f}" if configured is not None else "n/a"
            print(f"{tier:<10} {bench['rtf_median']:<14.3f} {configured_str:<16} {delta:<10}")

    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nraw results written to {args.out}")

    print(
        "\nIf a tier's measured RTF here is far from config.TIER_RTF (or the "
        "cheapest tier isn't the one config.ARCH_RUNTIME_PROFILES defaults "
        f"this arch bucket ({host.profile_key!r}) to), update those constants "
        "to match — they're deliberately measured-not-guessed (see their own "
        "docstrings for the precedent)."
    )


if __name__ == "__main__":
    main()
