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

Progress (PRD §4, Phase 6): when `--job-id` is given, `_main()` wires the
routed Separator's `on_chunk` callback straight to `jobs.update_progress()`
— a direct SQLite write from this child process into the same jobs table
`pool.py`'s parent process reads, no pipe back to the parent needed. Chunk
counts for a job are small (tens, not thousands — see config.TIER_RTF /
segment sizing), so a write per finished chunk never hammers SQLite; that's
simpler than plumbing a pipe/queue back through subprocess.run's
capture_output, which only ever sees the child's stdout/stderr once it's
already exited.
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
    stem_count: int = 4,
    timeout: float = 300.0,
    job_id: str | None = None,
) -> dict[str, Path]:
    """Run one separation of `input_wav` in an isolated subprocess.

    Returns {stem_name: wav_path} for the stems written under `output_dir`.
    Raises SeparationTimeout if the child doesn't finish in time (it is
    killed), or SeparationFailed if it exits non-zero. If `job_id` is given,
    the child reports live chunk progress onto that job's row as it works
    (see module docstring).
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
        "--stem-count",
        str(stem_count),
    ]
    if job_id is not None:
        cmd += ["--job-id", job_id]
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
    parser.add_argument("--stem-count", type=int, default=4)
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args()

    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    if sr != MUSIC_SAMPLE_RATE:
        raise SeparationFailed(f"expected {MUSIC_SAMPLE_RATE} Hz input, got {sr}")
    audio = audio.T  # (samples, channels) -> (channels, samples)

    on_chunk = None
    if args.job_id is not None:
        from backend import jobs

        def on_chunk(done: int, total: int) -> None:
            jobs.update_progress(args.job_id, chunks_done=done, chunks_total=total)

    separator = select_separator(args.mode, args.tier, stem_count=args.stem_count)  # loads the model once
    stems = separator.separate(audio, on_chunk=on_chunk)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name, wav in stems.items():
        sf.write(output_dir / f"{name}.wav", wav.T, MUSIC_SAMPLE_RATE)
        names.append(name)
    (output_dir / "stems.json").write_text(json.dumps(names))


if __name__ == "__main__":
    _main()
