"""ffmpeg wrapper: any local audio/video file -> canonical WAV.

Two variants: `decode_music` (44.1kHz stereo — the only sample rate the
music/Demucs path may use) and `decode_speech` (16kHz mono, for the future
Bandit speech-only path). `decode()` itself refuses to downsample below
44.1kHz unless `mono=True`, so a caller can't accidentally send the music
path through the cheap speech-only rate (CLAUDE.md §6).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.config import DECODE_TIMEOUT_SECONDS, MUSIC_SAMPLE_RATE, SPEECH_SAMPLE_RATE
from backend.ingest._sandbox import run_subprocess


class DecodeError(RuntimeError):
    """ffmpeg failed, timed out, or the input file couldn't be normalized."""


def decode(
    path: Path,
    sample_rate: int = MUSIC_SAMPLE_RATE,
    mono: bool = False,
    timeout: float = DECODE_TIMEOUT_SECONDS,
) -> Path:
    """Normalize `path` (anything ffmpeg can read: mp3/mp4/wav/...) to a WAV
    at `sample_rate`, stereo unless `mono=True`. Returns the path to a new
    temp WAV file; the caller owns cleanup of its parent dir.
    """
    if sample_rate < MUSIC_SAMPLE_RATE and not mono:
        raise ValueError("only the mono speech path may go below 44.1kHz — the music path is locked at 44.1kHz stereo")

    path = Path(path)
    if not path.exists():
        raise DecodeError(f"input file not found: {path}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise DecodeError("ffmpeg not found on PATH")

    tmp_dir = Path(tempfile.mkdtemp(prefix="stemmer-decode-"))
    output_path = tmp_dir / "normalized.wav"
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        str(path),
        "-vn",  # drop any video stream — audio only
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(output_path),
    ]
    try:
        proc = run_subprocess(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise DecodeError(f"ffmpeg exceeded {timeout}s timeout") from exc

    if proc.returncode != 0 or not output_path.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise DecodeError(f"ffmpeg failed (exit {proc.returncode}):\n{proc.stderr.strip()}")

    return output_path


def decode_music(path: Path, timeout: float = DECODE_TIMEOUT_SECONDS) -> Path:
    return decode(path, sample_rate=MUSIC_SAMPLE_RATE, mono=False, timeout=timeout)


def decode_speech(path: Path, timeout: float = DECODE_TIMEOUT_SECONDS) -> Path:
    return decode(path, sample_rate=SPEECH_SAMPLE_RATE, mono=True, timeout=timeout)
