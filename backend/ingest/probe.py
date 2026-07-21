"""ffprobe wrapper: reads codec/bitrate/sample-rate/channel-count off the
*original* source file (before decode.py normalizes it to 44.1kHz stereo
WAV), so the job record can always show what the pipeline actually started
from — a low-bitrate YouTube rip looks very different from a lossless
upload, and that's otherwise invisible once everything's been normalized to
the same WAV shape.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from backend.config import DECODE_TIMEOUT_SECONDS
from backend.ingest._sandbox import run_subprocess


def probe_source_info(path: Path, timeout: float = DECODE_TIMEOUT_SECONDS) -> dict:
    """Returns {"codec": str|None, "bitrate": int|None, "sample_rate": int|None,
    "channels": int|None} for the first audio stream in `path`.

    Never raises on a malformed/unreadable file — a probe failure shouldn't
    fail the whole job over metadata that's purely informational, so every
    field just comes back None instead.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return _empty()

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-select_streams",
        "a:0",
        str(path),
    ]
    try:
        proc = run_subprocess(cmd, timeout=timeout)
    except Exception:
        return _empty()

    if proc.returncode != 0:
        return _empty()

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _empty()

    streams = info.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = info.get("format") or {}

    bitrate = _to_int(stream.get("bit_rate")) or _to_int(fmt.get("bit_rate"))
    return {
        "codec": stream.get("codec_name"),
        "bitrate": bitrate,
        "sample_rate": _to_int(stream.get("sample_rate")),
        "channels": _to_int(stream.get("channels")),
    }


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty() -> dict:
    return {"codec": None, "bitrate": None, "sample_rate": None, "channels": None}
