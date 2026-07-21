"""Single entrypoint: {file | url} -> normalized WAV ready for the separator.

Routes a local file straight to decode(); a URL through fetch() first, then
decode(). Also computes the content hash of the decoded audio here, so
Phase 3's cache can key on it without re-reading the file.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from backend.config import MAX_DURATION_SECONDS, MAX_UPLOAD_BYTES, MUSIC_SAMPLE_RATE
from backend.ingest import decode as decode_mod
from backend.ingest import fetch as fetch_mod
from backend.ingest.probe import probe_source_info

_HASH_CHUNK_BYTES = 1024 * 1024


class IngestError(RuntimeError):
    """Input is missing, or the normalized audio violates a duration/size cap."""


@dataclass(frozen=True)
class IngestResult:
    wav_path: Path
    content_hash: str
    # What the pipeline actually started from, read off the source file
    # before decode() normalizes it away (PRD Addendum §2.5 job metadata
    # panel) — None fields if ffprobe couldn't read it.
    source_codec: str | None
    source_bitrate: int | None
    source_sample_rate: int | None
    source_channels: int | None


def ingest(file: Path | str | None = None, url: str | None = None, sample_rate: int = MUSIC_SAMPLE_RATE) -> IngestResult:
    """Exactly one of `file`/`url` must be given.

    file: local path, passed straight to decode() (ffmpeg normalize only).
    url:  fetched via yt-dlp first (fetch.py), then decoded.

    Returns the normalized WAV's path and its content hash (sha256). The
    caller owns cleanup of wav_path's parent dir once done with it.
    """
    if (file is None) == (url is None):
        raise ValueError("pass exactly one of file= or url=")

    fetch_dir: Path | None = None
    try:
        if url is not None:
            source_path = fetch_mod.fetch(url)
            fetch_dir = source_path.parent
        else:
            source_path = Path(file)
            if not source_path.exists():
                raise IngestError(f"no such file: {source_path}")
            if source_path.stat().st_size > MAX_UPLOAD_BYTES:
                raise IngestError(f"file is {source_path.stat().st_size} bytes, exceeds cap of {MAX_UPLOAD_BYTES} bytes")

        source_info = probe_source_info(source_path)
        wav_path = decode_mod.decode(source_path, sample_rate=sample_rate, mono=False)
    finally:
        if fetch_dir is not None:
            shutil.rmtree(fetch_dir, ignore_errors=True)

    duration = sf.info(wav_path).duration
    if duration > MAX_DURATION_SECONDS:
        shutil.rmtree(wav_path.parent, ignore_errors=True)
        raise IngestError(f"duration {duration:.0f}s exceeds cap of {MAX_DURATION_SECONDS}s")

    return IngestResult(
        wav_path=wav_path,
        content_hash=_hash_file(wav_path),
        source_codec=source_info["codec"],
        source_bitrate=source_info["bitrate"],
        source_sample_rate=source_info["sample_rate"],
        source_channels=source_info["channels"],
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
