"""ffmpeg normalize -> WAV 44.1k stereo (or 16k mono for the speech-only path).

Stub only — implemented in Phase 2.
"""

from __future__ import annotations

from pathlib import Path


def decode(path: Path, sample_rate: int, mono: bool = False) -> Path:
    """Run ffmpeg in a sandboxed subprocess with a timeout. Phase 2."""
    raise NotImplementedError
