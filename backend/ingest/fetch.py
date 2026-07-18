"""yt-dlp wrapper: link -> local audio file. URL validation, size/duration caps,
sandboxed subprocess, timeout. Public URLs only — no private/authenticated content.

Stub only — implemented in Phase 2.
"""

from __future__ import annotations

from pathlib import Path


def fetch(url: str) -> Path:
    """Validate url, run yt-dlp (YouTube/TikTok/Instagram) with a timeout. Phase 2."""
    raise NotImplementedError


def validate_url(url: str) -> bool:
    """Reject anything but public http(s) URLs from supported hosts. Phase 2."""
    raise NotImplementedError
