"""yt-dlp wrapper: public link -> local audio file, downloaded into a fresh
temp dir. YouTube/TikTok/Instagram only (decision 10); URL validated against
an explicit host allowlist before yt-dlp ever runs; no cookies/auth are ever
wired up, so private/authenticated content simply can't be reached — if a
video needs a login, yt-dlp fails and that error is surfaced as-is, never
retried or scraped around (CLAUDE.md §6: "do not fetch private/authenticated
content ... surface yt-dlp errors").
"""

from __future__ import annotations

import ipaddress
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from backend.config import FETCH_TIMEOUT_SECONDS, MAX_DURATION_SECONDS, MAX_UPLOAD_BYTES
from backend.ingest._sandbox import run_subprocess

ALLOWED_SCHEMES = ("http", "https")
ALLOWED_HOST_SUFFIXES = ("youtube.com", "youtu.be", "tiktok.com", "instagram.com")


class InvalidURLError(ValueError):
    """URL failed scheme/host validation."""


class FetchError(RuntimeError):
    """yt-dlp failed, timed out, or the content violates a duration/size cap."""


def validate_url(url: str) -> str:
    """Returns the URL unchanged if it passes validation; raises InvalidURLError
    otherwise. Public http(s) URLs on an allowed platform only — no private/
    internal hosts (defends against SSRF-style abuse of the fetch step, and
    a private IP is never a legitimate YouTube/TikTok/Instagram URL anyway)."""
    url = (url or "").strip()
    if not url:
        raise InvalidURLError("empty URL")

    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise InvalidURLError(f"unsupported scheme {parsed.scheme!r}; only http/https allowed")

    host = (parsed.hostname or "").lower()
    if not host:
        raise InvalidURLError("URL has no host")
    if _is_private_host(host):
        raise InvalidURLError(f"host {host!r} is private/internal, not a public URL")
    if not any(host == suffix or host.endswith(f".{suffix}") for suffix in ALLOWED_HOST_SUFFIXES):
        raise InvalidURLError(f"host {host!r} is not an allowed platform (YouTube/TikTok/Instagram)")

    return url


def _is_private_host(host: str) -> bool:
    if host in ("localhost", "0.0.0.0"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # not an IP literal (a normal domain name) — fine
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified


def fetch(url: str, timeout: float = FETCH_TIMEOUT_SECONDS) -> Path:
    """Download audio-only from `url` into a fresh temp dir; return the path
    to the downloaded file. The caller owns cleanup of the returned path's
    parent dir (ingest() does this).

    Raises InvalidURLError (bad/private URL) or FetchError (yt-dlp failure,
    timeout, or a duration/size cap violation).
    """
    validated = validate_url(url)

    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp is None:
        raise FetchError("yt-dlp not found on PATH")

    tmp_dir = Path(tempfile.mkdtemp(prefix="stemmer-fetch-"))
    try:
        duration = _probe_duration(yt_dlp, validated, timeout)
        if duration is not None and duration > MAX_DURATION_SECONDS:
            raise FetchError(f"duration {duration:.0f}s exceeds cap of {MAX_DURATION_SECONDS}s")

        cmd = [
            yt_dlp,
            "--no-config",  # ignore any user/global yt-dlp config files
            "--no-playlist",
            "--no-cookies",
            "--no-cookies-from-browser",
            "--no-cache-dir",
            "--restrict-filenames",
            "--socket-timeout",
            str(int(timeout)),
            "--max-filesize",
            str(MAX_UPLOAD_BYTES),
            "--extract-audio",
            "--audio-format",
            "wav",
            "-o",
            "%(id)s.%(ext)s",
            validated,
        ]
        try:
            proc = run_subprocess(cmd, timeout=timeout, cwd=tmp_dir)
        except subprocess.TimeoutExpired as exc:
            raise FetchError(f"yt-dlp exceeded {timeout}s timeout") from exc

        if proc.returncode != 0:
            raise FetchError(f"yt-dlp failed (exit {proc.returncode}):\n{proc.stderr.strip()}")

        files = [p for p in tmp_dir.iterdir() if p.is_file()]
        if not files:
            raise FetchError("yt-dlp reported success but produced no output file")
        output_path = files[0]

        size = output_path.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            raise FetchError(f"downloaded file is {size} bytes, exceeds cap of {MAX_UPLOAD_BYTES} bytes")

        return output_path
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _probe_duration(yt_dlp: str, url: str, timeout: float) -> float | None:
    """Cheap metadata-only probe (no download) so an over-length video is
    rejected before we spend bandwidth/time downloading it."""
    cmd = [
        yt_dlp,
        "--no-config",
        "--no-playlist",
        "--no-cookies",
        "--no-cookies-from-browser",
        "--no-cache-dir",
        "--socket-timeout",
        str(int(timeout)),
        "--dump-json",
        "--skip-download",
        url,
    ]
    try:
        proc = run_subprocess(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FetchError(f"yt-dlp probe exceeded {timeout}s timeout") from exc

    if proc.returncode != 0:
        raise FetchError(f"yt-dlp failed (exit {proc.returncode}):\n{proc.stderr.strip()}")

    try:
        info = json.loads(proc.stdout.splitlines()[0])
    except (IndexError, json.JSONDecodeError) as exc:
        raise FetchError("yt-dlp probe returned no usable metadata") from exc
    return info.get("duration")
