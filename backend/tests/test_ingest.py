"""URL validation (good/bad/private), caps enforcement, and the ffmpeg
normalize step. No network: yt-dlp is always mocked via monkeypatching
`run_subprocess`. ffmpeg itself runs for real on a tiny generated fixture —
it's a fast local binary, not a network dependency.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import backend.ingest as ingest_pkg
from backend.ingest import decode, fetch

# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

GOOD_URLS = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "http://www.tiktok.com/@user/video/123",
    "https://vm.tiktok.com/abc123",
    "https://www.instagram.com/reel/abc123",
    "https://instagram.com/p/abc123",
]

BAD_URLS = [
    "",
    "   ",
    "ftp://youtube.com/watch?v=x",
    "javascript:alert(1)",
    "https:///no-host",
    "https://example.com/video",
    "https://evil.com/youtube.com",
    "https://youtube.com.evil.com/watch?v=x",
    "https://notyoutube.com/watch?v=x",
]

PRIVATE_URLS = [
    "http://localhost/video",
    "http://127.0.0.1/video",
    "http://0.0.0.0/video",
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata SSRF target
    "http://10.0.0.5/video",
    "http://192.168.1.1/video",
    "http://[::1]/video",
]


@pytest.mark.parametrize("url", GOOD_URLS)
def test_validate_url_accepts_known_platforms(url):
    assert fetch.validate_url(url) == url


@pytest.mark.parametrize("url", BAD_URLS)
def test_validate_url_rejects_bad_urls(url):
    with pytest.raises(fetch.InvalidURLError):
        fetch.validate_url(url)


@pytest.mark.parametrize("url", PRIVATE_URLS)
def test_validate_url_rejects_private_hosts(url):
    with pytest.raises(fetch.InvalidURLError):
        fetch.validate_url(url)


# ---------------------------------------------------------------------------
# fetch() — yt-dlp always mocked, no network
# ---------------------------------------------------------------------------


def _fake_run_subprocess(probe_duration=10.0, download_size=1024, returncode=0, stderr=""):
    """Builds a fake run_subprocess: --dump-json calls return canned probe
    metadata; anything else simulates a download by writing a file into cwd."""
    calls = []

    def _fake(cmd, timeout, cwd=None):
        calls.append(cmd)
        if returncode != 0:
            return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)
        if "--dump-json" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"duration": probe_duration}) + "\n", stderr="")
        (Path(cwd) / "abc123.wav").write_bytes(b"\x00" * download_size)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    _fake.calls = calls
    return _fake


@pytest.fixture(autouse=True)
def _fake_yt_dlp_on_path(monkeypatch):
    # shutil is a single shared module object, so patching fetch.shutil.which
    # directly would also break decode.py's real (unmocked) ffmpeg lookup —
    # fall through to the real shutil.which for everything but "yt-dlp".
    real_which = shutil.which
    monkeypatch.setattr(fetch.shutil, "which", lambda name: "/usr/bin/yt-dlp" if name == "yt-dlp" else real_which(name))


def test_fetch_rejects_invalid_url_without_running_subprocess(monkeypatch):
    fake = _fake_run_subprocess()
    monkeypatch.setattr(fetch, "run_subprocess", fake)
    with pytest.raises(fetch.InvalidURLError):
        fetch.fetch("https://evil.com/video")
    assert fake.calls == []


def test_fetch_enforces_duration_cap_before_downloading(monkeypatch):
    monkeypatch.setattr(fetch, "MAX_DURATION_SECONDS", 60)
    fake = _fake_run_subprocess(probe_duration=999)
    monkeypatch.setattr(fetch, "run_subprocess", fake)
    with pytest.raises(fetch.FetchError, match="duration"):
        fetch.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert len(fake.calls) == 1  # only the probe ran, never the download


def test_fetch_enforces_size_cap_after_downloading(monkeypatch):
    monkeypatch.setattr(fetch, "MAX_UPLOAD_BYTES", 100)
    fake = _fake_run_subprocess(probe_duration=10, download_size=200)
    monkeypatch.setattr(fetch, "run_subprocess", fake)
    with pytest.raises(fetch.FetchError, match="exceeds cap"):
        fetch.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_fetch_surfaces_yt_dlp_error_verbatim(monkeypatch):
    fake = _fake_run_subprocess(returncode=1, stderr="ERROR: Private video. Sign in if you've been granted access")
    monkeypatch.setattr(fetch, "run_subprocess", fake)
    with pytest.raises(fetch.FetchError, match="Private video"):
        fetch.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_fetch_wraps_timeout(monkeypatch):
    def _timeout(cmd, timeout, cwd=None):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(fetch, "run_subprocess", _timeout)
    with pytest.raises(fetch.FetchError, match="timeout"):
        fetch.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_fetch_happy_path(monkeypatch):
    fake = _fake_run_subprocess(probe_duration=10, download_size=1024)
    monkeypatch.setattr(fetch, "run_subprocess", fake)
    path = fetch.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert path.exists()
    assert path.stat().st_size == 1024
    assert len(fake.calls) == 2  # probe + download


# ---------------------------------------------------------------------------
# decode() — real ffmpeg on a tiny generated fixture (local, fast, no network)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_fixture(tmp_path) -> Path:
    """1 second of a 22050Hz mono sine wave — deliberately NOT 44.1k/stereo,
    so a passing decode test proves normalization actually happened."""
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "fixture.wav"
    sf.write(path, tone, sr)
    return path


def test_decode_music_normalizes_to_44100_stereo(tiny_fixture):
    out = decode.decode_music(tiny_fixture)
    info = sf.info(out)
    assert info.samplerate == 44100
    assert info.channels == 2


def test_decode_speech_normalizes_to_16000_mono(tiny_fixture):
    out = decode.decode_speech(tiny_fixture)
    info = sf.info(out)
    assert info.samplerate == 16000
    assert info.channels == 1


def test_decode_rejects_low_rate_without_mono(tiny_fixture):
    with pytest.raises(ValueError):
        decode.decode(tiny_fixture, sample_rate=16000, mono=False)


def test_decode_missing_input_raises():
    with pytest.raises(decode.DecodeError):
        decode.decode(Path("/no/such/file.wav"))


def test_decode_wraps_timeout(tiny_fixture, monkeypatch):
    def _timeout(cmd, timeout, cwd=None):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(decode, "run_subprocess", _timeout)
    with pytest.raises(decode.DecodeError, match="timeout"):
        decode.decode_music(tiny_fixture)


def test_decode_surfaces_ffmpeg_failure(tiny_fixture, monkeypatch):
    def _fail(cmd, timeout, cwd=None):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ffmpeg: invalid data")

    monkeypatch.setattr(decode, "run_subprocess", _fail)
    with pytest.raises(decode.DecodeError, match="invalid data"):
        decode.decode_music(tiny_fixture)


# ---------------------------------------------------------------------------
# ingest() — single entrypoint, routing + content hash
# ---------------------------------------------------------------------------


def test_ingest_requires_exactly_one_of_file_or_url():
    with pytest.raises(ValueError):
        ingest_pkg.ingest()
    with pytest.raises(ValueError):
        ingest_pkg.ingest(file="a", url="b")


def test_ingest_missing_file_raises():
    with pytest.raises(ingest_pkg.IngestError):
        ingest_pkg.ingest(file="/no/such/file.wav")


def test_ingest_file_passthrough_produces_hashed_wav(tiny_fixture):
    result = ingest_pkg.ingest(file=tiny_fixture)
    assert result.wav_path.exists()
    info = sf.info(result.wav_path)
    assert info.samplerate == 44100
    assert info.channels == 2

    expected_hash = hashlib.sha256(result.wav_path.read_bytes()).hexdigest()
    assert result.content_hash == expected_hash


def test_ingest_enforces_file_size_cap(tiny_fixture, monkeypatch):
    monkeypatch.setattr(ingest_pkg, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ingest_pkg.IngestError, match="exceeds cap"):
        ingest_pkg.ingest(file=tiny_fixture)


def test_ingest_enforces_duration_cap_post_decode(tiny_fixture, monkeypatch):
    monkeypatch.setattr(ingest_pkg, "MAX_DURATION_SECONDS", 0.1)
    with pytest.raises(ingest_pkg.IngestError, match="duration"):
        ingest_pkg.ingest(file=tiny_fixture)


def test_ingest_url_routes_through_fetch_then_decode(tiny_fixture, monkeypatch, tmp_path):
    fetched_dir = tmp_path / "fetched"
    fetched_dir.mkdir()
    fetched_path = fetched_dir / "source.wav"
    fetched_path.write_bytes(tiny_fixture.read_bytes())

    monkeypatch.setattr(ingest_pkg.fetch_mod, "fetch", lambda url, timeout=None: fetched_path)

    result = ingest_pkg.ingest(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    info = sf.info(result.wav_path)
    assert info.samplerate == 44100
    assert info.channels == 2
    assert not fetched_dir.exists()  # fetch dir cleaned up after decode
