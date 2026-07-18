"""API endpoint tests via FastAPI's TestClient. The separator is always
mocked (no ONNX/model loading) and, for URL tests, fetch.fetch() is mocked
too — no network, no subprocess heavy-lifting, so these stay fast.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend import pool as pool_mod
from backend.app import app
from backend.ingest import fetch as fetch_mod

SR = 44100


def _fake_separate_factory(data, sr=SR):
    def _fake(input_wav, output_dir, mode="music", tier="balanced", timeout=300):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for name in ("vocals", "drums", "bass", "other"):
            p = output_dir / f"{name}.wav"
            sf.write(p, data * 0.5, sr)
            out[name] = p
        return out

    return _fake


@pytest.fixture
def mocked_separator(monkeypatch):
    data = (0.05 * np.random.default_rng(0).standard_normal((SR * 2, 2))).astype(np.float32)
    monkeypatch.setattr(pool_mod.runner_mod, "separate_in_subprocess", _fake_separate_factory(data))
    return data


@pytest.fixture
def client(db, mocked_separator):
    with TestClient(app) as c:
        yield c


def _wait_for_terminal_status(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal status in time: {status}")


def _upload_bytes(seconds=1.0, sr=SR, seed=0) -> bytes:
    data = (0.05 * np.random.default_rng(seed).standard_normal((int(sr * seconds), 2))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_post_jobs_file_upload_end_to_end(client):
    resp = client.post(
        "/jobs",
        files={"file": ("song.wav", _upload_bytes(seed=1), "audio/wav")},
        data={"mode": "music", "tier": "balanced"},
    )
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]

    status = _wait_for_terminal_status(client, job_id)
    assert status["status"] == "done"
    assert status["mode"] == "music"
    assert status["tier"] == "balanced"
    assert len(status["stems"]) == 8  # 4 stems x {wav,mp3}


def test_post_jobs_file_upload_defaults_mode_and_tier(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=2), "audio/wav")})
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]
    status = _wait_for_terminal_status(client, job_id)
    assert status["mode"] == "music"  # DEFAULT_MODE
    assert status["tier"] == "fast"  # DEFAULT_TIER


def test_post_jobs_rejects_invalid_mode(client):
    resp = client.post(
        "/jobs",
        files={"file": ("song.wav", _upload_bytes(seed=3), "audio/wav")},
        data={"mode": "bogus"},
    )
    assert resp.status_code == 422


def test_post_jobs_rejects_invalid_tier(client):
    resp = client.post(
        "/jobs",
        files={"file": ("song.wav", _upload_bytes(seed=4), "audio/wav")},
        data={"tier": "bogus"},
    )
    assert resp.status_code == 422


def test_post_jobs_multipart_without_file_field_is_422(client):
    resp = client.post("/jobs", data={"mode": "music"}, files={})
    # starlette needs SOME multipart marker to route here; force content-type
    resp = client.post(
        "/jobs",
        files={"not_file": ("x.txt", b"hi", "text/plain")},
    )
    assert resp.status_code == 422


def test_post_jobs_url_rejects_unsupported_host(client):
    resp = client.post("/jobs", json={"url": "https://evil.com/video", "mode": "music"})
    assert resp.status_code == 400


def test_post_jobs_url_rejects_private_host(client):
    resp = client.post("/jobs", json={"url": "http://169.254.169.254/x", "mode": "music"})
    assert resp.status_code == 400


def test_post_jobs_json_without_url_is_422(client):
    resp = client.post("/jobs", json={"mode": "music"})
    assert resp.status_code == 422


def test_post_jobs_url_end_to_end(client, monkeypatch, tmp_path):
    fetched_dir = tmp_path / "fetched"
    fetched_dir.mkdir()
    fetched_path = fetched_dir / "video.wav"
    data = (0.05 * np.random.default_rng(5).standard_normal((SR, 2))).astype(np.float32)
    sf.write(fetched_path, data, SR)

    monkeypatch.setattr(fetch_mod, "fetch", lambda url, timeout=None: fetched_path)

    resp = client.post("/jobs", json={"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw", "tier": "balanced"})
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]

    status = _wait_for_terminal_status(client, job_id)
    assert status["status"] == "done"


def test_get_job_404_for_unknown_id(client):
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_get_stems_list(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=6), "audio/wav")})
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)

    stems = client.get(f"/jobs/{job_id}/stems").json()
    assert len(stems) == 8
    assert any(s["name"] == "vocals" and s["format"] == "wav" for s in stems)


def test_get_stem_streams_wav_and_mp3(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=7), "audio/wav")})
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)

    wav_resp = client.get(f"/jobs/{job_id}/stems/vocals", params={"format": "wav"})
    assert wav_resp.status_code == 200
    assert wav_resp.headers["content-type"] == "audio/wav"
    assert len(wav_resp.content) > 0

    mp3_resp = client.get(f"/jobs/{job_id}/stems/vocals", params={"format": "mp3"})
    assert mp3_resp.status_code == 200
    assert mp3_resp.headers["content-type"] == "audio/mpeg"

    default_resp = client.get(f"/jobs/{job_id}/stems/vocals")
    assert default_resp.status_code == 200
    assert default_resp.headers["content-type"] == "audio/mpeg"  # mp3 is the default (preview)


def test_get_stem_404_for_unknown_name(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=8), "audio/wav")})
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)
    assert client.get(f"/jobs/{job_id}/stems/nonexistent").status_code == 404


def test_get_stem_422_for_bad_format(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=9), "audio/wav")})
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)
    assert client.get(f"/jobs/{job_id}/stems/vocals", params={"format": "flac"}).status_code == 422


def test_download_zip_contains_all_wav_stems(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=10), "audio/wav")})
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)

    zip_resp = client.get(f"/jobs/{job_id}/download")
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"

    zf = zipfile.ZipFile(io.BytesIO(zip_resp.content))
    assert set(zf.namelist()) == {"vocals.wav", "drums.wav", "bass.wav", "other.wav"}


def test_download_409_before_job_done(client, monkeypatch):
    # block separation forever (until we're done asserting) so the job stays "running"
    import threading

    release = threading.Event()

    def _blocking(*args, **kwargs):
        release.wait(timeout=2)
        raise pool_mod.SeparationFailed("cancelled for test")

    monkeypatch.setattr(pool_mod.runner_mod, "separate_in_subprocess", _blocking)
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=11), "audio/wav")})
    job_id = resp.json()["job_id"]

    # give the worker a moment to pick it up and move past "queued"
    deadline = time.time() + 2
    while client.get(f"/jobs/{job_id}").json()["status"] == "queued" and time.time() < deadline:
        time.sleep(0.02)

    assert client.get(f"/jobs/{job_id}/download").status_code == 409
    release.set()


def test_delete_job_purges_and_404s_on_repeat(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=12), "audio/wav")})
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)

    del_resp = client.delete(f"/jobs/{job_id}")
    assert del_resp.status_code == 204

    status = client.get(f"/jobs/{job_id}").json()
    assert status["status"] == "expired"
    assert status["stems"] == []

    assert client.delete(f"/jobs/{job_id}").status_code == 404


def test_delete_unknown_job_404s(client):
    assert client.delete("/jobs/does-not-exist").status_code == 404
