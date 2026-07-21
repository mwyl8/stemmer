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

from backend import jobs
from backend import pool as pool_mod
from backend.app import app
from backend.ingest import fetch as fetch_mod

SR = 44100


def _fake_separate_factory(data, sr=SR):
    def _fake(input_wav, output_dir, mode="music", tier="balanced", stem_count=4, timeout=300, job_id=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for name in ("vocals", "drums", "bass", "other"):
            p = output_dir / f"{name}.wav"
            sf.write(p, data * 0.5, sr)
            out[name] = p
        if job_id is not None:
            from backend import jobs as jobs_mod

            jobs_mod.update_progress(job_id, chunks_done=1, chunks_total=1)
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


# ---------------------------------------------------------------------------
# Phase 6: live progress reporting (chunks_done/total, eta, from_cache).
# These replace the `mocked_separator` fixture's fake with one that reports
# chunk progress through jobs.update_progress() the same way the real child
# subprocess does (runner.py's on_chunk callback), with small sleeps between
# chunks so a polling loop actually observes multiple intermediate states.
# ---------------------------------------------------------------------------


def _write_fake_stems(output_dir, names, data, sr=SR):
    out = {}
    for name in names:
        p = output_dir / f"{name}.wav"
        sf.write(p, data * 0.5, sr)
        out[name] = p
    return out


def test_progress_advances_monotonically_and_chunks_never_exceed_total(client, monkeypatch):
    data = (0.05 * np.random.default_rng(20).standard_normal((SR, 2))).astype(np.float32)
    total_chunks = 5

    def _fake(input_wav, output_dir, mode="music", tier="balanced", stem_count=4, timeout=300, job_id=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for done in range(total_chunks + 1):
            if job_id is not None:
                jobs.update_progress(job_id, chunks_done=done, chunks_total=total_chunks)
            time.sleep(0.03)
        return _write_fake_stems(output_dir, ("vocals", "drums", "bass", "other"), data)

    monkeypatch.setattr(pool_mod.runner_mod, "separate_in_subprocess", _fake)

    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=20), "audio/wav")})
    job_id = resp.json()["job_id"]

    seen_progress = []
    seen_chunks = []
    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        seen_progress.append(status["progress_pct"])
        seen_chunks.append((status["chunks_done"], status["chunks_total"]))
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.01)

    assert status is not None and status["status"] == "done"
    assert seen_progress == sorted(seen_progress)  # never regresses
    for pct in seen_progress:
        assert isinstance(pct, int)
        assert 0 <= pct <= 100
    for done, total in seen_chunks:
        if total:
            assert done <= total  # chunks_done never exceeds chunks_total
    assert seen_chunks[-1] == (total_chunks, total_chunks)
    assert seen_progress[-1] == 100


def test_chained_mode_progress_is_monotonic_across_both_passes(client, monkeypatch):
    data = (0.05 * np.random.default_rng(21).standard_normal((SR, 2))).astype(np.float32)
    bandit_total, demucs_total = 3, 2
    grand_total = bandit_total + demucs_total

    def _fake(input_wav, output_dir, mode="full", tier="balanced", stem_count=4, timeout=300, job_id=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for done in range(bandit_total + 1):  # bandit pass: [0, bandit_total]
            if job_id is not None:
                jobs.update_progress(job_id, chunks_done=done, chunks_total=grand_total)
            time.sleep(0.02)
        for done in range(demucs_total + 1):  # demucs pass continues, doesn't reset to 0
            if job_id is not None:
                jobs.update_progress(job_id, chunks_done=bandit_total + done, chunks_total=grand_total)
            time.sleep(0.02)
        return _write_fake_stems(output_dir, ("speech", "vocals", "drums", "bass", "other"), data)

    monkeypatch.setattr(pool_mod.runner_mod, "separate_in_subprocess", _fake)

    resp = client.post(
        "/jobs",
        files={"file": ("song.wav", _upload_bytes(seed=21), "audio/wav")},
        data={"mode": "full"},
    )
    job_id = resp.json()["job_id"]

    seen_chunks_done = []
    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        seen_chunks_done.append(status["chunks_done"])
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.01)

    assert status is not None and status["status"] == "done"
    assert seen_chunks_done == sorted(seen_chunks_done)  # monotonic across the whole chained job
    assert max(seen_chunks_done) == grand_total
    assert bandit_total in seen_chunks_done  # the handoff point between passes was observed, not skipped


def test_cache_hit_reports_from_cache(client):
    upload_bytes = _upload_bytes(seed=22)

    resp1 = client.post("/jobs", files={"file": ("song.wav", upload_bytes, "audio/wav")})
    job1 = resp1.json()["job_id"]
    status1 = _wait_for_terminal_status(client, job1)
    assert status1["status"] == "done"
    assert status1["from_cache"] is False

    resp2 = client.post("/jobs", files={"file": ("song.wav", upload_bytes, "audio/wav")})
    job2 = resp2.json()["job_id"]
    status2 = _wait_for_terminal_status(client, job2)
    assert status2["status"] == "done"
    assert status2["from_cache"] is True


def test_job_summary_includes_progress_fields(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=23), "audio/wav")})
    job_id = resp.json()["job_id"]
    status = _wait_for_terminal_status(client, job_id)

    for key in (
        "stem_count",
        "progress_pct",
        "submitted_at",
        "stage_started_at",
        "stage_timings",
        "chunks_done",
        "chunks_total",
        "elapsed_seconds",
        "eta_seconds",
        "from_cache",
        "source_codec",
        "source_bitrate",
        "source_sample_rate",
        "source_channels",
    ):
        assert key in status
    assert status["stem_count"] == 4
    assert status["progress_pct"] == 100


def test_job_summary_reports_real_source_metadata(client):
    """The uploaded fixture is a real 44.1kHz stereo PCM WAV — ffprobe runs
    for real here (not mocked), so this is what the pipeline actually saw,
    not a canned value."""
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=24), "audio/wav")})
    job_id = resp.json()["job_id"]
    status = _wait_for_terminal_status(client, job_id)

    assert status["source_codec"] in ("pcm_s16le", "pcm_f32le")
    assert status["source_sample_rate"] == SR
    assert status["source_channels"] == 2
    assert isinstance(status["progress_pct"], int)
    assert status["elapsed_seconds"] >= 0
    assert isinstance(status["stage_timings"], dict)
    assert "separating" in status["stage_timings"]
    assert status["stage_timings"]["separating"]["started_at"]
    assert status["stage_timings"]["separating"]["ended_at"]


# ---------------------------------------------------------------------------
# Phase 7b Part B: A/B original, re-run at a different mode/tier, export mix.
# ---------------------------------------------------------------------------


def test_job_summary_reports_has_original(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=30), "audio/wav")})
    job_id = resp.json()["job_id"]
    status = _wait_for_terminal_status(client, job_id)
    assert status["has_original"] is True


def test_get_original_streams_mp3(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=31), "audio/wav")})
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)

    original_resp = client.get(f"/jobs/{job_id}/original")
    assert original_resp.status_code == 200
    assert original_resp.headers["content-type"] == "audio/mpeg"
    assert len(original_resp.content) > 0


def test_rerun_at_a_different_tier_reuses_source_without_reupload(client):
    resp = client.post(
        "/jobs",
        files={"file": ("song.wav", _upload_bytes(seed=32), "audio/wav")},
        data={"mode": "music", "tier": "fast"},
    )
    job_id = resp.json()["job_id"]
    original_status = _wait_for_terminal_status(client, job_id)
    assert original_status["tier"] == "fast"

    # by now the original upload's temp dir has already been cleaned up by
    # pool.py's own finally block -- a successful rerun proves it didn't
    # need it, only the persisted source.wav
    rerun_resp = client.post(f"/jobs/{job_id}/rerun", json={"tier": "balanced"})
    assert rerun_resp.status_code == 201
    new_job_id = rerun_resp.json()["job_id"]
    assert new_job_id != job_id

    new_status = _wait_for_terminal_status(client, new_job_id)
    assert new_status["status"] == "done"
    assert new_status["mode"] == "music"  # inherited from the original, not overridden
    assert new_status["tier"] == "balanced"
    assert new_status["source_sample_rate"] == original_status["source_sample_rate"]
    assert len(new_status["stems"]) == 8


def test_rerun_at_the_same_tier_is_a_cache_hit(client):
    resp = client.post(
        "/jobs",
        files={"file": ("song.wav", _upload_bytes(seed=33), "audio/wav")},
        data={"tier": "balanced"},
    )
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)

    rerun_resp = client.post(f"/jobs/{job_id}/rerun", json={"tier": "balanced"})
    new_job_id = rerun_resp.json()["job_id"]
    new_status = _wait_for_terminal_status(client, new_job_id)
    assert new_status["from_cache"] is True


def test_rerun_404s_for_unknown_job(client):
    assert client.post("/jobs/does-not-exist/rerun", json={}).status_code == 404


def test_rerun_409s_once_the_original_has_been_deleted(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=34), "audio/wav")})
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)

    assert client.delete(f"/jobs/{job_id}").status_code == 204
    assert client.post(f"/jobs/{job_id}/rerun", json={"tier": "balanced"}).status_code == 409


def test_export_mix_returns_a_downloadable_wav(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=35), "audio/wav")})
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)

    mix_resp = client.post(
        f"/jobs/{job_id}/export-mix",
        json={"tracks": [{"name": "drums", "muted": True}], "master_volume": 0.8},
    )
    assert mix_resp.status_code == 200
    assert mix_resp.headers["content-type"] == "audio/wav"
    assert len(mix_resp.content) > 0


def test_export_mix_422s_when_everything_is_muted(client):
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=36), "audio/wav")})
    job_id = resp.json()["job_id"]
    _wait_for_terminal_status(client, job_id)

    mix_resp = client.post(f"/jobs/{job_id}/export-mix", json={"master_muted": True})
    assert mix_resp.status_code == 422


def test_export_mix_409_before_job_done(client, monkeypatch):
    import threading

    release = threading.Event()

    def _blocking(*args, **kwargs):
        release.wait(timeout=2)
        raise pool_mod.SeparationFailed("cancelled for test")

    monkeypatch.setattr(pool_mod.runner_mod, "separate_in_subprocess", _blocking)
    resp = client.post("/jobs", files={"file": ("song.wav", _upload_bytes(seed=37), "audio/wav")})
    job_id = resp.json()["job_id"]

    deadline = time.time() + 2
    while client.get(f"/jobs/{job_id}").json()["status"] == "queued" and time.time() < deadline:
        time.sleep(0.02)

    assert client.post(f"/jobs/{job_id}/export-mix", json={}).status_code == 409
    release.set()
