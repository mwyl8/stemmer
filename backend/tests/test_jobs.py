"""Job lifecycle transitions and TTL purge, against an isolated temp DB."""

from __future__ import annotations

import time

from backend import jobs


def test_create_job_starts_queued(db):
    job_id = jobs.create_job("music", "balanced", "upload", "song.mp3")
    job = jobs.get_job(job_id)
    assert job.status == "queued"
    assert job.stage == jobs.Stage.QUEUED.value
    assert job.progress == 0
    assert job.mode == "music"
    assert job.tier == "balanced"
    assert job.source_type == "upload"
    assert job.source_ref == "song.mp3"
    assert job.content_hash is None
    assert job.error is None
    assert job.stems == []


def test_get_job_returns_none_for_unknown_id(db):
    assert jobs.get_job("does-not-exist") is None


def test_update_stage_advances_status_and_progress(db):
    job_id = jobs.create_job("music", "fast", "upload", "song.mp3")

    jobs.update_stage(job_id, jobs.Stage.DECODING)
    job = jobs.get_job(job_id)
    assert job.stage == "decoding"
    assert job.status == "running"
    assert 0 < job.progress < 1

    jobs.update_stage(job_id, jobs.Stage.SEPARATING)
    assert jobs.get_job(job_id).stage == "separating"

    jobs.update_stage(job_id, jobs.Stage.DONE)
    job = jobs.get_job(job_id)
    assert job.stage == "done"
    assert job.status == "done"
    assert job.progress == 1.0


def test_mark_error_sets_error_stage_and_message(db):
    job_id = jobs.create_job("music", "fast", "upload", "song.mp3")
    jobs.update_stage(job_id, jobs.Stage.SEPARATING)
    jobs.mark_error(job_id, "ffmpeg exploded")
    job = jobs.get_job(job_id)
    assert job.status == "error"
    assert job.stage == "error"
    assert job.error == "ffmpeg exploded"


def test_set_content_hash(db):
    job_id = jobs.create_job("music", "fast", "upload", "song.mp3")
    jobs.set_content_hash(job_id, "abc123")
    assert jobs.get_job(job_id).content_hash == "abc123"


def test_add_stem_appears_on_job(db):
    job_id = jobs.create_job("music", "fast", "upload", "song.mp3")
    jobs.add_stem(job_id, "vocals", "wav", "/data/stems/x/vocals.wav", 12.5)
    jobs.add_stem(job_id, "vocals", "mp3", "/data/stems/x/vocals.mp3", 12.5)
    job = jobs.get_job(job_id)
    assert len(job.stems) == 2
    names_formats = {(s.name, s.format) for s in job.stems}
    assert names_formats == {("vocals", "wav"), ("vocals", "mp3")}


# ---------------------------------------------------------------------------
# TTL purge
# ---------------------------------------------------------------------------


def test_purge_job_removes_files_and_marks_expired(db):
    job_id = jobs.create_job("music", "fast", "upload", "song.mp3")
    jobs.add_stem(job_id, "vocals", "wav", "x", 1.0)
    job_dir = db / "data" / "stems" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "vocals.wav").write_bytes(b"fake audio")

    assert jobs.purge_job(job_id) is True

    assert not job_dir.exists()
    job = jobs.get_job(job_id)
    assert job.status == "expired"
    assert job.stems == []


def test_purge_job_is_not_repeatable(db):
    job_id = jobs.create_job("music", "fast", "upload", "song.mp3")
    assert jobs.purge_job(job_id) is True
    assert jobs.purge_job(job_id) is False  # already purged — nothing to do


def test_purge_job_returns_false_for_unknown_id(db):
    assert jobs.purge_job("does-not-exist") is False


def test_purge_expired_jobs_only_purges_past_ttl(db):
    job_id = jobs.create_job("music", "fast", "upload", "song.mp3")
    job_dir = db / "data" / "stems" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "vocals.wav").write_bytes(b"x")

    # not expired yet: a purge sweep "now" should do nothing
    purged = jobs.purge_expired_jobs(now=time.time())
    assert purged == []
    assert job_dir.exists()
    assert jobs.get_job(job_id).status == "queued"

    # simulate the TTL having passed
    purged = jobs.purge_expired_jobs(now=time.time() + 24 * 3600 + 1)
    assert purged == [job_id]
    assert not job_dir.exists()
    assert jobs.get_job(job_id).status == "expired"


def test_purge_expired_jobs_skips_already_expired(db):
    job_id = jobs.create_job("music", "fast", "upload", "song.mp3")
    jobs.purge_job(job_id)
    # a later sweep shouldn't try (and fail) to re-purge an already-expired job
    purged = jobs.purge_expired_jobs(now=time.time() + 24 * 3600 + 1)
    assert purged == []
