"""cache.py's dedup key includes pipeline_version (config.PIPELINE_VERSION)
precisely so a separation/limiting/encoding code change invalidates results
computed under the old code — see cache.py's module docstring for the
incident this guards against: a clipping fix was invisible on already-cached
audio until the DB was wiped by hand, because the old key was just
(content_hash, mode, tier).

cache.find() joins against jobs.status = 'done', so these tests create real
(done) job rows via jobs.py rather than bare job-id strings.
"""

from __future__ import annotations

from backend import cache, jobs


def _make_done_job(mode: str = "music", tier: str = "balanced") -> str:
    job_id = jobs.create_job(mode, tier, "upload", "/fake/path.wav")
    jobs.update_stage(job_id, jobs.Stage.DONE)
    return job_id


def test_find_hits_on_identical_key(db):
    job_id = _make_done_job()
    cache.put("hash-a", "music", "balanced", 1, job_id)
    assert cache.find("hash-a", "music", "balanced", 1) == job_id


def test_find_misses_on_different_pipeline_version(db):
    job_id = _make_done_job()
    cache.put("hash-a", "music", "balanced", 1, job_id)
    assert cache.find("hash-a", "music", "balanced", 2) is None


def test_bumping_pipeline_version_invalidates_prior_cache_entries_for_identical_audio(db):
    """The exact scenario from the incident: same audio, same mode/tier,
    but the pipeline changed (e.g. the clipping fix) between the two runs —
    the second run must miss and re-separate, not serve the stale result."""
    content_hash = "same-audio-hash"
    old_version = 1
    new_version = old_version + 1

    old_job_id = _make_done_job()
    cache.put(content_hash, "music", "balanced", old_version, old_job_id)
    assert cache.find(content_hash, "music", "balanced", old_version) == old_job_id

    # Code changes; PIPELINE_VERSION bumps. Same audio, same mode/tier.
    assert cache.find(content_hash, "music", "balanced", new_version) is None


def test_find_misses_on_different_mode_or_tier_as_before(db):
    job_id = _make_done_job()
    cache.put("hash-a", "music", "balanced", 1, job_id)
    assert cache.find("hash-a", "video", "balanced", 1) is None
    assert cache.find("hash-a", "music", "fast", 1) is None


def test_karaoke_mode_gets_its_own_cache_entry_distinct_from_other_modes(db):
    """New mode, same generic (content_hash, mode, tier, pipeline_version)
    key cache.py already used for singing/full — same audio run in karaoke
    mode must not collide with (or be served by) an entry from any other
    mode, and vice versa."""
    karaoke_job_id = _make_done_job(mode="karaoke")
    cache.put("hash-a", "karaoke", "balanced", 1, karaoke_job_id)

    music_job_id = _make_done_job(mode="music")
    cache.put("hash-a", "music", "balanced", 1, music_job_id)

    assert cache.find("hash-a", "karaoke", "balanced", 1) == karaoke_job_id
    assert cache.find("hash-a", "music", "balanced", 1) == music_job_id
    assert cache.find("hash-a", "singing", "balanced", 1) is None
