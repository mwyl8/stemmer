"""SQLite: jobs, stems, content-hash cache. Stems live on disk; only metadata here.

No pickle for cross-process state — rows are plain columns, paths are strings.

Reads `config.DB_PATH`/`config.STEMS_DIR` at call time (not as bound default
arguments) so tests can monkeypatch `backend.config.DB_PATH` to an isolated
temp database per test.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    status        TEXT NOT NULL,      -- queued | running | done | error | expired
    stage         TEXT NOT NULL,      -- queued | downloading | decoding | separating | encoding | done | error
    mode          TEXT NOT NULL,      -- music | video | full
    tier          TEXT NOT NULL,      -- fast | balanced | best
    stem_count    INTEGER NOT NULL DEFAULT 4,  -- music/full mode only: 4 (default) or 6 (htdemucs_6s)
    source_type   TEXT NOT NULL,      -- upload | youtube | tiktok | instagram
    source_ref    TEXT,               -- original filename/temp upload path, or URL
    content_hash  TEXT,
    progress_pct  INTEGER NOT NULL DEFAULT 0,  -- 0-100
    error         TEXT,
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL,
    chunks_done       INTEGER NOT NULL DEFAULT 0,
    chunks_total      INTEGER NOT NULL DEFAULT 0,
    stage_timings     TEXT NOT NULL DEFAULT '{}',  -- JSON {stage: {started_at, ended_at}}
    stage_started_at  TEXT,           -- when the current `stage` began
    eta_seconds       REAL,           -- seeded from audio_duration*RTF, refined from chunk throughput
    from_cache        INTEGER NOT NULL DEFAULT 0,
    source_codec       TEXT,          -- ffprobe codec_name of the original source, pre-normalization
    source_bitrate     INTEGER,       -- bits/sec
    source_sample_rate INTEGER,
    source_channels    INTEGER,
    source_wav_path    TEXT,          -- normalized WAV persisted in job_dir(id) for this job's TTL,
                                       -- so "A/B against original" and "re-run at a different mode/
                                       -- tier" don't need the raw upload/URL again (pool.py)
    runtime_arch        TEXT,         -- backend.arch.HostArch.profile_key that actually ran this job
    runtime_provider    TEXT,         -- ONNX Runtime execution provider actually in effect
    runtime_model       TEXT          -- ONNX model file (stem) actually loaded
);

CREATE INDEX IF NOT EXISTS idx_jobs_content_hash ON jobs (content_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_expires_at ON jobs (expires_at);

CREATE TABLE IF NOT EXISTS stems (
    id       TEXT PRIMARY KEY,
    job_id   TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    name     TEXT NOT NULL,          -- vocals | drums | bass | other | speech | music | effects
    format   TEXT NOT NULL,          -- wav | mp3
    path     TEXT NOT NULL,
    duration REAL
);

CREATE INDEX IF NOT EXISTS idx_stems_job_id ON stems (job_id);

-- Dedup key is (content_hash, mode, tier, pipeline_version): the same audio
-- at a different tier is a different quality result, so it gets its own
-- cache entry. pipeline_version (config.PIPELINE_VERSION) is part of the key
-- so a separation/limiting/encoding code change can't serve stale results
-- for audio that was already processed under the old behavior.
-- ON DELETE CASCADE means a cache entry disappears the moment the job it
-- points at is purged (TTL or DELETE /jobs/{id}) — it can never point at
-- stems that no longer exist on disk.
CREATE TABLE IF NOT EXISTS cache (
    content_hash     TEXT NOT NULL,
    mode             TEXT NOT NULL,
    tier             TEXT NOT NULL,
    pipeline_version INTEGER NOT NULL,
    job_id           TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (content_hash, mode, tier, pipeline_version)
);
"""


# Columns added after the initial jobs table shipped. CREATE TABLE IF NOT
# EXISTS above only applies to brand-new databases; a pre-existing
# data/stemmer.db needs each of these ALTER TABLE'd in explicitly (idempotent
# — skipped if the column's already there) so upgrading in place never loses
# job history.
_JOBS_MIGRATIONS = [
    ("stem_count", "INTEGER NOT NULL DEFAULT 4"),
    ("chunks_done", "INTEGER NOT NULL DEFAULT 0"),
    ("chunks_total", "INTEGER NOT NULL DEFAULT 0"),
    ("stage_timings", "TEXT NOT NULL DEFAULT '{}'"),
    ("stage_started_at", "TEXT"),
    ("eta_seconds", "REAL"),
    ("from_cache", "INTEGER NOT NULL DEFAULT 0"),
    ("source_codec", "TEXT"),
    ("source_bitrate", "INTEGER"),
    ("source_sample_rate", "INTEGER"),
    ("source_channels", "INTEGER"),
    ("source_wav_path", "TEXT"),
    ("runtime_arch", "TEXT"),
    ("runtime_provider", "TEXT"),
    ("runtime_model", "TEXT"),
]


def job_dir(job_id: str) -> Path:
    return config.STEMS_DIR / job_id


def ensure_data_dirs() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.STEMS_DIR.mkdir(parents=True, exist_ok=True)


def init_db(db_path: Path | None = None) -> None:
    ensure_data_dirs()
    with sqlite3.connect(db_path or config.DB_PATH) as conn:
        conn.executescript(SCHEMA)
        _migrate_jobs_table(conn)
        _migrate_cache_table(conn)


def _migrate_jobs_table(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    # `progress` (REAL 0-1) was renamed to `progress_pct` (INTEGER 0-100).
    # ALTER TABLE ... RENAME COLUMN would keep the old REAL type affinity
    # forever (SQLite has no ALTER COLUMN TYPE), which would silently turn
    # every future write back into a float — so add the new INTEGER column
    # fresh, backfill it, and drop the old one instead.
    if "progress" in existing and "progress_pct" not in existing:
        conn.execute("ALTER TABLE jobs ADD COLUMN progress_pct INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE jobs SET progress_pct = CAST(ROUND(progress * 100) AS INTEGER)")
        conn.execute("ALTER TABLE jobs DROP COLUMN progress")
        existing.discard("progress")
        existing.add("progress_pct")
    for column, decl in _JOBS_MIGRATIONS:
        if column not in existing:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {decl}")
    conn.commit()


def _migrate_cache_table(conn: sqlite3.Connection) -> None:
    # pipeline_version joined the PRIMARY KEY, and SQLite can't ALTER a
    # table's primary key in place, so a pre-existing cache table (from
    # before PIPELINE_VERSION existed) is rebuilt rather than patched.
    # Old rows get pipeline_version=0, a sentinel below any real
    # PIPELINE_VERSION (starts at 1) — every one of them misses on lookup,
    # which is the intended effect: entries written under unknown, possibly
    # pre-fix code are never served as if they matched today's pipeline.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(cache)")}
    if not existing or "pipeline_version" in existing:
        return
    conn.execute("ALTER TABLE cache RENAME TO cache_old")
    conn.execute(
        "CREATE TABLE cache ("
        "    content_hash     TEXT NOT NULL,"
        "    mode             TEXT NOT NULL,"
        "    tier             TEXT NOT NULL,"
        "    pipeline_version INTEGER NOT NULL,"
        "    job_id           TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,"
        "    created_at       TEXT NOT NULL,"
        "    PRIMARY KEY (content_hash, mode, tier, pipeline_version)"
        ")"
    )
    conn.execute(
        "INSERT INTO cache (content_hash, mode, tier, pipeline_version, job_id, created_at) "
        "SELECT content_hash, mode, tier, 0, job_id, created_at FROM cache_old"
    )
    conn.execute("DROP TABLE cache_old")
    conn.commit()


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
