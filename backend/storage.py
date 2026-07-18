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
    source_type   TEXT NOT NULL,      -- upload | youtube | tiktok | instagram
    source_ref    TEXT,               -- original filename/temp upload path, or URL
    content_hash  TEXT,
    progress      REAL NOT NULL DEFAULT 0,
    error         TEXT,
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL
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

-- Dedup key is (content_hash, mode, tier): the same audio at a different
-- tier is a different quality result, so it gets its own cache entry.
-- ON DELETE CASCADE means a cache entry disappears the moment the job it
-- points at is purged (TTL or DELETE /jobs/{id}) — it can never point at
-- stems that no longer exist on disk.
CREATE TABLE IF NOT EXISTS cache (
    content_hash TEXT NOT NULL,
    mode         TEXT NOT NULL,
    tier         TEXT NOT NULL,
    job_id       TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (content_hash, mode, tier)
);
"""


def job_dir(job_id: str) -> Path:
    return config.STEMS_DIR / job_id


def ensure_data_dirs() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.STEMS_DIR.mkdir(parents=True, exist_ok=True)


def init_db(db_path: Path | None = None) -> None:
    ensure_data_dirs()
    with sqlite3.connect(db_path or config.DB_PATH) as conn:
        conn.executescript(SCHEMA)


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
