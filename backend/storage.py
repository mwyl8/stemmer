"""SQLite: jobs, stems, content-hash cache. Stems live on disk; only metadata here.

No pickle for cross-process state — rows are plain columns, paths are strings.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.config import DATA_DIR, DB_PATH, STEMS_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    status        TEXT NOT NULL,      -- queued | running | done | error
    stage         TEXT NOT NULL,      -- queued | downloading | decoding | separating | encoding | done | error
    mode          TEXT NOT NULL,      -- music | video | full
    tier          TEXT NOT NULL,      -- fast | balanced | best
    source_type   TEXT NOT NULL,      -- upload | youtube | tiktok | instagram
    source_ref    TEXT,               -- original filename or URL
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

CREATE TABLE IF NOT EXISTS cache (
    content_hash TEXT PRIMARY KEY,
    job_id       TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL
);
"""


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STEMS_DIR.mkdir(parents=True, exist_ok=True)


def init_db(db_path: Path = DB_PATH) -> None:
    ensure_data_dirs()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
