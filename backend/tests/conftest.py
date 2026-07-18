"""Shared fixtures for Phase 3 (jobs/pool/app) tests: an isolated temp SQLite
DB + stems dir per test, so tests never touch the real data/ directory and
never see state left behind by another test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend import config
from backend.storage import init_db


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Redirect config.DB_PATH/DATA_DIR/STEMS_DIR to a fresh tmp dir and
    initialize the schema there. storage.py reads these at call time (not as
    bound defaults), so this monkeypatch is all that's needed."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "STEMS_DIR", tmp_path / "data" / "stems")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "stemmer.db")
    init_db()
    return tmp_path


@pytest.fixture
def make_upload(tmp_path):
    """Writes a tiny real WAV fixture into its own dedicated temp dir (like a
    real upload would get) and returns its path. Call multiple times for
    multiple independent "uploads" of the same or different audio content."""
    counter = {"n": 0}

    def _make(seconds: float = 1.0, sr: int = 44100, seed: int = 0) -> Path:
        counter["n"] += 1
        rng = np.random.default_rng(seed)
        data = (0.05 * rng.standard_normal((int(sr * seconds), 2))).astype(np.float32)
        d = tmp_path / f"upload-{counter['n']}"
        d.mkdir()
        path = d / "audio.wav"
        sf.write(path, data, sr)
        return path

    return _make
