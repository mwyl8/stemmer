"""Tiers, sample rates, pool size, TTL — quality/speed is config, not branching."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STEMS_DIR = DATA_DIR / "stems"
DB_PATH = DATA_DIR / "stemmer.db"

# Music path is locked at 44.1kHz stereo; only the speech-only Bandit path may use 16k mono.
MUSIC_SAMPLE_RATE = 44100
SPEECH_SAMPLE_RATE = 16000

# Isolated subprocess worker pool cap — do not let this grow unbounded.
POOL_SIZE = int(os.environ.get("STEMMER_POOL_SIZE", 2))

# Threaded runtime: intra-op threads default to physical cores.
INTRA_OP_THREADS = os.cpu_count() or 4

# Ephemeral persistence: auto-delete on TTL.
TTL_SECONDS = int(os.environ.get("STEMMER_TTL_SECONDS", 24 * 3600))

# Ingestion caps (validated before yt-dlp/ffmpeg ever run).
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_DURATION_SECONDS = 15 * 60  # 15 minutes
FETCH_TIMEOUT_SECONDS = 120
DECODE_TIMEOUT_SECONDS = 120

# Isolated-subprocess separation timeout (runner.py) and ffmpeg mp3-preview
# encode timeout (pool.py). Separation is the slow one — CPU-bound, minutes
# for a long song; encoding is just a transcode, seconds at most.
SEPARATE_TIMEOUT_SECONDS = int(os.environ.get("STEMMER_SEPARATE_TIMEOUT_SECONDS", 900))
ENCODE_TIMEOUT_SECONDS = 60
MP3_BITRATE = "192k"

# How often the TTL background task sweeps for expired jobs.
PURGE_INTERVAL_SECONDS = int(os.environ.get("STEMMER_PURGE_INTERVAL_SECONDS", 300))

MODES = ("music", "video", "full")
DEFAULT_MODE = "music"

# Fast/Balanced/Best map to model + shifts + segment size (decision 6, PRD §5).
TIERS = {
    "fast": {"music": "htdemucs_onnx_q", "shifts": 0, "segment": 7},
    "balanced": {"music": "htdemucs_onnx", "shifts": 0, "segment": 7},
    "best": {"music": "htdemucs_ft", "shifts": 1, "segment": 7},
}
DEFAULT_TIER = "fast"
