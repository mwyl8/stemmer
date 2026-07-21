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

# Bump whenever separation, limiting, or encoding behavior changes — it's
# part of the cache key (cache.py) so a code change invalidates stale
# results instead of serving them from before the fix (e.g. the PCM_16
# clipping bug: identical audio kept resolving to pre-fix clipped stems
# until the DB was wiped by hand).
#
# v3: peak limiting switched from per-stem to global (apply_peak_safety_global)
# — stems cached under v2 or earlier were each scaled by their own peak,
# which destroys inter-stem balance (drums vs. vocals), so they must not be
# served as cache hits anymore.
PIPELINE_VERSION = 3

MODES = ("music", "video", "full")
DEFAULT_MODE = "music"

# Fast/Balanced/Best map to model + shifts + segment size (decision 6, PRD §5).
TIERS = {
    "fast": {"music": "htdemucs_onnx_q", "shifts": 0, "segment": 7},
    "balanced": {"music": "htdemucs_onnx", "shifts": 0, "segment": 7},
    "best": {"music": "htdemucs_ft", "shifts": 1, "segment": 7},
}
DEFAULT_TIER = "fast"

# 4-stem (vocals/drums/bass/other) is the locked default; 6-stem
# (+guitar/piano, htdemucs_6s) is opt-in — same STFT-split ONNX export path
# (scripts/export_onnx.py --model), just a different pretrained checkpoint.
MUSIC_MODELS = {4: "htdemucs", 6: "htdemucs_6s"}
STEM_COUNTS = tuple(MUSIC_MODELS)
DEFAULT_STEM_COUNT = 4

# Measured CPU real-time-factor (separation seconds per second of audio) per
# tier, htdemucs/4-stem, music mode — used to seed eta_seconds before a job's
# own chunk throughput is available (jobs.py/pool.py), then refined from
# actual chunk timing as chunks land. Measured on the reference dev machine
# (INTRA_OP_THREADS=15) via a 20s clip of data/test.mp3, averaged over
# repeated runs; "fast" (int8 dynamic-quantized) measuring *slower* than
# "balanced" (fp32) is real and reproducible here — onnxruntime's CPU EP
# lacks fused int8 GEMM kernels for this graph's op pattern, so dynamic
# quantization shrinks the model (58MB vs 174MB) without speeding it up on
# this machine; re-measure via the Phase 6 eval harness before trusting these
# elsewhere. "best" (htdemucs_ft) isn't wired to the product path yet
# (router.py raises NotImplementedError) — its RTF is the PRD's documented
# 4x-of-single-model estimate, not a measurement.
TIER_RTF = {
    "fast": 0.45,
    "balanced": 0.25,
    "best": 1.0,
}
