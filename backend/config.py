"""Tiers, sample rates, pool size, TTL — quality/speed is config, not branching."""

from __future__ import annotations

import os
from dataclasses import dataclass
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

MODES = ("music", "video", "full", "singing")
DEFAULT_MODE = "music"

# Fast/Balanced/Best map to model + shifts + segment size (decision 6, PRD §5).
TIERS = {
    "fast": {"music": "htdemucs_onnx_q", "shifts": 0, "segment": 7},
    "balanced": {"music": "htdemucs_onnx", "shifts": 0, "segment": 7},
    "best": {"music": "htdemucs_ft", "shifts": 1, "segment": 7},
}
# No single DEFAULT_TIER constant: which tier a job runs at when the caller
# doesn't pin one is now architecture-aware (backend.arch.resolve_default_tier,
# ARCH_RUNTIME_PROFILES below) — e.g. "fast" on a VNNI x86_64 host, "balanced"
# elsewhere, per what's actually measured to be faster on that hardware.

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


# Each provider entry is either a bare EP name, or an (EP name, provider
# options dict) pair — exactly the two shapes onnxruntime's own
# `providers=[...]` list accepts.
ProviderSpec = str | tuple[str, dict]


@dataclass(frozen=True)
class RuntimeProfile:
    tier: str  # which tier this arch defaults to when a job doesn't pin one
    providers: tuple[ProviderSpec, ...]  # ONNX Runtime EPs to prefer, in order (backend/arch.py)


# Architecture-aware runtime selection (backend/arch.py detects the host at
# startup; scripts/bench_arch.py measures real-time-factor per (arch, tier)
# so these come from measurement, not guesses — re-run it on new hardware
# before trusting the "tier" defaults here). Every entry is a *preference*:
# demucs_onnx.py always appends CPUExecutionProvider as the final fallback,
# and onnxruntime itself silently skips a provider that isn't compiled into
# the current build — so no arch or provider is ever a hard requirement.
ARCH_RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    # x86_64 with AVX-512 VNNI: fused int8 GEMM kernels make the "fast"
    # (int8-quantized) model genuinely fast here, unlike the non-VNNI
    # reference machine TIER_RTF above was measured on (see that constant's
    # docstring — onnxruntime's plain CPU EP has no fused int8 kernel for
    # this graph without VNNI). OpenVINO's CPU EP is what exploits VNNI.
    "x86_64_vnni": RuntimeProfile(tier="fast", providers=("OpenVINOExecutionProvider",)),
    # Apple Silicon / Graviton: no fused int8 path, so default to the fp32
    # ("balanced") model, on plain CPUExecutionProvider — NOT CoreML. Both
    # halves of that were measured, not assumed, on this graph specifically:
    # (1) CoreML's default compute-unit selection routes onto the Neural
    # Engine, which runs in fp16 — the STFT magnitude spectrogram's dynamic
    # range overflows fp16, producing silent `inf` stems (bit-for-bit
    # reproduced via DemucsONNXSeparator(providers=("CoreMLExecutionProvider",))
    # against test_onnx_vs_oracle on this host). (2) Pinning CoreML to
    # "CPUOnly" compute units fixes that correctness bug (~2e-4 max-abs-diff
    # vs. the oracle) but is ~16x *slower* than just using
    # CPUExecutionProvider directly on a clean, uncontended run (RTF 3.22 vs
    # 0.20 on a 15s real clip) — CoreML's graph conversion/partitioning
    # overhead on this op pattern isn't worth paying once it can't touch the
    # GPU/ANE anyway. Net result: CoreML has no role here at all, correct or
    # not — plain CPU wins outright.
    "arm64": RuntimeProfile(tier="balanced", providers=()),
    # x86_64 without VNNI, or an unrecognized arch: same fp32 default as
    # arm64 (int8 isn't faster without fused kernels here either — same
    # reasoning as the "fast" tier's TIER_RTF note), plain CPU EP only.
    "default": RuntimeProfile(tier="balanced", providers=()),
}
