# CLAUDE.md — Stemmer (CPU stem separation)

Guidance for Claude Code in this repo. Full spec: `StemSep_PRD.md`. Model/algorithm background: `StemSep_Explainer.md`. This file is the fast-reference + hard rules.

## 1. What we're building
A CPU-only web service that takes an uploaded `mp3/wav/mp4` **or a public link** (YouTube/TikTok/Instagram), separates the audio into **stems** (music: vocals/drums/bass/other; cinematic: speech/music/effects), and lets the user **play back, inspect (waveforms, mute/solo), and download** each stem. Open-source ML only. **CPU-only inference** — optimization is first-class.

## 2. Locked v1 decisions
Modes: **Music + Video + Full chained**. Stems: **4-stem music + speech/music/effects**. Frontend: **multitrack player + mute/solo/volume + downloads**. Persistence: **ephemeral, TTL auto-delete**. Execution: **in-process asyncio + isolated subprocess worker pool (cap N)**. Runtime: **ONNX Runtime + int8 quantization from the start** (PyTorch = reference oracle only). Packaging: **local `uv`, no Docker**. Stack: **best-fit per layer**. Concurrency: **small pool, a few concurrent**. Ingestion: **file + YouTube + TikTok + Instagram**. Testing: **unit tests + small SDR harness**. Output: **WAV + MP3 preview + zip**.

## 3. Architecture (keep these boundaries)
- **`backend/app.py` is a thin switchboard** — HTTP routing only.
- **All separators sit behind `separators/base.py`** (`separate(audio) -> {stem: waveform}`). Adding/swapping a model never touches the API or frontend.
- **Heavy compute runs in an isolated subprocess** (`runner.py`), dispatched by an async **worker pool** (`pool.py`, cap `POOL_SIZE = N`). The web process never blocks on separation.
- **State in SQLite + files.** `storage.py` owns the DB (jobs, stems, cache). Stems on disk. Dedup by **content hash**.
- **Quality/speed is config**, not branching — tiers in `config.py`.
- **Frontend**: React + Vite + Tailwind; stem player is `wavesurfer.js` + multitrack plugin.
- **Progress is real, never faked.** The worker reports `chunks_done / chunks_total` as it goes; the UI shows a live elapsed timer, a chunk-derived progress bar, an ETA from the tier's measured real-time factor, and per-stage timings. Never ship an indeterminate spinner for separation — the chunk loop already knows the true fraction complete.

## 4. Tech choices (chosen per layer on merit)
- **API:** FastAPI (async, Python-native for the models).
- **Inference:** **ONNX Runtime, int8-quantized** — the product path. PyTorch Demucs lives in `_oracle_torch.py` for eval ground-truth ONLY.
- **Music separation:** Demucs `htdemucs` (ONNX); `htdemucs_ft` only for the opt-in "Best" tier.
- **Speech separation:** Bandit-family (`speech/music/effects`). **Full mode** chains Bandit → Demucs on the music bus.
- **CPU path:** ONNX + int8 quant + segment/overlap-add + per-stem sub-models + threaded runtime (OpenVINO on Intel, CoreML on Mac).
- **Ingestion:** `yt-dlp` (YouTube/TikTok/Instagram + file passthrough) + `ffmpeg` (normalize → WAV 44.1k stereo; 16k mono only for the speech-only path).
- **Env:** **`uv`** for Python. Node already installed at `~/.local/node`.

## 5. Build order
Phase 0 skeleton/env → 1 Demucs **via ONNX (quantized)** behind the interface, subprocess runner, oracle diff-check → 2 ingestion (yt-dlp all sources + ffmpeg) → 3 jobs + API + worker pool (cap N) + cache + TTL → 4 speech mode + chained Full mode → 5 frontend (multitrack player) → 6 eval harness + tune tiers from measurements. Each phase independently demoable. **CPU optimization is baked into Phase 1, not a late phase.**

## 6. Things NOT to do
- **Do not run separation synchronously in the request handler.** Queue it; run in the subprocess pool.
- **Do not load models per-request or per-job.** Load once per worker and reuse.
- **Do not make PyTorch the product inference path.** ONNX (quantized) serves requests; `_oracle_torch.py` is reference-only.
- **Do not default to `htdemucs_ft`** — it's 4× CPU cost. Default is single-model `htdemucs`; `_ft` is the "Best" tier only.
- **Do not let the worker pool grow unbounded.** Cap at `POOL_SIZE = N`; queue the rest.
- **Do not hold whole long files in memory.** Segment/stream; serve mp3 previews (large Web Audio decodes crash wavesurfer).
- **Do not downsample the music path below 44.1 kHz.** Only the speech-only Bandit path may use 16 kHz mono.
- **Do not fetch private/authenticated content or bypass logins/paywalls.** Public URLs only; surface `yt-dlp` errors. Authorized content only.
- **Do not persist source or stem audio indefinitely.** Auto-delete on TTL and on `DELETE`. Never redistribute fetched third-party audio beyond the session.
- **Do not trust user links blindly.** Validate/normalize URLs; cap duration and file size; run `yt-dlp`/`ffmpeg` sandboxed with a timeout.
- **Do not use `pickle` for cross-process state.** Safe serialization; state in DB/files.
- **Do not skip the content-hash cache.** Re-separating identical audio on CPU wastes minutes.
- **Do not commit model weights to git.** Gitignore `models/`; pull + quantize on setup; pin versions.
- **Do not `brew install python`** on this machine (brew compiles from source). Use `uv`.
- **Do not rebase shared branches.** Squash-merge; feature branches stay linear.

## 7. Definition of done (v1)
File **and** public link (YT/TikTok/IG) both produce stems; Music + Video + chained Full mode work; CPU-only via **quantized ONNX** within the agreed latency (numbers recorded per tier); **worker pool** runs up to N concurrent jobs; multitrack player with per-stem waveform + mute/solo/volume + downloads (wav/mp3/zip); progress UI with live elapsed timer, chunk-derived bar, ETA and per-stage timings; content-hash cache; TTL auto-purge; `test_router` + `test_ingest` green; eval harness reports per-tier SDR + latency.
