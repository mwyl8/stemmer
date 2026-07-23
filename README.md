# Stemmer

A CPU-only web service that splits an uploaded `mp3`/`wav`/`mp4` — or a public
YouTube/TikTok/Instagram link — into stems you can play back, mute/solo/pan,
inspect as waveforms or spectrograms, mix, and download individually or as a
zip. No GPU required, no closed-source models: open-source ML, optimized to
run acceptably fast on CPU.

## Modes

Selectable per job:

- **Music** — `vocals` / `drums` / `bass` / `other` (Demucs `htdemucs`; an
  opt-in 6-stem variant adds `guitar`/`piano`).
- **Video** — `speech` / `music` / `effects` (Bandit-family BandSplitRNN).
- **Full** (chained) — Bandit splits the mix into speech/music/effects, then
  Demucs further splits the *music* stem into vocals/drums/bass/other. Final
  output: `speech`, `vocals`, `drums`, `bass`, `other`.
- **Speech vs Singing** (chained) — same Bandit → Demucs chain as Full mode,
  merged differently: `spoken_speech` (Bandit's speech stem), `sung_vocals`
  (Demucs's vocals stem), `instruments` (every other Demucs source, summed).
  Singing-voice-vs-speech is a known-hard, unsolved MSS problem — see
  `scripts/eval_singing_vs_speech.py` for a measured, honestly-reported bleed
  number on a synthetic fixture, not a claim of clean separation.

## CPU-only, and how that's met

There's no GPU path to fall back on, so CPU performance is a first-class
constraint, not an afterthought bolted on later:

- **ONNX Runtime, quantized where it actually helps**, not PyTorch, serves
  every request (see below).
- **Segment/overlap-add chunking** keeps memory bounded and lets long files
  stream through fixed-size model inputs instead of one enormous forward pass.
- **One model load per subprocess, not per request** — the isolated runner
  loads the model once and reuses it for the whole file.
- **A capped worker pool** (`POOL_SIZE`) bounds how many separations run
  concurrently, so load past that cap queues instead of degrading everything
  running at once.
- **Content-hash caching** means identical audio is never re-separated —
  on CPU, minutes of redundant compute is real money.
- **Quality/speed is config, not branching**: `fast` / `balanced` / `best`
  tiers map to model + shift count + segment size in one place
  (`backend/config.py`), not scattered conditionals.

## Architecture

```
Browser (React + wavesurfer.js multitrack player)
        │  HTTP
        ▼
backend/app.py            thin FastAPI switchboard — routing only, no heavy work
        │  enqueue job id
        ▼
backend/pool.py           async WorkerPool, capped at POOL_SIZE concurrent jobs
        │  asyncio.to_thread(process_job)
        ▼
backend/jobs.py + ingest/  ingest (yt-dlp fetch / ffmpeg decode) → content-hash
        │                  cache check → skip straight to cached stems if hit
        ▼
backend/separators/router.py     mode + tier → picks/chains a Separator
        │
        ▼
backend/separators/base.py       Separator.separate(audio) -> {stem: waveform}
        │   (DemucsONNXSeparator | BanditSeparator | ChainedSeparator — same interface)
        ▼
backend/runner.py         spawns `python -m backend.runner` as an isolated
                           subprocess; loads the model once, separates, exits
        │
        ▼
storage.py (SQLite) + data/stems/  job/stem metadata + content-hash cache in
                                    SQLite, audio on disk, TTL sweep purges both
```

The load-bearing boundary is `separators/base.py`: every separator —
Demucs-via-ONNX, Bandit-via-PyTorch, or the chained composition of the two —
implements the same `separate(audio) -> {stem: waveform}` interface. Swapping
or adding a model touches exactly one file in `separators/`; the API, job
pipeline, and frontend never know which model ran.

Heavy compute never runs in the request handler and never runs on the event
loop thread. `POST /jobs` does only the minimal synchronous work an HTTP
handler can't defer (stream an upload to disk, or a cheap URL check) and
hands off immediately. The actual pipeline — ingest, separate, encode — runs
via `asyncio.to_thread` inside the capped worker pool, so at most `POOL_SIZE`
jobs run at once and everything past that queues instead of piling load onto
the box. Separation itself happens in a freshly spawned subprocess per job
(`backend/runner.py`), not in the worker pool's own process — the model loads
once per subprocess, runs the whole file, and the process exits, giving a
clean memory reclamation point and a hard kill switch on timeout that a
long-lived in-process worker wouldn't have.

The content-hash cache key is `(hash, mode, tier, pipeline_version)` —
`pipeline_version` (`config.PIPELINE_VERSION`) exists specifically so a
separation/limiting/encoding code change invalidates results computed under
the old code, rather than silently serving stale output for audio that was
already processed (this is exactly how a real clipping bug shipped invisibly
until the DB was wiped by hand — see `cache.py`'s module docstring).

## How the ONNX path works

The product inference path is **ONNX Runtime**, quantized where measurement
actually justifies it — PyTorch is not in the request path at all (only
`_oracle_torch.py`, used solely as reference ground-truth for correctness
tests and the eval harness).

The catch: `torch.onnx.export` can't trace htdemucs's `forward()` directly —
`aten::stft`/`aten::istft` with complex output and `view_as_complex` aren't
exportable ops (both the legacy tracer and the dynamo exporter fail on them).
The fix, the same approach used by `sevagh/demucs.onnx` and the Mixxx GSoC
2025 htdemucs export: **split the graph at the STFT boundary**. Everything
from the magnitude spectrogram through the conv/transformer network to the
pre-ISTFT output is pure real-valued tensor math and exports cleanly as
`DemucsCore` (`backend/separators/_demucs_core.py`); the STFT and ISTFT
themselves stay outside the graph as plain numpy (`_stft_numpy.py`) — cheap
FFT work that doesn't need acceleration anyway. `demucs_onnx.py` computes the
spectrogram, feeds only the real-valued magnitude + raw mixture into the ONNX
session, then reconstructs the waveform from the model's output on the numpy
side.

Long audio is handled with the same segment/overlap-add scheme Demucs itself
uses: the exported graph has a fixed input shape (`training_length`, Demucs's
native ~7.8s segment), so the file is chunked at a configured stride with 25%
overlap, each chunk is *centered* in a `training_length` window pulled from
surrounding real audio (never zero-padded except past the actual file
boundary — matching `TensorChunk.padded()`/`center_trim`), and outputs are
crossfaded back together with a triangular weight.

This was verified in stages, not asserted: the numpy STFT/ISTFT reimplementation
matches `torch.stft` to **~1e-6** (pure floating-point rounding — the
signature of "the algorithm is identical," not merely close), and the final
ONNX graph's output matches the PyTorch oracle to **~2e-4** max-abs-diff
(`test_onnx_vs_oracle.py`) — about −74 dB relative to full scale, well below
the noise floor of any recording and inaudible. That diff test earned its
keep once already: it's what caught a real bug where an early chunking
implementation zero-padded chunk boundaries instead of feeding real
surrounding context, a discrepancy a human ear would likely have missed.

Bandit (the speech/music/effects model) has no ONNX export yet — it's a
complex-mask band-split RNN, the same class of export problem Demucs had.
Per the PRD, PyTorch-behind-the-same-interface is the explicitly sanctioned
fallback until that export exists; `router.py` imports it lazily so
selecting music mode never pulls in `torch`.

## Current state vs. planned

Built, with passing tests:

- **Ingestion** — `yt-dlp` (YouTube/TikTok/Instagram, host-allowlisted,
  SSRF-guarded, duration/size capped, no cookies/auth) + `ffmpeg` normalize to
  44.1kHz stereo WAV; content-hash computed at ingest time.
- **Separation** — htdemucs exported to ONNX via the STFT-split approach
  above (4-stem default, opt-in 6-stem), Bandit speech/music/effects
  (PyTorch), and the chained Bandit→Demucs `full` mode, all behind one
  `Separator` interface, run in the isolated subprocess runner.
- **Jobs + API** — SQLite-backed job model, FastAPI routes (`POST /jobs`,
  status polling, stem/download/peaks endpoints, re-run, export-mix,
  `DELETE /jobs/{id}`), the capped async worker pool, content-hash +
  pipeline-version cache, TTL background purge.
- **Frontend** — React + Vite + Tailwind multitrack player: per-stem
  waveform/spectrogram toggle, mute/solo/pan/VU meters, A/B against the
  original mix, master volume/mute-all/reset-mix, saved mix presets, custom
  mix export, live progress (elapsed timer, chunk-derived bar, ETA,
  per-stage timings), recent-jobs list with re-run at a different mode/tier,
  shareable result links, server-side precomputed waveform peaks so the
  browser never decodes a full file just to draw a waveform.
- A Playwright end-to-end suite (`frontend/e2e/`) drives the real backend +
  frontend and asserts on rendered output — not just that the API responds,
  but that the browser actually painted something and threw no console
  errors, which is a different (and previously missing) class of coverage
  from the backend's unit/integration tests.

Open items — not silent gaps, tracked in `docs/StemSep_PRD.md` /
`docs/StemSep_PRD_Addendum.md`:

- **Instrument separation** (guitar/piano/organ/reeds beyond the 6-stem
  Demucs variant, ideally via a query-based model like Banquet) — scoped,
  not started.
- **SDR eval harness** (`backend/eval/harness.py`) — still a stub. Every
  quality/latency claim in this repo right now (tier RTFs, "fast" vs
  "balanced" behavior) is a measured-once comment or a documented estimate,
  not a tracked, repeatable benchmark. This is the one honest gap behind
  every other quality claim here.
- **Architecture-aware runtime selection** (`backend/arch.py`,
  `config.ARCH_RUNTIME_PROFILES`) now handles the ARM-vs-x86 int8 gap above
  automatically: on x86_64 with AVX-512 VNNI, jobs default to the int8 "fast"
  tier via the OpenVINO execution provider (fused int8 GEMM kernels actually
  help there); everywhere else (Apple Silicon, Graviton, non-VNNI x86)
  defaults to full-precision "balanced" on plain `CPUExecutionProvider`,
  matching what was measured on the reference machine. Every choice here is
  a preference, never a requirement — onnxruntime silently drops a provider
  that isn't compiled into the current build. CoreML was tried for Apple
  Silicon and measured out, not just skipped: its default compute-unit
  selection runs this graph on the Neural Engine in fp16, which overflows to
  `inf` on the STFT magnitude spectrogram's dynamic range (reproduced
  directly against `test_onnx_vs_oracle.py`); pinning it to
  `MLComputeUnits=CPUOnly` fixes that correctness bug (~2e-4 vs. the oracle)
  but made it measurably **~16x slower** than just using
  `CPUExecutionProvider` directly on a clean, uncontended run (RTF 3.22 vs.
  0.20 on a 15s real clip) — CoreML's conversion/partitioning overhead isn't
  worth paying once it can't touch the GPU/ANE anyway, so Apple Silicon gets
  plain CPU. Run `scripts/bench_arch.py` on new hardware before trusting the
  tier defaults there — and run it alone: separating two jobs concurrently
  on one box (e.g. this script racing a real request) caused severe
  CPU/memory contention in testing, at one point stalling a 900s subprocess
  timeout's own bookkeeping for tens of minutes. Nothing here reads
  `bench_arch.py`'s output automatically, by design (config changes stay
  deliberate, reviewed edits).
- **`best` tier (`htdemucs_ft`)** — deliberately unwired; `router.py` raises
  rather than silently falling back, per the "never default to `_ft`" rule.
- **Production deploy story** — local `uv`/`npm` dev setup only (no Docker,
  per the locked v1 decision); no packaged deploy path yet.

## Run it

Requires `uv` (Python) and Node on `PATH`, plus `ffmpeg` and `yt-dlp`
available as CLI tools. Model weights are **gitignored** — never committed —
pulled/exported/quantized locally via the scripts below.

```bash
# Backend deps
uv sync

# Pull + prepare model weights (gitignored — never committed)
uv run --group eval python scripts/export_onnx.py      # htdemucs -> ONNX (needs torch+demucs)
uv run python scripts/quantize.py                       # -> int8 "fast" tier weights
uv run --group speech python scripts/fetch_bandit_weights.py   # Bandit checkpoint, for video/full modes

# Run the API
uv run uvicorn backend.app:app --reload --port 8000
```

Frontend (separate terminal):

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173, proxies /jobs and /health to :8000
```

Tests:

```bash
uv run pytest                 # backend unit/integration tests; music mode needs the exported ONNX weights
uv run ruff check .

cd frontend
npm run lint
npm run build
npm run test:e2e              # Playwright, real backend + frontend, headless Chromium
```

Manual smoke checks (hit the real pipeline end-to-end, not part of `pytest`):

```bash
uv run python scripts/smoke.py                          # music mode on data/test.mp3
uv run --group speech python scripts/smoke_modes.py      # music/video/full/singing on the same clip
uv run python scripts/fetch_demo.py "<youtube-url>"      # real yt-dlp fetch
uv run python scripts/bench_arch.py                      # RTF per tier on this host's arch
uv run --group speech python scripts/eval_singing_vs_speech.py   # speech/singing bleed on a synthetic fixture
```

`POOL_SIZE`, `TTL_SECONDS`, and the separation/purge timeouts are all
overridable via `STEMMER_*` environment variables — see `backend/config.py`.
