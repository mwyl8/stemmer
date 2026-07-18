# Stemmer

A CPU-only web service for stem separation. Give it an uploaded `mp3`/`wav`/`mp4`
or a public YouTube/TikTok/Instagram link, and it splits the audio into stems
you can play back, mute/solo, inspect as waveforms, and download individually
or as a zip. No GPU required, no closed-source models — open-source ML,
optimized to run acceptably fast on CPU.

Two families of stems, selectable per job:

- **Music mode** — `vocals` / `drums` / `bass` / `other` (Demucs `htdemucs`)
- **Video mode** — `speech` / `music` / `effects` (Bandit-family BandSplitRNN)
- **Full mode** — both, chained: Bandit splits the mix into speech/music/effects,
  then Demucs further splits the *music* stem into vocals/drums/bass/other.
  Final output: `speech`, `vocals`, `drums`, `bass`, `other`.

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
storage.py (SQLite) + data/stems/  job/stem metadata in SQLite, audio on disk,
                                    TTL sweep auto-purges both
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
via `asyncio.to_thread` inside a capped worker pool (`POOL_SIZE`, default 2),
so at most N jobs run at once and everything past that queues instead of
piling load onto the box. Separation itself happens in a freshly spawned
subprocess per job (`backend/runner.py`), not in the worker pool's own
process — the model loads once per subprocess, runs the whole file, and the
process exits, giving a clean memory reclamation point and a hard kill switch
on timeout that a long-lived in-process worker wouldn't have.

## The ONNX / STFT-split approach

The product inference path is **ONNX Runtime, int8-quantized** — PyTorch is
not in the request path at all (only `_oracle_torch.py`, used solely as
reference ground-truth for correctness tests and the eval harness).

The catch: `torch.onnx.export` can't trace htdemucs's `forward()` directly —
`aten::stft`/`aten::istft` with complex output and `view_as_complex` aren't
exportable ops (both the legacy tracer and the dynamo exporter fail on them).
The fix, same approach used by `sevagh/demucs.onnx` and the Mixxx GSoC 2025
htdemucs export: **split the graph at the STFT boundary**. Everything from
the magnitude spectrogram through the conv/transformer network to the
pre-ISTFT output is pure real-valued tensor math and exports cleanly as
`DemucsCore` (`backend/separators/_demucs_core.py`); the STFT and ISTFT
themselves stay outside the graph as plain numpy (`_stft_numpy.py`) — cheap
FFT work that doesn't need acceleration anyway. `demucs_onnx.py` computes the
spectrogram, feeds only the real-valued magnitude + raw mixture into the
ONNX session, then reconstructs waveform from the model's output on the
numpy side.

Long audio is handled with the same segment/overlap-add scheme Demucs itself
uses: the exported graph has a fixed input shape (`training_length`, Demucs's
native ~7.8s segment), so the file is chunked at a configured stride with 25%
overlap, each chunk is *centered* in a `training_length` window pulled from
surrounding real audio (never zero-padded except past the actual file
boundary — matching `TensorChunk.padded()`/`center_trim`), and outputs are
crossfaded back together with a triangular weight. This is verified against
the PyTorch oracle directly (`test_onnx_vs_oracle.py`) — getting the padding
convention wrong measurably shifts output at chunk boundaries.

Bandit (the speech/music/effects model) has no ONNX export yet — it's a
complex-mask band-split RNN, the same class of export problem Demucs had, and
exporting it is scoped as later work (Phase 6-ish), not forgotten. Per the
PRD, PyTorch-behind-the-same-interface is the explicitly sanctioned fallback
until that export exists; `router.py` imports it lazily so selecting music
mode never pulls in `torch`.

Quality/speed is config, not branching: `config.TIERS` maps `fast` /
`balanced` / `best` to model + shift count + segment size. `fast` points at
the int8-quantized graph, `balanced` at full precision, `best` (`htdemucs_ft`,
a 4x-cost ensemble) is reserved for later — the router explicitly refuses to
default to it.

## Running it

Requires `uv` (Python) and Node (`~/.local/node` if following this repo's
setup) already on `PATH`, plus `ffmpeg` and `yt-dlp` available as CLI tools.

```bash
# Backend deps
uv sync

# Pull + prepare model weights (gitignored — never committed)
uv run --group eval python scripts/export_onnx.py      # htdemucs -> ONNX (needs torch+demucs)
uv run python scripts/quantize.py                       # -> int8 "fast" tier weights
uv run --group speech python scripts/fetch_bandit_weights.py   # Bandit checkpoint, for video/full modes

# Run the API
uv run uvicorn backend.app:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173, proxies /jobs and /health to :8000
```

Tests:

```bash
uv run pytest                 # unit tests; music mode needs the exported ONNX weights
uv run ruff check .
```

Manual smoke checks (hit the real pipeline end-to-end, not part of `pytest`):

```bash
uv run python scripts/smoke.py                          # music mode on data/test.mp3
uv run --group speech python scripts/smoke_modes.py      # music/video/full on the same clip
uv run python scripts/fetch_demo.py "<youtube-url>"      # real yt-dlp fetch
```

`POOL_SIZE`, `TTL_SECONDS`, and the separation/purge timeouts are all
overridable via `STEMMER_*` environment variables — see `backend/config.py`.

## Current state vs. planned

Built (phases 0–4, all with passing tests):

- **Phase 0** — repo skeleton, `uv` env, PRD/CLAUDE docs.
- **Phase 1** — htdemucs exported to ONNX via the STFT-split approach above,
  int8 quantization, wired behind `Separator`, run via the isolated subprocess
  runner, diffed against the PyTorch oracle.
- **Phase 2** — ingestion: `yt-dlp` (YouTube/TikTok/Instagram, host-allowlisted,
  SSRF-guarded, duration/size capped, no cookies/auth) + `ffmpeg` normalize to
  44.1kHz stereo WAV; content-hash computed at ingest time.
- **Phase 3** — job model + SQLite storage, FastAPI routes (`POST /jobs`,
  `GET /jobs/{id}`, stem/download endpoints, `DELETE /jobs/{id}`), the capped
  async worker pool, content-hash cache (dedup by `(hash, mode, tier)`), TTL
  background purge.
- **Phase 4** — Bandit speech/music/effects separator (PyTorch, vendored
  BandSplitRNN checkpoint), `full` mode's chained pipeline (Bandit → Demucs on
  the music stem), mode routing in `router.py`.
- **Frontend** — React + Vite + Tailwind app exists in `frontend/` (uploader,
  URL input, mode/tier picker, job-status polling, wavesurfer.js multitrack
  player with per-stem mute/solo, download bar) and is functional against the
  backend, but this work is **not yet committed to git** (shows as untracked
  in `git status`) and hasn't gone through the same review/test rigor as the
  backend phases.

Not yet built:

- **`best` tier (`htdemucs_ft`)** — deliberately unwired; `router.py` raises
  rather than silently falling back, per the "never default to `_ft`" rule.
  Exporting/quantizing the 4-model ensemble is future work.
- **Bandit ONNX export** — video/full modes currently run Bandit through
  PyTorch (the PRD's sanctioned interim path), which is heavier on CPU than
  the product target. The STFT-split technique from Phase 1 is expected to
  transfer, but hasn't been done.
- **Eval harness (Phase 6)** — `backend/eval/harness.py` is a stub
  (`raise NotImplementedError`). No measured per-tier SDR numbers or latency
  benchmarks exist yet; tier definitions (segment size, shifts) are
  reasoned defaults, not measurement-tuned.
- **Frontend test coverage** — no frontend automated tests; backend has
  `test_app`, `test_ingest`, `test_router`, `test_pool`, `test_jobs`,
  `test_ttl`, `test_chained_sep`, `test_onnx_vs_oracle`, all green, but there's
  no `test_router`/`test_ingest`-equivalent rigor on the client side yet.
- **Production deploy story** — this is a local `uv`/`npm` dev setup
  (no Docker, per the locked decision); there's no packaged deploy path.

Everything above marked "not yet built" is scoped in `docs/StemSep_PRD.md`
and `CLAUDE.md` §5 (build order) — nothing here is a silent gap, it's the
next items on the plan.
