# PRD — Stemmer: CPU Stem Separation Service

**Owner:** William Wang · **Reviewer:** Jack Dempsey (Cipher Music) · **Status:** Draft v2 (build decisions locked)
**Companion docs:** `StemSep_Explainer.md` (background + model research), `StemSep_CLAUDE.md` (repo guardrails for Claude Code)

> Source of truth for the build. States the product, architecture, phased plan, tech choices with rationale, hard guardrails, and the open scope questions for Jack. **Build decisions (William's) are locked in §0.** Client-scope items still to confirm with Jack are marked **[CONFIRM WITH JACK]**.

---

## 0. Decisions locked for v1 (William)

| # | Decision | Choice |
|---|---|---|
| 1 | Modes | **Music + Video + Full chained** (Demucs stems · Bandit speech/music/fx · chained pipeline) |
| 2 | Stems | **4-stem music** (vocals/drums/bass/other) **+ speech/music/effects** |
| 3 | Frontend | **Multitrack player** (per-stem waveform, mute/solo/volume) **+ download stems/zip** |
| 4 | Persistence | **Ephemeral**, TTL auto-delete, no accounts |
| 5 | Job execution | **In-process async + isolated subprocess workers** (no Redis/Celery in v1) |
| 6 | Model runtime | **ONNX Runtime + int8 quantization from the start** (PyTorch only as a reference oracle for eval) |
| 7 | Packaging | **Local `uv`, no Docker** in v1 (optional Dockerfile later) |
| 8 | Stack | **Best-fit chosen per layer on merit** (not a copy of provenance; see §3) |
| 9 | Concurrency | **Small worker pool** — a few concurrent jobs, capped at configured `N` |
| 10 | Ingestion | **File upload + YouTube + TikTok + Instagram**, all in v1 |
| 11 | Testing | **Unit tests + a small SDR eval harness** |
| 12 | Output | **WAV (download) + MP3 (preview) + zip-all** |

---

## 1. Problem & goal

Cipher needs a tool that takes an **audio or video source** — an uploaded `mp3/wav/mp4` **or a link** (YouTube, TikTok, Instagram) — and **separates it into stems**: vocals, drums, bass, other instruments, and spoken **speech**. Users **play back and inspect** each stem in the browser (waveforms, mute/solo) and download them.

**Hard constraints (from Jack):** open-source AI/ML only; **CPU-only** inference — CPU optimization is a first-class requirement.

**Non-goals for v1:** real-time/streaming separation; mobile app; user accounts/billing; training our own models.

**Success =** a user submits a file or a public link, watches staged progress, and ends with a synced multitrack player (mute/solo/waveform per stem) plus downloadable stems — a 4-minute song separating on CPU within a measured, agreed target **[CONFIRM WITH JACK: target latency]**.

---

## 2. Key product decision: "vocals" ≠ "speech"

Music models (Demucs) output `vocals/drums/bass/other` and have **no speech stem**. Spoken dialogue lives in **cinematic** separation (Bandit-family: `speech/music/effects`). The app routes to the right model by **mode**:

- **Music mode** → Demucs → `vocals · drums · bass · other`.
- **Video/Podcast mode** → Bandit → `speech · music · effects`.
- **Full mode (chained, the showcase)** → Bandit first (`speech/music/effects`), then Demucs on the **music** bus → spoken dialogue **+** sung vocals **+** instruments.

Auto-mode detection ("is there speech?") is a **v2** nice-to-have (can reuse a YAMNet-style tagger).

---

## 3. Architecture

Stack chosen **per layer on merit** (decision 8). Several choices happen to match provenance.fm — noted where true, but each stands on its own rationale.

| Layer | Choice | Why (on merit) |
|---|---|---|
| Backend API | **FastAPI** | Async, Python-native (models are Python), great file/stream handling, typed. Best fit for an ML service. *(Also what provenance used.)* |
| Inference | **ONNX Runtime (int8-quantized)** | Fastest CPU path, drops the PyTorch dep at inference, threadable (OpenVINO/CoreML EPs). Chosen over PyTorch on merit for a CPU-only product. |
| Job execution | **In-process asyncio + subprocess pool** | No external broker to run; subprocess isolation gives clean peak memory + hard timeouts + killability. Right weight for a single-box v1. |
| Metadata store | **SQLite** | Zero-ops, file-based, perfect for single-box job/stem/cache metadata. Revisit only if we scale out. |
| Stem storage | **Local disk** (object store later) | Simplest durable store for large wavs on one box. |
| Frontend | **React + Vite + Tailwind** | Fast DX, and **wavesurfer.js** (the best OSS multitrack waveform player) is a first-class React fit. Chosen for the player requirement. |
| Ingestion | **yt-dlp + ffmpeg** | The de-facto OSS tools; yt-dlp covers all target platforms, ffmpeg normalizes everything. |

```
stemmer/
  backend/
    app.py                FastAPI: endpoints only; wires modules, no heavy logic
    config.py             tiers, segment size, sample rates, thread counts, pool size N, TTLs
    storage.py            SQLite: jobs, stems, cache (metadata); stems on disk
    jobs.py               job lifecycle + status/progress model
    pool.py               async worker pool (cap = N); dispatches to subprocess runners
    runner.py             the isolated-subprocess entrypoint that runs one separation
    ingest/
      fetch.py            yt-dlp wrapper (link -> audio); URL validation, caps, sandbox, timeout
      decode.py           ffmpeg normalize -> WAV 44.1k stereo (or 16k mono for speech path)
    separators/
      base.py             Separator interface: separate(audio) -> {stem: waveform}
      demucs_onnx.py      htdemucs via ONNX Runtime (quantized); music stems
      bandit_sep.py       speech / music / effects (cinematic)
      router.py           mode + tier -> pick/chain separators
      _oracle_torch.py    PyTorch Demucs, REFERENCE ONLY (eval harness ground truth; not the product path)
    cache.py              content-hash dedup (skip re-separating identical audio)
    eval/
      harness.py          small labeled set -> SDR per model/tier (measured, not guessed)
    tests/
      test_router.py      routing + tier selection
      test_ingest.py      URL validation, caps, ffmpeg normalize
  frontend/               React + Vite + Tailwind
    src/
      pages/              Home (submit), Job (progress), Result (player)
      components/         Uploader, LinkInput, ModeTierPicker, ProgressStages,
                          StemPlayer (wavesurfer multitrack), StemRow (mute/solo/vol), DownloadBar
      api.js
  models/                 downloaded/quantized ONNX weights (gitignored; pulled + quantized on setup)
  README.md · Makefile · pyproject.toml (uv) · CLAUDE.md
```

**Design principles (kept because they're good, not because they're provenance's):**
- `app.py` is a **thin switchboard** — HTTP/routing only.
- **Separators sit behind one interface** (`base.py`) so swapping/adding a model never touches the API or UI.
- **Heavy compute runs in an isolated subprocess** — clean memory, killable, hard timeout.
- **Config-driven tiers** — quality/speed is data in `config.py`, not scattered branches.

---

## 4. API (v1)

- `POST /jobs` — file upload **or** `{url, mode, tier}`. Returns `{job_id}`. Validates size/duration/URL; dedups via content hash.
- `GET /jobs/{id}` — `{status, stage, progress, stems?}`, `stage ∈ {queued, downloading, decoding, separating, encoding, done, error}`.
- `GET /jobs/{id}/stems` — list of `{name, url, format, duration}`.
- `GET /jobs/{id}/stems/{name}` — stream/download a stem (mp3 preview or wav).
- `GET /jobs/{id}/download` — zip of all stems.
- `DELETE /jobs/{id}` — purge source + stems now (also auto-purged on TTL).

Progress = worker updates the job row; frontend polls `GET /jobs/{id}` (SSE/websocket is a v2 upgrade).

---

## 5. Model & CPU strategy (ONNX-first — decision 6)

- **Music separator:** Demucs **htdemucs** exported to **ONNX**, **int8-quantized**. **Best** tier = `htdemucs_ft` (4×, opt-in). *(No `htdemucs_6s` in v1 — 4-stem locked in decision 2.)*
- **Speech separator:** Bandit-family (`speech/music/effects`), ONNX where an export exists; otherwise PyTorch behind the same interface with a note to export in a follow-up.
- **Product inference path is ONNX Runtime (quantized) from day one.** PyTorch Demucs exists only as `_oracle_torch.py` — the ground-truth reference the eval harness diffs against; it is **never** the path a user request takes.
- **CPU optimizations built in from the start (not a later phase):**
  1. **ONNX Runtime** inference (≈1.3× faster than PyTorch on CPU, ~identical output).
  2. **int8 quantization** (up to ~80% inference-time cut) — applied at setup, checked into the eval harness for quality delta.
  3. **Stem-specific sub-models** where the request only needs some stems (vocals-only ≈ ¼ cost).
  4. **Segment/chunk** long audio with overlap-add → bounded memory + streamable progress.
  5. **Tiers** (`Fast / Balanced / Best`) map to model + `shifts` + segment size.
  6. **Threaded runtime** (intra-op threads = physical cores; OpenVINO EP on Intel, CoreML EP on Mac).
  7. **Content-hash cache**; **load model once per worker**; **44.1k for music, 16k mono for speech**.

Tiers live in `config.py`, e.g.:
```
TIERS = {
  "fast":     {"music": "htdemucs_onnx_q", "shifts": 0, "segment": 7},   # quantized, single pass
  "balanced": {"music": "htdemucs_onnx",   "shifts": 0, "segment": 7},   # fp, single model
  "best":     {"music": "htdemucs_ft",     "shifts": 1, "segment": 7},   # opt-in, 4x
}
POOL_SIZE = N   # max concurrent separations (decision 9)
```

---

## 6. Frontend requirements (v1)

- **Submit**: drag-drop file **or** paste link; pick **mode** (Music/Video/Full) and **tier** (Fast/Balanced/Best).
- **Progress**: staged bar (download → decode → separate → encode) with % where available.
- **Result — multitrack player** (`wavesurfer.js` + multitrack plugin): one synced transport; per-stem **waveform**, **mute / solo / volume**, seek, loop region.
- **Karaoke toggle** (one-click mute vocal/speech stem).
- **Download**: per-stem (**wav** + **mp3**) and **"download all" zip**.
- Serve **mp3 previews** to the player, keep **wav** for download (avoid in-browser memory blowups on long clips).

**v2 frontend (out of scope now):** remix/export, A/B compare, spectrogram, session history, auto-mode badge.

---

## 7. Build plan (phased — how Claude Code should sequence it)

**Phase 0 — Skeleton & env.** Repo layout, `uv` env, `pyproject.toml`, FastAPI hello, SQLite schema, `config.py`, `CLAUDE.md`. Install/verify `ffmpeg`, `yt-dlp`. No Docker.

**Phase 1 — Core separation, ONNX from the start.** Obtain/quantize an **htdemucs ONNX** model; wire it behind `base.Separator` (`demucs_onnx.py`); prove `separate(wav) -> stems` end-to-end from a local file, run in the **subprocess runner**. Bring in `_oracle_torch.py` and a tiny diff check so we know the ONNX/quant output matches. `test_router` green.

**Phase 2 — Ingestion (all four sources).** `ingest/fetch.py` (yt-dlp: YouTube/TikTok/Instagram + file passthrough, URL validation + size/duration caps + sandbox + timeout) and `ingest/decode.py` (ffmpeg → 44.1k stereo wav).

**Phase 3 — Jobs + API + worker pool.** `POST /jobs`, `GET /jobs/{id}`, stems endpoints; `pool.py` runs up to **N** concurrent jobs, each in an isolated subprocess, with staged progress; content-hash cache; TTL purge.

**Phase 4 — Speech mode + chained Full mode.** Add **Bandit** separator + the **chained** pipeline (Bandit → Demucs on the music bus). Router handles mode selection.

**Phase 5 — Frontend.** Submit page, staged progress, and the **wavesurfer multitrack** player with mute/solo/volume + downloads (wav/mp3/zip).

**Phase 6 — Eval harness + tuning.** Small labeled set + SDR script comparing tiers and the quantized vs oracle output; record per-tier latency on the reference CPU. Tune tier defaults from measurements.

Phases 1–3 are the first working slice (file/link → stems via API). CPU optimization is **not** a separate late phase — it's baked into Phase 1.

---

## 8. Things NOT to do (guardrails — read before coding)

*(Style Jack liked in the provenance CLAUDE.md. Also in `StemSep_CLAUDE.md`.)*

- **Do not run separation synchronously in the request handler.** Queue it; run in the subprocess pool, or the server blocks and times out.
- **Do not load models inside the request or per-job.** Load once per worker and reuse — cold-loading weights dominates latency.
- **Do not make PyTorch the product inference path.** ONNX Runtime (quantized) serves requests; `_oracle_torch.py` is reference-only for the eval harness.
- **Do not default to `htdemucs_ft`.** It's 4 models (4× CPU time) — the opt-in "Best" tier only; default is single-model `htdemucs`.
- **Do not let the worker pool grow unbounded.** Cap concurrent separations at `POOL_SIZE = N`; queue the rest.
- **Do not hold whole long files in memory.** Segment/stream; serve mp3 previews — large in-browser Web Audio decodes crash wavesurfer.
- **Do not downsample the music path below 44.1 kHz.** Only the speech-only Bandit path may use 16 kHz mono.
- **Do not fetch private/authenticated content or bypass logins/paywalls.** Public URLs only; surface `yt-dlp` errors instead of scraping around them. Process only content the user is authorized to use.
- **Do not persist source or stem audio indefinitely.** Auto-delete on TTL and on `DELETE`. Never redistribute fetched third-party audio beyond the user's session.
- **Do not trust user links blindly.** Validate/normalize URLs; cap duration and file size; run `yt-dlp`/`ffmpeg` in a sandboxed subprocess with a timeout.
- **Do not use `pickle` for cross-process state.** Use safe serialization; keep state in the DB/files.
- **Do not skip the content-hash cache.** Re-separating identical audio on CPU wastes minutes.
- **Do not commit model weights to git.** Gitignore `models/`; pull + quantize on setup; pin versions.
- **Do not `brew install python`** on this machine (brew compiles from source). Use **`uv`**; Node is already installed via tarball at `~/.local/node`.
- **Do not rebase shared branches.** Squash-merge; feature branches stay linear.

---

## 9. Open questions for Jack (client scope — not build prefs)

**Stems & modes**
1. Is **"speech"** specifically **spoken dialogue** (podcast/interview/voiceover) as distinct from **sung vocals**? Confirms the Bandit path is needed (we've assumed yes).
2. Any near-term need for **6-stem** (guitar/piano)? We locked 4-stem for v1 but the model supports it.

**Performance & deployment**
3. Acceptable **wait** for a 4-minute song on CPU (e.g., < 90 s)? Need a fast **preview** before the full result?
4. Reference **hardware**: cores/RAM per box? What should **`N` (concurrent jobs)** be, and expected peak load?
5. Max **audio length** and **file size**?

**Ingestion & legal**
6. Any need for **private/authenticated** content (would need cookies — currently a guardrail we avoid)?
7. **Legal/ToS posture** — internal tooling on authorized content only, or user-facing? Sets retention/watermarking/ToS behavior.

**Output & product**
8. For v1 we do **download stems** (no remix/export) — confirm that's enough, or is level-adjust/export needed sooner?
9. Ephemeral with TTL is locked — what **TTL** (e.g., 24h)? Any need for a saved library later?
10. Does this **plug into existing Cipher infra** (auth, storage, a queue) or stand alone? Any **model/vendor constraints** (offline-only, no cloud APIs, licensing)?

**Quality bar**
11. Target **quality** — an SDR threshold or a golden set to validate against, or is "sounds clean" the v1 bar?

---

## 10. Acceptance criteria (v1 "done")

- Submit a **file** and a **public link** (YouTube/TikTok/Instagram); both produce stems.
- **Music** and **Video** modes both work; **Full chained** mode produces speech + sung vocals + instruments.
- Runs **CPU-only via quantized ONNX**; a 4-min song completes within the agreed target on the reference machine, numbers recorded per tier.
- **Worker pool** runs up to **N** concurrent jobs without blocking the API.
- **Multitrack player** with per-stem waveform + **mute/solo/volume**; **download** per-stem (wav+mp3) and zip.
- **Content-hash cache** returns instantly on a repeat submission.
- Source/stems **auto-purge** on TTL; no private-content fetching.
- `test_router` + `test_ingest` green; **eval harness** reports per-tier SDR + latency.
