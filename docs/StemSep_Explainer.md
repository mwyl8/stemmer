# Stem Separation — Everything You Need to Know (Explainer)

*For William, before building Cipher's intro task. This is the "learn it well enough to defend it to Jack" doc. The companion `StemSep_PRD.md` is the build spec for Claude Code; `StemSep_CLAUDE.md` is the repo guardrails file.*

---

## 0. The task in one paragraph

Build a CPU-only web app that takes an **audio/video file** (mp3/wav/mp4) **or a link** (YouTube, TikTok, Instagram, etc.), pulls the audio, and **separates it into stems** — vocals, drums, bass, other instruments, and spoken **speech** — then lets you **play back and view** each stem in the browser (mute/solo, waveforms) and download them. The hard constraints Jack set: **open-source AI/ML libraries** and **optimize for CPU (no GPU)**.

The single most important design realization: **"vocals" and "speech" are two different separation problems**, and the best systems for each are different. Getting that distinction right is what will make your solution look thought-through rather than "I ran Demucs."

---

## 1. What "source separation" actually is (intuition first)

You have a **mixture** — one waveform where everything is summed together (voice + guitar + drums all overlapping in time and frequency). Separation means recovering the individual **sources** that were added together. It's the "cocktail party problem": un-mixing a smoothie back into fruit.

Two families of the problem matter for us:

- **Music Source Separation (MSS).** Split a *song* into **vocals / drums / bass / other** (the classic 4 stems), or 6 stems (adds **guitar** and **piano**). Benchmarked on the **MUSDB18-HQ** dataset, scored in **SDR** (signal-to-distortion ratio, in dB — higher is cleaner; ~9 dB is state of the art).
- **Cinematic / general Audio Source Separation (CASS).** Split a *video's* audio into **speech (dialogue) / music / effects (SFX)**. This is the world of film, podcasts, TikToks. Benchmarked on the **DnR ("Divide and Remaster")** dataset. This is where **"speech"** separation lives — MSS models don't have a speech stem.

Because Jack wants instruments **and** vocals **and** speech, you almost certainly want **both kinds of model**, not one. More on that in §4.

---

## 2. How the algorithms work (enough to explain on a whiteboard)

There are three architectural approaches. All modern top models are variations/combinations of these.

**(a) Spectrogram masking (frequency domain).** Take the mixture, run an **STFT** to get a spectrogram (energy at each time × frequency). A neural net predicts a **soft mask** for each source — a number between 0 and 1 for every time-frequency cell saying "how much of this cell belongs to vocals?" Multiply mask × mixture spectrogram → the source's spectrogram; inverse-STFT back to audio (usually reusing the mixture's phase). Intuition: *highlight the pixels of the spectrogram that belong to each instrument.* Used by **Spleeter**, **Open-Unmix**, and (with fancier nets) **MDX-Net**.

**(b) Waveform / time-domain.** Skip the spectrogram; a learned encoder/decoder operates directly on the raw samples (e.g., Conv-TasNet, early Demucs). Intuition: *learn your own "spectrogram" end-to-end instead of using a fixed STFT.* Better at transients (drum hits), can struggle with tonal bleed.

**(c) Hybrid + Transformers (the current best).** Run **both** a waveform branch and a spectrogram branch and fuse them with a **cross-domain transformer** (self-attention within each domain, cross-attention across them). This is **Demucs v4 / htdemucs** — it gets the best of both and ~9 dB SDR. **Band-split** models (Bandit, BS-RoFormer) are a related idea: chop the spectrogram into frequency **sub-bands**, process each with its own RNN/transformer, recombine — currently the strongest approach for vocals and for cinematic speech/music/effects.

One line to remember: *"Older models mask a spectrogram; the modern ones (Demucs, band-split transformers) work in both time and frequency and use attention — that's why they're cleaner."*

---

## 3. The open-source model landscape (what to actually use)

| Model | Stems | Quality | CPU speed | Framework | Use it for |
|---|---|---|---|---|---|
| **Demucs v4 — htdemucs** | 4 (voc/drums/bass/other) | Excellent (~9 dB SDR) | ~1.5× real-time (slow but OK) | PyTorch (`pip install demucs`; also in torchaudio) | **Default music separator** |
| **htdemucs_ft** | 4 | Best | **4× slower** (it's 4 models) | PyTorch | Opt-in "best quality" tier only |
| **htdemucs_6s** | 6 (+guitar, +piano) | Very good | ~same as htdemucs | PyTorch | When user wants guitar/piano |
| **Spleeter** (Deezer) | 2/4/5 | Lower (bass bleed) | **Fast** | TensorFlow (no updates since 2019) | Optional **fast preview** tier |
| **Open-Unmix (umxl)** | 4 | Good | Light | PyTorch | Lightweight fallback |
| **MDX-Net / BS-RoFormer / UVR models** | vocals/instrumental (+more) | Top-tier vocals | Medium (ONNX available) | ONNX/PyTorch (Ultimate Vocal Remover) | Best **vocal isolation / karaoke** |
| **Bandit v2 / Banquet** | **speech / music / effects** (+singing-voice in newer work) | SOTA cinematic | Medium | PyTorch (`kwatcharasupat/source-separation-landing`) | **The speech/dialogue separator** |

**Recommendation:** standardize on **Demucs (htdemucs)** for music and a **Bandit-family** model for speech/music/effects, behind a common interface so either can be swapped. Keep **Spleeter** as an optional fast-preview tier and an **MDX/UVR vocals** model as an optional high-quality karaoke path.

**The elegant "full pipeline" (worth pitching to Jack):** for a TikTok/IG video that has *both* dialogue and a music bed, **chain** them — run **Bandit first** to pull `speech / music / effects`, then feed the **music** bus into **Demucs** to break it into `drums / bass / vocals / other`. That gives you spoken dialogue **and** the sung vocal **and** the instruments, cleanly, and it directly handles the known-hard case of *singing voice vs. speech* (there's active 2024–2025 research on exactly this). This routed/chained design is the "architecture Jack will like."

---

## 4. Routing: how one app serves music, video, and podcasts

Don't force one model onto every input. Expose a small number of **modes**, and/or auto-route:

- **Music mode** → Demucs 4-stem (or 6-stem) → drums/bass/vocals/other(/guitar/piano).
- **Video / Podcast mode** → Bandit → speech/music/effects.
- **Full mode (chained)** → Bandit → then Demucs on the music stem → speech + dialogue-free instruments + sung vocals.
- **Vocals-only / karaoke** → MDX/UVR vocals model (or Demucs, keep `other+drums+bass` as the instrumental).

**Auto-routing (nice-to-have):** a cheap audio classifier (e.g., YAMNet — which you already used in the homomorphic paper) can tag "is there speech here?" and pick the mode automatically. That's a clean reuse of something you know, and a good v2 feature.

---

## 5. CPU optimization — the part Jack cares most about

Running these on CPU is the whole challenge. Levers, roughly in order of impact:

1. **Model choice is the biggest knob.** `htdemucs` (1 model) vs `htdemucs_ft` (4 models = 4× the time). Default to the single model; make `_ft` an explicit "best quality, slower" toggle. Offer **quality tiers** (Fast = Spleeter/single-pass, Balanced = htdemucs, Best = htdemucs_ft).
2. **ONNX Runtime instead of PyTorch.** Exported **HT-Demucs → ONNX** runs **~1.3× faster on CPU** than PyTorch and is numerically ~identical (verified exports exist; e.g. `demucs.onnx` and StemSplit's htdemucs-ft ONNX; Mixxx did a 2025 GSoC converting Demucs v4 to ONNX). ONNX also drops the heavy PyTorch dependency at inference time.
3. **Quantization (int8).** Post-training quantization can cut inference time by **up to ~80%** with little quality loss. Biggest single speedup after ONNX.
4. **Stem-specific sub-models.** If the user only wants vocals, a **vocals-only** ONNX model costs ~**¼** of the full 4-stem bag. Ship per-stem models and only run what's requested.
5. **Segment / chunk the audio.** Process in overlapping windows (Demucs `--segment`) with overlap-add. Lowers peak memory, prevents OOM on long files, and lets you **stream progress** to the UI.
6. **Kill test-time augmentation on the fast path.** `--shifts 0` (no random-shift averaging), no `_ft`. Those are quality-vs-time trades you expose as tiers.
7. **Thread the runtime.** Set ONNX Runtime intra-op threads = physical cores; on Intel CPUs use the **OpenVINO** execution provider, on Mac use **CoreML**. Big free wins.
8. **Cache by content hash.** Re-separating the same 4-minute song is minutes of CPU wasted — hash the decoded audio and cache stems. (Same dedup instinct as your provenance `item_hash`.)
9. **Sample-rate discipline.** Keep **44.1 kHz** for music (downsampling destroys the high-frequency detail separation needs). For a **speech-only** Bandit pass you can go **16 kHz mono** — much cheaper and fine for dialogue.
10. **Load the model once.** Cold-loading weights per request dominates latency. Load into a persistent worker and reuse.

Realistic expectation to set with Jack: a 4-minute song with `htdemucs` on a decent multi-core CPU is on the order of a couple minutes; ONNX + quantization + segmenting brings that down meaningfully and, crucially, keeps memory bounded and the UI responsive.

---

## 6. Ingestion (files + links)

- **`yt-dlp`** is the ingestion workhorse — it supports **1,700+ sites** including **YouTube, TikTok, Instagram (posts/Reels/Stories/IGTV), Facebook, X, SoundCloud**, and is actively maintained. `-x` extracts audio; it shells out to **ffmpeg** to convert/merge.
- **`ffmpeg`** normalizes everything to a canonical format before separation: decode mp4/mp3/whatever → **WAV, 44.1 kHz, stereo** (or 16 kHz mono for the speech path).
- **Caveats to design around:** public content usually needs no login; **private/authed** content (some IG stories) needs cookies — don't build around bypassing logins. Respect platform ToS and **only process content the user is authorized to use** (this is a guardrail, see the PRD). Cap **duration and file size**, validate/normalize URLs, and run yt-dlp in a **sandboxed subprocess** with a timeout.

---

## 7. Frontend features (brainstorm)

Jack explicitly asked for **listen-back / view** of stems. Core MVP + extensions:

**MVP**
- Drag-drop upload (mp3/wav/mp4) **or** paste a link.
- Mode/quality selector (Music / Video / Full · Fast / Balanced / Best).
- Live **progress** with per-stage status: *download → decode → separate → encode*.
- **Multitrack stem player** — one synced transport, a **waveform per stem** (use **wavesurfer.js** + its **multitrack** plugin), **mute / solo / volume** per stem, seek, loop-region.
- **Download** each stem (wav/mp3) + **"download all" (zip)**.

**High-value extensions**
- **Karaoke / instrumental toggle** (mute the vocal or speech stem in one click).
- **Remix/export**: adjust per-stem levels and export a new mix.
- **A/B compare** original vs. recombined stems.
- **Spectrogram view** toggle (nice for showing separation quality).
- **Session history** / shareable result link.
- **Auto-detected mode** badge ("we detected speech + music").

**Frontend caveat:** wavesurfer decodes audio in-browser via Web Audio; **very long clips can exhaust browser memory**. Serve compressed stem previews (mp3) for the player and keep the lossless wav for download; consider server-side pre-computed waveform peaks for long files.

---

## 8. Recommended architecture (deliberately echoing provenance.fm)

This maps almost one-to-one onto the provenance shape Jack liked — same bones, new domain:

- **Backend: FastAPI** (like provenance). Endpoints: `POST /jobs` (file upload **or** link), `GET /jobs/{id}` (status + progress), `GET /jobs/{id}/stems` (list + URLs), static/streamed stem files.
- **Async job queue + workers.** Separation is a long CPU job, so it **must not run in the request handler**. A worker pulls jobs, runs separation in an **isolated subprocess** (this is exactly your benchmark-subprocess pattern from the paper — clean memory, no leaks, killable on timeout), writes stems to disk, updates status.
- **Storage: SQLite** for job/stem metadata (like provenance's SQLite), stems on disk (or object store later). **Content-hash dedup cache** so repeats are instant.
- **`separators/` package with one interface.** `Separator.separate(audio) -> {stem_name: waveform}`, with `demucs.py`, `bandit.py`, `spleeter.py`, `mdx.py` implementations behind it, plus a `router.py`. Swapping a model never touches the API or the UI — the same "index-agnostic matcher" decoupling you pitched for provenance.
- **`ingest/` module.** `yt-dlp` + `ffmpeg` wrappers, URL validation, size/duration caps, sandboxed subprocess.
- **Config-driven tiers/thresholds** (like provenance's `config.py`): model per tier, segment size, sample rate, thread count.
- **Frontend: React + Vite + Tailwind** (like provenance) + **wavesurfer.js** multitrack.

Why this reads well to Jack: it's the *same disciplined layering* he already praised — a thin API switchboard, swappable model modules behind a stable interface, subprocess isolation for the heavy compute, SQLite + content-hash caching, and config-driven quality tiers. You're transferring a proven architecture, not improvising.

---

## 9. Risks & honest unknowns (say these out loud)

- **CPU latency is the core risk.** Full-quality Demucs on CPU is minutes per song; the mitigation stack is ONNX + quantization + segmenting + tiers + caching. Set expectations and measure.
- **Speech vs. singing voice is genuinely hard** — chaining Bandit → Demucs helps but isn't perfect; there's live 2024–25 research on it. Don't overclaim clean dialogue extraction from a dense musical mix.
- **Link ingestion is legally/ToS-sensitive.** Position this as a tool operating on **authorized** content; don't build features that bypass auth/paywalls; auto-delete fetched source audio on a TTL.
- **No golden eval yet.** Like provenance, the missing piece is a **validation harness** — a small labeled set to measure SDR per model/tier so tier choices are measured, not guessed. Building that early is the same "measure, don't assert" habit.

---

## Sources
- Demucs / htdemucs (Meta): https://github.com/facebookresearch/demucs · https://pypi.org/project/demucs/
- HT-Demucs v4 production guide: https://tomodahinata.com/en/blog/demucs-v4-music-source-separation-production-guide
- Demucs → ONNX (CPU) exports: https://github.com/sevagh/demucs.onnx · https://stemsplit.io/blog/htdemucs-ft-onnx-export · https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/
- Demucs vs Spleeter (quality/speed): https://stemsplit.io/blog/spleeter-vs-demucs · https://beatstorapon.com/blog/demucs-vs-spleeter-the-ultimate-guide/
- Cinematic separation / Bandit (speech·music·effects): https://mvsep.com/algorithms/45 · https://arxiv.org/abs/2408.03588 · https://github.com/kwatcharasupat/source-separation-landing
- Real-time / low-latency MSS: https://arxiv.org/pdf/2511.13146
- yt-dlp (supported sites, audio extract): https://github.com/yt-dlp/yt-dlp/
- wavesurfer.js multitrack player: https://github.com/katspaugh/wavesurfer.js · https://github.com/katspaugh/wavesurfer-multitrack
