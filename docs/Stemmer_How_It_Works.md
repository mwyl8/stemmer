# Stemmer — How It All Works

### A complete, plain-English explainer of every layer you've built

*For William. Covers Phases 0–5 as actually implemented. Each section starts with an intuition/analogy, then adds the technical detail underneath. If you can explain this document out loud, you can defend the whole project to anyone.*

---

## Part 0 — The one-paragraph version

You built a web service that takes a song or a video (uploaded, or pulled from a YouTube/TikTok/Instagram link) and **splits the audio into separate tracks** — the vocals on their own, the drums on their own, spoken dialogue on its own, and so on — then lets you play them back in a browser with a waveform for each, muting and soloing whichever you want. It does all of this **on a normal CPU, with no graphics card**, which is the hard part, and it does it by running a state-of-the-art AI model through a fast, stripped-down runtime you built by surgically splitting the model in half.

---

## Part 1 — The core idea: what "separation" even is

When a band records a song, every instrument is captured separately — that's a **multitrack**. Then it's **mixed down**: all those tracks are added together into one stereo file. That addition is lossy in the sense that it's *scrambled* — once summed, there's no simple arithmetic that recovers the originals. It's a smoothie: you can't un-blend it by subtraction.

**Source separation is un-blending the smoothie.** A neural network that has heard millions of examples of "here's the mix, and here's what the isolated drums actually were" learns the statistical fingerprint of each instrument, and can then estimate each one from a mix it's never heard.

Two important things follow:

1. **The output is an estimate, not a recovery.** There's no perfect answer hiding in the file. That's why you hear faint "bleed" — traces of other instruments — in a stem. Quality is measured, not guaranteed.
2. **"Instruments" and "speech" are different problems.** Splitting a *song* into vocals/drums/bass/other is **music source separation**. Splitting a *video's* audio into dialogue/music/sound-effects is **cinematic separation** — a different task with different models and different training data. Your app does both, which is why it has modes.

---

## Part 2 — Two ways to look at sound (you need this for everything else)

**View 1: the waveform.** Sound is air pressure wobbling over time. Digitally, that's a long list of numbers — 44,100 of them per second per channel ("44.1 kHz"). This is the raw, honest representation. It's great for sharp events (a snare hit) but it's hard to look at a wall of numbers and say "that's a bass guitar."

**View 2: the spectrogram.** Chop the audio into thousands of tiny overlapping slices (a few tens of milliseconds each). For each slice, ask: *how much of each pitch is present right now?* Stack those answers side by side and you get a **picture of the sound** — time running left to right, pitch running bottom to top, brightness = loudness. A bass line is a bright band along the bottom; a cymbal is a smear across the top.

The math that turns view 1 into view 2 is the **STFT** (Short-Time Fourier Transform). The reverse — picture back into audio — is the **ISTFT**. Both matter enormously to your project, because your big engineering win was about exactly this boundary.

**Why two views?** Because instruments are easier to *identify* in the picture (harmony, timbre, pitch) but easier to *reconstruct cleanly* from the wave (crisp transients, no phase artifacts). The best models use both. Yours does.

---

## Part 3 — The models, explained intuitively

### 3.1 Demucs / htdemucs — your music separator

**The intuition:** imagine someone who has listened to so much music that they can sit in a room with a full band playing and mentally write down each player's part separately. That's Demucs. You hand it a mixed song; it hands back four audio files: **vocals, drums, bass, other**.

**How it actually works:** it's a *hybrid* model, meaning it looks at both views of the sound at once.

- One branch reads the **raw waveform**.
- Another branch reads the **spectrogram**.
- In the middle, a **transformer** lets the two branches talk to each other, and lets every moment in the song look at every other moment.

That last bit — "attention" — is why it's good. When deciding whether a sound at second 47 is vocals, the model can consult second 12 and second 90 for context, rather than judging one instant in isolation. It's the same mechanism that powers language models, applied to audio.

It was trained on datasets where the true isolated stems were known, so it learned by being corrected millions of times. The version you're running (**htdemucs**) scores about **9 dB SDR** on the standard benchmark, which is roughly state-of-the-art.

**Why you didn't use `htdemucs_ft`:** that's a "fine-tuned" variant that's actually **four separate models** run one after another. Slightly better output, four times the CPU cost. You made it an opt-in "Best" tier rather than the default — the right call for a CPU-only product.

### 3.2 The ONNX conversion — your biggest engineering win

**The problem.** Demucs ships as a PyTorch model. PyTorch is a huge, heavy library — great for research, needlessly bulky for just *running* a trained model in production. **ONNX** is a portable format that describes a neural network in a framework-neutral way, and **ONNX Runtime** executes it much more efficiently, without needing PyTorch installed at all. For a CPU-only product, that's a meaningful speed and footprint win.

**Why it wasn't straightforward.** You tried to export Demucs directly and it failed — with both the legacy and the newer exporters. Two reasons:

- The STFT step uses **complex numbers** (each frequency has a magnitude *and* a phase, which mathematically pairs into a complex number), and the exporter can't represent those operations.
- Parts of the graph make decisions based on the *data itself* ("data-dependent asserts"), which a static exported graph can't express.

**Your solution — the split.** Instead of giving up, you **cut the model at the STFT boundary**:

- The **middle** of the model — the convolutional and transformer layers, all plain real-number math — is exportable. That became the ONNX graph.
- The **STFT and ISTFT at the edges** were moved *outside* the exported network and reimplemented in pure NumPy, matching PyTorch's versions to about one part in a million (~1e-6).

The analogy: the factory machine wouldn't fit through the door in one piece, so you detached the input and output conveyor belts, carried the core machine through, and rebuilt matching belts on the other side. Critically, **you verified the split reproduced the original forward pass exactly (0.0 difference) before exporting** — you proved the surgery was safe before committing to it.

The payoff is more than speed: because the STFT is NumPy, `demucs_onnx.py` **never imports PyTorch or demucs at all**. The heavy research libraries stay completely out of the production process — which is exactly what your own guardrail demanded.

### 3.3 Chunking and overlap-add

A five-minute song is too much audio to push through the network in one gulp — memory blows up. So you process it in **segments** and glue the results back together, overlapping the seams and cross-fading so there's no audible click at each join. Think of translating a long book page by page, with a few overlapping sentences so the transitions read naturally.

The subtlety you caught: Demucs has its own precise chunking behavior — it feeds the model **surrounding context** around short and final chunks rather than padding them with silence. A naive implementation that zero-pads produces slightly different (worse) output at the edges. You found that discrepancy **because your ONNX-vs-oracle diff test flagged it**, and fixed it. That's a real bug caught by a test rather than by luck — the strongest kind of evidence that your testing setup is sound.

### 3.4 Quantization (and why you correctly shelved it)

Neural networks normally store their numbers as **32-bit floating point**. **Quantization** squeezes them down to **8-bit integers** — roughly a quarter of the size, and on the right hardware, considerably faster. You did this: **166 MB → 56 MB**.

But you measured, and found two things:

- On this **transformer-heavy** model, naive dynamic int8 quantization **noticeably degraded output quality**.
- On **Apple Silicon**, it was actually **slower** — because the big int8 speedup comes from special instructions on Intel/x86 chips (VNNI) that ARM doesn't have in the same form.

So you kept the full-precision ONNX as the default path and left int8 as tunable future work. **This is the correct engineering decision, and it's also a great story**: you didn't assume the textbook optimization would help, you measured it, and you reported the honest result instead of shipping something worse.

### 3.5 The PyTorch "oracle"

You kept the original PyTorch Demucs as `_oracle_torch.py`, but it is **never used to serve a user request**. Its only job is to be the **answer key**: run the same clip through both the original model and your ONNX version, and confirm they agree (they do — max difference ~2e-4, i.e. inaudible).

This is a genuinely mature pattern. When you rewrite something for speed, you keep the slow-but-trusted version around as ground truth so you can *prove* the fast version didn't break anything.

### 3.6 Bandit — your speech separator

Different problem, different model. Where Demucs splits a *song* into instruments, **Bandit** splits a *video's audio* into **speech / music / effects**. This is "cinematic" separation — built for film, TV, podcasts, TikToks.

**How it works, intuitively:** it's a **band-split** model. It slices the spectrogram into horizontal **frequency bands** and gives each band its own small analyzer, then recombines their conclusions. The intuition: the cues that distinguish a human voice from a synth pad live in specific frequency ranges, so letting specialists examine each range beats one generalist squinting at the whole picture.

### 3.7 The chained "Full" mode — your showcase

This is the clever bit, and it's your own architectural idea rather than something a library hands you.

A TikTok clip typically has *someone talking over a music bed*. Neither model alone handles that well: Demucs has no concept of speech, and Bandit gives you a single lumped "music" track with no instrument breakdown.

So you **chain them**:

1. **Bandit** runs first → `speech`, `music`, `effects`.
2. The **`music`** output is then fed into **Demucs** → `vocals`, `drums`, `bass`, `other`.

The result from a single video: **spoken dialogue, the sung vocal, and the individual instruments, all separated.** Six stems from one clip.

**And you validated it elegantly.** On a pure-music file with no dialogue, Bandit correctly found no speech and passed the audio through nearly untouched — so Full mode's output matched plain Music mode to **3–4 decimal places**. That single test proves two things at once: the chain is wired correctly, *and* Bandit isn't hallucinating speech that isn't there. That's a really well-designed invariant.

---

## Part 4 — The architecture, layer by layer

Here's the journey of a request, and what every file does. The organizing principle throughout: **each layer does one job and knows as little as possible about the others.**

### 4.1 `app.py` — the front desk

**Analogy:** a hotel front desk. It receives every request, figures out who should handle it, and passes it along. It never carries luggage itself.

This is the FastAPI web server. It exposes the endpoints — submit a job, check status, list stems, download a stem, download a zip, delete a job — and does *nothing heavy*. Deliberately: if the web layer did the separating, the whole server would freeze for 44 seconds per song and every other user would time out. Keeping it thin is a hard rule in your guardrails.

### 4.2 `config.py` — the settings dial

**Analogy:** the control panel on a machine, with all the knobs in one place.

Sample rates, the quality **tiers** (Fast / Balanced / Best and which model each maps to), how many jobs may run at once (`POOL_SIZE`), how long files live before deletion (TTL), and the maximum file size and duration you'll accept. Putting these in one file means changing behavior is editing data, not hunting through code for scattered `if` statements.

### 4.3 `storage.py` — the filing cabinet

**Analogy:** a filing cabinet that remembers every order ever placed.

A **SQLite** database — which is just a single file on disk, no separate database server to install or run. It holds three things: **jobs** (what was requested, what stage it's at), **stems** (which output files belong to which job), and the **cache** index. The actual audio files live on disk; the database just remembers where they are and what they mean.

### 4.4 `jobs.py` — the order ticket

**Analogy:** the ticket that travels with your order through a kitchen, getting stamped at each station.

It defines what a job *is* and the stages it moves through: `queued → downloading → decoding → separating → encoding → done` (or `error`). The frontend's progress bar is literally reading this stage plus a percentage. When something fails, the error is recorded here rather than vanishing.

### 4.5 `pool.py` — the kitchen with a fixed number of cooks

**Analogy:** a restaurant that only has N cooks. Orders beyond that wait in line — they aren't thrown away, and the kitchen never collapses under a rush.

Separation is expensive. If ten people submit songs at once and you naively run all ten, the CPU thrashes and *everything* becomes slow — possibly to the point of crashing. The pool enforces a hard cap: at most `POOL_SIZE` separations run simultaneously, the rest queue. Predictable behavior under load, which is what "production-ready" actually means.

### 4.6 `runner.py` — the sealed workstation

**Analogy:** a separate soundproof booth. If something goes wrong in there, you close the door and it doesn't affect the rest of the building.

Each separation runs in its **own isolated subprocess**. Three concrete benefits:

- **Memory is clean.** When the process exits, every byte it used is returned — no slow leak in a long-running server.
- **It's killable.** If a job hangs, you terminate that process without touching the web server.
- **Isolation of dependencies.** The heavy model code lives in there, not in your API process.

The model is loaded **once** per worker, not per request, because loading weights is slow and would otherwise dominate your latency.

### 4.7 `ingest/fetch.py` — the errand runner

**Analogy:** an assistant you send out to fetch something, with strict instructions about where they're allowed to go and how long they can take.

This wraps **yt-dlp**, the open-source tool that can pull audio from YouTube, TikTok, Instagram and ~1,700 other sites. Your version adds the guardrails:

- **URL validation** — the link must match an allowed scheme and host.
- **No private or logged-in content** — you refuse to bypass authentication, deliberately.
- **Caps** on duration and file size, so nobody hands you a three-hour video.
- **Sandboxed subprocess with a timeout**, so a hung download can't wedge the service.
- Errors from yt-dlp are **surfaced honestly**, not silently retried around.

### 4.8 `ingest/decode.py` — the format converter

**Analogy:** a universal adapter. Whatever plug comes in, a standard plug comes out.

Wraps **ffmpeg** to convert *anything* — mp3, mp4, m4a, whatever yt-dlp returned — into one canonical format: **WAV, 44.1 kHz, stereo**. Everything downstream can then make one assumption instead of handling a dozen input formats. There's a second variant producing **16 kHz mono** for the speech path, because dialogue doesn't need high fidelity and the smaller data is much cheaper to process.

### 4.9 `separators/base.py` — the universal plug shape

**Analogy:** the shape of a wall socket. Any appliance with that plug works, and the wall doesn't care what the appliance does.

This is an **interface**: a contract saying every separator must offer a `separate(audio)` method returning a dictionary of `{stem_name: audio}`. Demucs, Bandit, and any future model all implement it.

**Why this matters more than it looks:** it means adding or swapping a model never requires touching the API, the job system, or the frontend. This is the same decoupling idea that made your provenance matcher index-agnostic, and it's the property that will let you add instrument separation next without rewriting anything.

### 4.10 `separators/router.py` — the dispatcher

**Analogy:** a dispatcher who reads the order and decides which specialist — or which *sequence* of specialists — handles it.

Given a **mode** (Music / Video / Full) and a **tier** (Fast / Balanced / Best), it picks the model(s) and the order to run them. The chained Full mode lives here: run Bandit, take its music output, run Demucs on it, merge the results into one stem set. Because this logic is concentrated in one file, `app.py` has zero knowledge of modes — it just passes the request through.

### 4.11 The separator implementations

- **`demucs_onnx.py`** — the production music separator: ONNX Runtime inference plus the segmenting/overlap-add logic. Imports no PyTorch.
- **`_demucs_core.py` / `scripts/export_onnx.py`** — the split-and-export machinery from Part 3.2.
- **`_stft_numpy.py`** — your hand-built STFT/ISTFT in pure NumPy, matching PyTorch to ~1e-6.
- **`scripts/quantize.py`** — the int8 shrink (166 MB → 56 MB), currently shelved on quality/speed grounds.
- **`_oracle_torch.py`** — the PyTorch answer key, eval-only.
- **`bandit_sep.py`** — the speech/music/effects separator.

### 4.12 `cache.py` — the "haven't we done this already?" memory

**Analogy:** a photocopy shop that keeps a copy of everything it's printed. Same document again? Here's the copy, instantly.

You compute a **content hash** of the decoded audio — a short fingerprint where identical audio always produces an identical string. If that same audio, mode, and tier has been separated before, you return the saved stems immediately instead of burning another 44 seconds of CPU. On a CPU-only product this is one of the highest-value features you have, and it costs almost nothing.

### 4.13 The TTL janitor

A background task that **deletes source audio and stems after a set time**, plus a `DELETE` endpoint for purging on demand. Two reasons: disk fills up fast with WAV files, and — more importantly — you should not be sitting on copies of other people's copyrighted audio indefinitely. This is a legal/ethical guardrail expressed as code.

### 4.14 `eval/harness.py` — the scorecard (still ahead of you)

The planned measuring instrument: run known material through each tier and compute **SDR** (signal-to-distortion ratio, in dB — higher means cleaner separation) against ground-truth stems. This requires audio where the true isolated tracks are known, which is why benchmark datasets like MUSDB18-HQ exist. Once this is built, your tier defaults and the int8 question become **measured facts instead of judgment calls**.

### 4.15 The frontend

**React + Vite + Tailwind**, with **wavesurfer.js** and its multitrack plugin doing the audio visualization.

- **Submit page** — drag-and-drop a file or paste a link, choose mode and tier.
- **Progress page** — polls the job endpoint and shows which stage it's in.
- **Result page** — the multitrack player: one synchronized transport, a waveform per stem, and mute/solo/volume per stem, plus downloads.

One deliberate detail: the player loads **mp3 previews**, not the WAVs. Browsers decode audio entirely in memory, and a handful of full-quality WAVs will exhaust that and crash the tab. WAVs are reserved for downloading.

---

## Part 5 — The full journey, end to end

**Someone pastes a TikTok link and picks "Full" mode:**

1. `app.py` receives `POST /jobs`, validates the input, creates a job row (`queued`), returns a job ID immediately — the user isn't left hanging on a connection.
2. `pool.py` sees a queued job and, if fewer than N are running, picks it up.
3. Stage `downloading`: `fetch.py` validates the URL, and yt-dlp pulls the audio in a sandboxed subprocess under a timeout, respecting size/duration caps.
4. Stage `decoding`: `decode.py` runs ffmpeg to produce canonical 44.1 kHz stereo WAV. The **content hash** is computed here — if it's a cache hit, everything below is skipped and stems return instantly.
5. Stage `separating`: `runner.py` spawns an isolated subprocess. `router.py` sees Full mode: **Bandit** splits speech/music/effects, then **Demucs** runs on the music bus. Inside Demucs, audio is chunked, each chunk goes NumPy-STFT → ONNX network → NumPy-ISTFT, and chunks are overlap-added back together.
6. Stage `encoding`: each stem is written as a WAV (for download) and an MP3 (for the player).
7. Job marked `done`; stems recorded in SQLite. The frontend, which has been polling, navigates to the player.
8. The user plays all six stems in sync, solos the vocal, mutes the speech, downloads the zip.
9. Later, the TTL janitor deletes the audio.

---

## Part 6 — What's real vs. what's still ahead

**Real and working:** ONNX-based Demucs separation on CPU (~44 s for a 4:55 song, about 0.15× real-time); the STFT-split export verified against the PyTorch oracle to ~2e-4; ingestion from file plus YouTube/TikTok/Instagram with validation and caps; the job system, API, capped worker pool, content-hash cache, and TTL purge; Music, Video, and chained Full modes; the React multitrack frontend. Roughly 50+ tests.

**Honestly not done yet:** int8 quantization is built but shelved (hurts quality, no ARM speedup); the SDR eval harness isn't built, so quality claims are currently qualitative; instrument-level separation (guitar/piano/strings) is the next phase; the UI is functional but bare; nothing is deployed or containerized.

Being able to state that second list clearly is not a weakness — it's precisely the honesty that made your provenance work land well.

---

## Part 7 — Mini-glossary

- **Waveform** — sound as a list of numbers over time (44,100 per second).
- **Spectrogram** — sound as a picture: time × pitch, brightness = loudness.
- **STFT / ISTFT** — the math converting waveform → spectrogram and back.
- **Stem** — one separated track (just the vocals, just the drums).
- **Transformer / attention** — the mechanism letting every moment consider every other moment for context.
- **ONNX / ONNX Runtime** — a portable format for neural networks, and a fast engine to run them without PyTorch.
- **Quantization** — storing weights as 8-bit integers instead of 32-bit floats: smaller, sometimes faster, sometimes worse.
- **Overlap-add** — processing long audio in chunks and cross-fading the seams.
- **Subprocess isolation** — running heavy work in its own process so it can be killed and its memory reclaimed.
- **Content hash** — a fingerprint of a file used to detect "we've seen this exact audio before."
- **TTL** — time-to-live; how long files exist before automatic deletion.
- **SDR** — signal-to-distortion ratio in dB; the standard score for separation quality (~9 dB is state-of-the-art).
- **Oracle** — a trusted reference implementation kept solely to verify a faster one.
- **Cinematic separation** — splitting audio into speech/music/effects, as opposed to splitting a song into instruments.

---

## Part 8 — The three things to say if someone asks "what did you build?"

1. **"I built a CPU-only stem separation service."** Upload a file or paste a link from YouTube/TikTok/Instagram, and it splits the audio into vocals, drums, bass, other instruments, and spoken dialogue, with a browser player to solo and inspect each stem.
2. **"The interesting engineering was getting a state-of-the-art model onto CPU."** Demucs wouldn't export to ONNX because of complex-valued STFT operations, so I split the model at the STFT boundary — exported the real-valued network, reimplemented STFT/ISTFT in NumPy matching to 1e-6 — verified it against the PyTorch original at 0.0 difference before export, and now the production path never imports PyTorch at all. 4:55 song in 44 seconds.
3. **"The architecture is built so models are swappable."** Every separator sits behind one interface, so routing between music, speech, and the chained pipeline is pure dispatch — and adding instrument separation next won't require touching the API, the job system, or the UI.
