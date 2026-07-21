# PRD Addendum — Phase 6 (Instrument Separation) & Phase 7 (UI Overhaul)

**Owner:** William Wang · **Status:** Draft, pending Jack's answers (§3)
**Extends:** `StemSep_PRD.md`. Phases 0–5 are complete (skeleton → ONNX separation → ingestion → jobs/API/pool → speech + chained modes → frontend).

---

## 1. Phase 6 — Separation by instrument

**Goal:** go beyond `vocals/drums/bass/other` to individual instruments — guitar, piano, strings, brass, reeds, organ — including (best-effort) orchestral material. Explicitly *not* required to be perfect.

### 1.1 Model options, ranked

**Tier 1 — quick win: `htdemucs_6s`.**
Demucs' 6-stem model adds **guitar** and **piano** to the existing four. This is nearly free for us: same architecture, same ONNX export path (`_demucs_core.py` / `export_onnx.py`), same runner. Do this first — it's a config + export change, not new engineering. Limitation: only two extra instruments, pop/rock-oriented.

**Tier 2 — the right answer: Banquet (query-based).**
`Banquet` is a **stem-agnostic, query-based single-decoder** separator: instead of a fixed set of output heads, you give it a *query* naming the instrument you want, and one decoder extracts it. Why it's the standout choice for us:
- **We already have the integration.** It comes from `kwatcharasupat/source-separation-landing` — the same repo we used for Bandit in Phase 4. Same loading patterns, same checkpoints ecosystem.
- **It's tiny: ~24.9M trainable parameters.** That's an order of magnitude lighter than htdemucs — exactly what a CPU-only product needs.
- **It beats htdemucs_6s on guitar and piano** on MoisesDB while approaching 6-stem HTDemucs on the standard vocals/drums/bass/other stems.
- **It reaches narrow classes** — clean acoustic guitar, **reeds**, **organ** — i.e. real instrument granularity, not just "other."
- It pairs with a **PaSST** instrument-recognition model to detect which instruments are present, which we can use to auto-populate the available stems for a track rather than making the user guess.

**Tier 3 — stretch, maximum flexibility: AudioSep (language-queried).**
`AudioSep` is an open-domain separation foundation model driven by **natural-language queries** ("separate the solo violin"). Strong zero-shot generalization; it was the DCASE 2024 Task 9 baseline. Trade-off: broader coverage but lower musical fidelity than a dedicated MSS model, and heavier. Good as an **experimental "describe what you want" mode**, not the default path.

### 1.2 Honest expectation-setting on orchestral

Symphony/classical separation is **an open research problem**, not a solved one. The field is still building datasets for it (SynthSOD, and the 2025 *Spheres* multitrack orchestral dataset), and there's published work specifically on *"Source Separation of Small Classical Ensembles: Challenges and Opportunities."* Dense orchestral textures — many correlated instruments sharing harmonics, recorded in one room with heavy bleed — are much harder than a pop mix with isolated multitrack sources.

**So: pop/rock instrument separation (guitar, piano, organ, reeds) will work well via Banquet. A full symphony will be rough.** Say this to Jack up front rather than over-promising — it's the same "real vs. aspirational" honesty that landed well with provenance.

### 1.3 Proposed build

1. Add **`htdemucs_6s`** through the existing ONNX export path → guitar + piano stems. Ship this first.
2. Add **`separators/banquet_sep.py`** implementing the `Separator` interface with a **query parameter** (instrument name). Reuse the Bandit integration work.
3. Add **instrument detection** (PaSST) so the UI can show "this track contains: guitar, piano, organ" and offer those as extractable stems.
4. Extend `router.py` with an **Instruments mode**: detect → query Banquet per detected instrument → return an N-stem result. Keep chaining compatible (Bandit → music bus → instruments).
5. **[Stretch]** `separators/audiosep_sep.py` behind a "describe a sound" text-query mode.
6. **Light eval**: measure per-instrument quality on a couple of known multitrack songs so tier/model claims are measured, not asserted (folds Phase 6-eval work in here).

**Guardrail additions:** don't let instrument mode fan out unbounded — cap the number of instrument queries per job (each is a separate model pass, and this is CPU-only); reuse the content-hash cache keyed on `(hash, mode, tier, query)`.

---

## 2. Phase 7 — UI overhaul

Current UI is functional but bare. Target: something that feels like a real audio tool.

### 2.1 Transport & playback (the core gap)
- **Play / pause**, **skip ±10s and ±30s**, jump to start/end, next/previous marker.
- **Click-to-seek** on any waveform; a **timeline ruler** with time markers; current-time / total-duration readout.
- **Keyboard shortcuts** — space (play/pause), ← → (skip), `M` (mute focused stem), `S` (solo focused stem), `0-9` (jump to % of track).
- **Playback speed** (0.5×–2×) and **A/B loop region** for repeat-listening a passage.

### 2.2 Per-stem controls & viewing
- **Color-coded waveform per stem**, with a consistent palette (vocals = one hue, drums = another…).
- **Volume slider + pan** per stem; **mute / solo** with exclusive-solo behavior.
- **Live level meters (VU)** so you can *see* which stem is active.
- **Waveform zoom** (in/out) plus a **minimap** for navigating long tracks.
- **Spectrogram toggle** per stem — the most convincing way to *show* separation quality.
- Collapse/expand, reorder, and rename stems.

### 2.3 Mixing & comparison
- **Master volume**; mute-all / reset-mix.
- **A/B against the original** — one toggle to hear the source vs. the recombined stems.
- **Save a mix preset**; **export a custom mix** with your level adjustments (this was v2 in the original PRD — now worth pulling in).

### 2.4 Progress & timing feedback

The single most-requested missing piece. Separation takes tens of seconds on CPU, and right now the user has no idea how long is left.

- **Live elapsed timer** from submission, always visible.
- **True progress bar during separation** — derived from **chunks completed / total chunks**. Because the pipeline already splits audio into fixed ~7.8 s chunks, the exact fraction complete is known; this must not be a fake/indeterminate spinner.
- **ETA before work even starts** — from the measured real-time factor per tier (~0.15× real-time on the reference machine → a 4:55 song ≈ 45 s), then refined live as chunks land. *This is another reason to build the eval/benchmark harness: it produces the RTF numbers that make the ETA honest.*
- **Per-stage timings** (download / decode / separate / encode) so a slow network is visibly distinct from slow inference.
- **Final "took 44 s" summary** on the result page — useful for the user and for your own tuning.
- **Cache-hit state** clearly labelled ("returned instantly from cache") rather than flashing a 0→100% bar.
- **Chained Full mode** runs two separation passes (Bandit → Demucs); progress must be weighted across both rather than resetting to 0% midway.

*Implementation note:* the subprocess runner must report chunk completion back to the job row (per-chunk DB update or a pipe/callback), and `jobs.py` needs fields for `chunks_done`, `chunks_total`, `stage_timings`, `elapsed_seconds`, `eta_seconds`.

### 2.5 Workflow & polish
- **Recent jobs list** with thumbnails/metadata; **re-run this audio** at a different mode/tier without re-uploading.
- **Job metadata panel** — model version, tier, mode, duration, processing time (pairs with the receipt idea in §3).
- Per-stem and zip downloads (existing), plus a **shareable result link**.
- **Dark theme + a real design system**; loading skeletons; empty/error states.
- **Performance:** precompute waveform **peaks server-side** for long files so the browser never decodes full audio; keep serving mp3 previews to the player.
- Responsive layout; basic accessibility (focus states, ARIA on transport controls).

---

## 3. Questions for Jack (before refining further)

*Grouped; the architecture ones deliberately follow the threads he responded to in provenance.fm.*

**Architecture & engineering standards**
1. **Plugin-grade modularity?** Right now separators sit behind one interface with an explicit router. Do you want adding a model to be a true **drop-in plugin** (registry + config entry, zero changes elsewhere), or is the explicit router the right level of abstraction at this stage?
2. **Reproducibility manifest / receipt?** Should every job emit a manifest recording **model version, mode, tier, parameters, content hash, and timings** — so any output is reproducible and auditable later? (This is the provenance receipt instinct applied here; cheap to add, and it makes results defensible.)
3. **Living guardrails doc?** I've kept a `CLAUDE.md` with an explicit "Things NOT to do" section for this repo. Do you want that maintained as a standing artifact (and is there a Cipher house style I should fold into it)?

**Deployment & scale**
4. **Where does this run** — my machine, a Cipher box, containerized? That decides whether I invest in Docker + a real queue (Redis/Celery) or keep the current in-process capped pool.
5. **Concurrency and hardware target?** Cores/RAM, and how many simultaneous jobs — so I size `POOL_SIZE` and set tier defaults from **measurements** rather than guesses.
6. **Is CPU-only permanent or a v1 constraint?** It changes where I spend effort — more ONNX/quantization tuning (int8 currently hurts quality on this transformer and gets no ARM speedup) versus keeping a GPU path open.

**Product direction**
7. **Who's the user** — internal Cipher team, artists, labels? Determines API-first vs. polished UI investment.
8. **What's driving instrument separation** — sampling/remix work, catalog analysis, rights/provenance, A&R listening? The use case determines which instruments actually matter.
9. **Fixed taxonomy or arbitrary query?** Is a fixed set (guitar/piano/strings/brass/reeds/organ) enough, or do you want **free-text queries** ("solo violin")? This is a real architectural fork — the query-based model (Banquet) vs. a language-queried one (AudioSep).
10. **How central is orchestral/classical?** It's the hardest case — current models and datasets are weak there, and I'd rather set expectations than over-invest. Pop/rock instrument separation will be notably better.
11. **Quality bar** — is there a golden set or target SDR you want me to hit, or is "sounds right to an A&R ear" the standard for now?

**Scope & logistics**
12. **Retention/legal posture** — internal tooling on authorized content only, or user-facing? Sets TTL, storage, and whether we keep anything at all.
13. **Integrate with existing Cipher infra** (auth, storage, queue), or stay standalone?
14. **What does "done" look like for this intro task** — a working demo I walk you through, or something you'd actually deploy?

---

## 4. Suggested sequencing

1. **Ask Jack §3** (especially 8–10, which shape Phase 6 directly).
2. **Phase 6a:** `htdemucs_6s` → guitar + piano (quick win, ships in hours).
3. **Phase 7:** UI overhaul (biggest perceived-quality jump; makes everything demoable).
4. **Phase 6b:** Banquet query-based instrument separation + PaSST detection.
5. **Phase 6c [stretch]:** AudioSep free-text mode, if Jack wants arbitrary queries.
6. **Eval harness** throughout — measure per-instrument quality so claims stay honest.

---

## Sources
- Banquet — stem-agnostic single-decoder, query-based MSS: https://arxiv.org/abs/2406.18747 · repo: https://github.com/kwatcharasupat/source-separation-landing
- AudioSep — language-queried open-domain separation: https://arxiv.org/pdf/2501.15177 (survey context) · DCASE 2024 Task 9 baseline
- Orchestral datasets / difficulty: SynthSOD https://arxiv.org/pdf/2409.10995 · Spheres https://arxiv.org/html/2511.21247v1 · Small classical ensembles https://arxiv.org/pdf/2505.17823
