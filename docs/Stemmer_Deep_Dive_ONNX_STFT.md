# Deep Dive — Complex Numbers, the STFT, and the ONNX Split

### Understanding the hardest (and most impressive) part of Stemmer

*Read this top to bottom. Each section is a prerequisite for the next. By the end you'll understand exactly why the export failed, exactly where you cut the model, and exactly why the verification numbers are what they are.*

---

## 1. The goal and the obstacle, stated plainly

**Goal:** run Demucs fast on a CPU, without PyTorch installed in the production process.

**The tool for that:** ONNX. You convert the trained network into a portable graph, and ONNX Runtime executes it efficiently.

**The obstacle:** Demucs wouldn't convert. The exporter failed on both the legacy and newer (dynamo) paths, for two distinct reasons — **complex numbers** and **data-dependent control flow**.

To understand why, you need to understand what a spectrogram actually *is* numerically. That's sections 2–4.

---

## 2. Why a frequency needs *two* numbers (complex numbers from scratch)

Forget the name "complex" — it's a historical accident and it scares people off. Here's the actual idea.

Suppose I tell you "there's a 440 Hz tone in this audio." That's incomplete. To reproduce that tone exactly, you need **two** facts:

1. **How loud is it?** → the **magnitude** (or amplitude).
2. **Where in its cycle does it start?** → the **phase**.

Two sine waves at the same frequency and same loudness but different phases are different signals. Add them together and — depending on the phase difference — they can double in volume, or **completely cancel to silence**. Phase is not a detail; it's half the information.

So every frequency, at every instant, needs a **pair of numbers**. You could store them as `(magnitude, phase)`. Mathematicians instead store them as a pair `(a, b)` written `a + bi`, and call it a **complex number**. The two forms are interchangeable:

```
magnitude = sqrt(a² + b²)
phase     = atan2(b, a)
```

**Why bother with the `a + bi` form?** Because the arithmetic is far nicer. When you add or multiply signals, complex multiplication automatically handles magnitude *and* phase correctly in one operation, whereas `(magnitude, phase)` pairs require awkward special-case math. The `i` is just bookkeeping that keeps the two components from getting mixed up.

**The practical takeaway for you:** a complex number in audio is simply **"a pair of numbers describing one frequency's loudness and timing."** Nothing mystical. And crucially — a complex number is *not* a single number to a computer. It's a distinct **data type**, and that's precisely what will break the ONNX export.

### Why phase actually matters (the intuition that makes it concrete)

If you keep only magnitudes and throw phase away, then try to rebuild the audio, you get a smeared, watery, robotic sound — the classic "phasey" artifact. Phase encodes *when* things line up: the sharp attack of a snare drum is many frequencies all starting in alignment at one instant. Destroy the phase relationships and the snare turns to mush.

Older separators dodged this by borrowing the **mixture's** phase for every separated stem — a known compromise that limits quality. Modern hybrid models like Demucs handle it more honestly, which is part of why they sound better.

---

## 3. What the STFT actually does, step by step

The **Short-Time Fourier Transform** turns a waveform into a spectrogram. Mechanically:

1. **Chop** the audio into overlapping frames. In htdemucs the frame is on the order of **4096 samples** (~93 ms at 44.1 kHz) and the frames advance by a **hop** of about **1024 samples** (~23 ms), so each frame overlaps the previous by 75%. *(Check `_demucs_core.py` for your exact values.)*
2. **Window** each frame — multiply it by a bell-shaped curve (a **Hann window**) that fades the edges to zero. Without this, chopping creates artificial discontinuities at the frame edges that smear energy across all frequencies ("spectral leakage").
3. **FFT** each windowed frame. The Fast Fourier Transform answers: "which frequencies are present in this 93 ms slice, and with what magnitude and phase?" Its output is **complex numbers** — one per frequency bin, each carrying that magnitude/phase pair.
4. **Stack** the frames side by side. You now have a 2-D grid: frequency (vertical) × time (horizontal), where each cell holds a complex number.

**Concretely, the shapes:**

```
input waveform    : (2 channels, N samples)              real numbers
after STFT        : (2 channels, ~2048 freq bins, T frames)   COMPLEX numbers
```

The **ISTFT** is the exact inverse: take each frame's complex values, inverse-FFT back to a little chunk of waveform, apply the window again, and **overlap-add** the chunks back into a continuous signal — dividing by the summed window energy so the overlapping doesn't inflate the volume.

**Why the reverse needs care:** because frames overlap by 75%, every output sample is the sum of ~4 windowed contributions. Reconstructing correctly requires the windows to sum to a constant (the "NOLA / COLA" condition) and the right normalization. Get the normalization wrong and your audio comes out quiet, loud, or with a periodic wobble.

---

## 4. The key insight: Demucs is *already* real-valued inside

This is the fact that makes your whole solution natural rather than arbitrary, so it deserves its own section.

Neural networks are built from convolutions and matrix multiplies over **real** numbers. They can't consume complex numbers directly. So what does Demucs do with its complex spectrogram?

**It splits the complex numbers into their two real components and treats them as extra channels.** The real parts become one set of channels, the imaginary parts another:

```
(2 ch, 2048 freq, T frames) complex
        ↓  view_as_real  →  split into (real, imag)
(4 ch, 2048 freq, T frames) REAL
```

From that moment on, **everything inside the network is ordinary real-number math.** The convolutions, the transformer, the decoders — all real. The complex numbers exist *only* at two points:

- **The entrance:** the STFT that creates the spectrogram.
- **The exit:** the ISTFT that turns the network's spectral output back into audio.

So Demucs isn't "a complex-valued network." It's a **real-valued network with complex-valued plumbing bolted to its front and back.**

That's the whole basis of your split: you didn't hack the model apart at a random place — **you cut at the only two points where complex numbers exist.** The core came out clean because the core was already clean.

### How the two branches fit together

Worth having the full picture, since it explains the ONNX graph's inputs and outputs:

- **Temporal branch:** reads the **raw waveform** directly through 1-D convolutions. Real numbers throughout — no STFT involved.
- **Spectral branch:** reads the **spectrogram** (real/imag as channels) through 2-D convolutions.
- **Cross-domain transformer** in the middle: self-attention within each branch, cross-attention between them, so the waveform view and the spectrogram view inform each other.
- **Two decoders:** one produces a **time-domain** output; the other produces a **spectral-domain** output that must be ISTFT'd back to audio.
- **Final stems = the time-domain output + the ISTFT of the spectral output**, summed.

So the model has *two* inputs conceptually (waveform, spectrogram) and *two* outputs (waveform output, spectral output), with complex↔real conversion at the spectral edges only.

---

## 5. What ONNX is, and how "export" actually works

**ONNX** is a standardized description of a neural network: a **graph** of operations (Conv, MatMul, Add, Softmax…) with fixed input/output shapes and data types. It's deliberately restrictive — a fixed menu of allowed operations (an **opset**) and allowed data types (float32, int64, and friends).

**Export works by tracing.** You feed the model a dummy input, PyTorch runs it, and records every operation performed into a static graph. Two consequences fall out of that word *static*:

1. **Every operation must have an ONNX equivalent.** If PyTorch performs an operation ONNX has no op for, tracing fails.
2. **The graph can't make runtime decisions.** Whatever branch the code took during tracing is the branch baked into the graph forever.

Both of these bit you.

---

## 6. Failure #1 — complex numbers aren't an ONNX data type

Mainstream ONNX and ONNX Runtime **have no complex tensor type**. There's no `complex64` you can pass between nodes, and no fully-supported complex arithmetic.

So when the tracer reached `torch.stft(...)` and saw it produce a **complex tensor**, there was simply nothing to write into the graph. There's an `STFT` op in newer opsets, but it's constrained and doesn't map cleanly onto how PyTorch's STFT behaves (padding modes, normalization, one-sided output), so the export either errors outright or produces something that doesn't match.

**In one line:** *ONNX can represent the network's math, but it cannot represent the complex-number plumbing at its edges.*

---

## 7. Failure #2 — data-dependent asserts

The second failure is subtler. Demucs' code contains checks and computations that depend on **the actual values flowing through it** — things like validating that a computed length matches an expected length, or deriving padding from the input size.

A **static** graph can't express "check this at runtime and react." The tracer has three bad options: bake in the value it happened to see (silently wrong for other inputs), fail loudly, or — with the newer dynamo exporter — report that it can't resolve a **data-dependent condition**. You saw the loud versions.

This is a general truth worth internalizing: **models with dynamic, input-dependent control flow are hard to export.** Pure feed-forward math exports fine; `if` statements about data don't.

---

## 8. The split — exactly where you put the knife

Given sections 4, 6, and 7, the solution becomes almost obvious in hindsight: **keep the real-valued core in ONNX; move the complex-valued edges out into ordinary Python/NumPy code that runs before and after.**

```
        ┌──────────── OUTSIDE the ONNX graph (NumPy) ────────────┐
input   │  waveform ──► STFT ──► complex spec ──► split re/im    │
audio ──┤                                                        │
        └────────────────────────┬───────────────────────────────┘
                                 ▼
        ┌──────────── INSIDE the ONNX graph (real math) ─────────┐
        │  spectral conv encoder ─┐                              │
        │                         ├─ cross-domain transformer ─┐ │
        │  temporal conv encoder ─┘                            │ │
        │                    ┌─ spectral decoder ──────────────┘ │
        │                    └─ temporal decoder                 │
        └───────┬──────────────────────────┬─────────────────────┘
                ▼                          ▼
        ┌──── OUTSIDE (NumPy) ────────────────────────────────────┐
        │  re/im ──► complex ──► ISTFT ──► waveform               │
        │                          + temporal output = STEMS      │
        └─────────────────────────────────────────────────────────┘
```

**Inside ONNX:** both conv encoders, the transformer, both decoders — all real-valued, all exportable, and this is 99% of the compute. This is what ONNX Runtime accelerates.

**Outside ONNX, in NumPy:** the STFT at the front, the ISTFT at the back, the real/imag packing and unpacking, and the chunking logic. Cheap operations, and NumPy handles complex numbers natively without complaint.

**The bonus payoff:** since the STFT is now NumPy rather than `torch.stft`, `demucs_onnx.py` has **no reason to import torch or demucs at all**. The heavyweight research libraries are entirely absent from the production process — which is exactly what your own guardrail demanded, achieved as a side effect of the split rather than as extra work.

---

## 9. Rebuilding the STFT in NumPy — what has to match *exactly*

This is where the split could have quietly gone wrong. Your NumPy STFT must be **bit-for-bit equivalent in behavior** to `torch.stft`, or the network receives subtly different inputs than it was trained on and the output degrades in ways that are hard to trace.

Everything below had to match:

- **Window function** — Hann, and the exact same length and periodicity convention.
- **`n_fft` and `hop_length`** — the frame size and step (≈4096 / 1024).
- **`center=True` behavior** — PyTorch pads the signal by `n_fft//2` on both ends so that frame *k* is *centered* on sample `k*hop`. Miss this and everything is time-shifted by ~46 ms.
- **Padding mode** — PyTorch uses **reflect** padding by default, not zeros. Different padding changes the first and last frames.
- **`onesided=True`** — for real input, the spectrum is symmetric, so only the lower half of the bins is kept (~2049 for n_fft=4096). Keeping all of them would double the channel count and break the model's expected shape.
- **Normalization** — whether a `1/sqrt(n_fft)` factor is applied.
- **For the ISTFT: the window-sum normalization** — dividing the overlap-added result by the summed squared window so the 75% overlap doesn't inflate amplitude.

You verified this matched to **~1e-6**. That number is the signature of "the algorithm is identical; the only difference is floating-point rounding." A conceptual mistake — wrong padding, wrong centering, missing normalization — would have shown up as an error many orders of magnitude larger. **1e-6 means it's right, not merely close.**

---

## 10. The verification ladder — why each number is what it is

You produced three different accuracy numbers, and they mean three different things. This is worth being able to explain precisely, because it demonstrates disciplined engineering.

**Rung 1 — `0.0` difference: the refactor is faithful.**
Before exporting anything, you restructured the PyTorch model into "core + external STFT" and checked that the restructured version produced *literally identical* output to the original. Exactly zero, because it's still the same PyTorch operations in the same order — you only moved where the boundary sits. This proves **the surgery itself introduced no change**, before any conversion risk enters the picture. Doing this check *first* is the disciplined move; skipping it means that if the final output is wrong, you can't tell whether the refactor or the export caused it.

**Rung 2 — `~1e-6`: the NumPy STFT matches PyTorch's.**
Same algorithm, different implementation. The residual is pure floating-point rounding — different libraries sum things in slightly different orders, and floating-point addition isn't perfectly associative. As covered above, this magnitude confirms correctness rather than approximation.

**Rung 3 — `~2e-4`: ONNX output matches the PyTorch oracle.**
Larger than rung 2, and that's expected, not alarming. ONNX Runtime uses its own optimized kernels for convolution and matrix multiplication — different loop orders, different vectorization, possibly fused operations. Each tiny rounding difference propagates through dozens of layers and accumulates.

Is 2e-4 acceptable? Audio samples live in the range −1 to +1. An error of 2e-4 is about **−74 dB** relative to full scale — far below the noise floor of any recording and utterly inaudible. For context, 16-bit CD audio has a quantization step around 3e-5, so you're within a few least-significant bits.

**Together these three numbers form a proof chain:** the refactor is exact → the STFT reimplementation is exact → the exported graph is numerically equivalent. Any one of them alone would be weak evidence; the ladder is strong.

---

## 11. What actually happens at runtime, per chunk

Putting it together, here's one chunk of audio flowing through the system:

1. **Take a chunk** of the song — roughly 7.8 seconds, matching the segment length htdemucs was trained on. Feeding it a length it never saw in training degrades quality.
2. **NumPy STFT** → complex spectrogram, shape ≈ `(2 ch, 2049 freq, T frames)`.
3. **Split real/imag into channels** → real array ≈ `(4, 2049, T)`.
4. **Run the ONNX graph** with two inputs — the raw waveform chunk and the real-valued spectrogram — producing two outputs per source: a time-domain output and a spectral-domain output, for each of the 4 stems.
5. **Recombine real/imag → complex**, then **NumPy ISTFT** each stem's spectral output back to a waveform.
6. **Sum** each stem's ISTFT'd spectral output with its time-domain output → that stem's audio for this chunk.
7. **Overlap-add** this chunk into the running output with a fade at the seams, then move on to the next chunk.

Repeat until the song is done, and you have four full-length stems.

---

## 12. The chunking bug you caught — and why it's the best detail in the project

Chunking sounds trivial. It isn't. The question is: **what do you feed the model at the very beginning and very end of a chunk, and at the final short chunk of the song?**

The naive answer is to **pad with zeros**. But Demucs doesn't do that — it supplies **real surrounding audio as context** around short and final chunks. Why it matters: a convolutional network's output near a boundary depends on what's beyond that boundary. Feed it silence and it "sees" a sudden cliff into nothing, which isn't what it was trained on, and the output near the edges degrades. Feed it the actual neighbouring audio and the network behaves as it did in training.

Your naive implementation zero-padded; Demucs used context; the outputs differed. **And you found it because the ONNX-vs-oracle diff test flagged a discrepancy**, not because someone happened to hear an artifact.

That's the part worth telling Jack. It's the difference between "I got a model running" and "I built a verification harness that caught a subtle correctness bug in my own reimplementation." The test wasn't ceremony — it did real work.

---

## 13. How to explain this in 60 seconds

> "Demucs is a hybrid model — it reads both the raw waveform and the spectrogram, and fuses them with a transformer. To run it fast on CPU I wanted it in ONNX, but the export failed: the spectrogram step produces **complex numbers**, which ONNX has no data type for, and the model has data-dependent asserts a static graph can't express.
>
> The insight was that Demucs is **already real-valued internally** — it converts the complex spectrogram into real/imaginary channels immediately, so complex numbers only exist at the STFT going in and the ISTFT coming out. So I cut the model at exactly those two boundaries: the real-valued core became the ONNX graph, and I reimplemented STFT/ISTFT in pure NumPy, matching PyTorch to 1e-6.
>
> I verified in stages — the refactor reproduced the original at **0.0** difference before I exported anything, and the final ONNX matches the PyTorch oracle to **2e-4**, about −74 dB, inaudible. As a bonus the production path never imports PyTorch at all. And the diff test earned its keep: it caught that my chunking zero-padded where Demucs feeds real surrounding context, which was a genuine correctness bug."

---

## 14. Glossary for this document

- **Complex number** — a pair of numbers `(a, b)` written `a + bi`, describing one frequency's **magnitude** and **phase** together.
- **Magnitude / phase** — how loud a frequency is, and where in its cycle it is.
- **FFT** — algorithm that decomposes a slice of audio into its frequency components (outputs complex numbers).
- **STFT / ISTFT** — running the FFT over overlapping windowed frames to get a spectrogram, and the inverse.
- **Hann window** — the bell curve applied to each frame to prevent edge discontinuities.
- **Hop length** — how far each frame advances; smaller hop = more overlap = more frames.
- **`center=True`** — pad the signal so frames are centered on their nominal positions.
- **Onesided** — keeping only half the frequency bins, valid because a real signal's spectrum is symmetric.
- **NOLA / window-sum normalization** — the correction that makes overlap-add reconstruct the original amplitude.
- **Opset** — the fixed menu of operations a given ONNX version supports.
- **Tracing** — running a model once and recording its operations to build a static graph.
- **Data-dependent control flow** — code whose branching depends on values only known at runtime; hostile to static graphs.
- **`view_as_real`** — PyTorch operation that reinterprets a complex tensor as a real tensor with an extra size-2 dimension.
- **Oracle** — the trusted reference implementation kept solely to verify a faster one.
- **dB relative to full scale** — 2e-4 on a −1…+1 signal ≈ −74 dB, well below audibility.
