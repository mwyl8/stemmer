"""Synthetic eval fixtures for Speech-vs-Singing and Lead-vs-Backing-Vocals
modes. No real speech/singing/harmony recordings ship in this repo —
CLAUDE.md forbids redistributing third-party audio, and licensing a real
clip with the specific vocal arrangement each eval needs isn't something to
do casually. These procedurally generate stand-ins with the structural
property each eval needs (a talking-vs-singing cadence difference; a
lead-only section vs. a stacked-harmony section), mixed with a simple
instrument bed.

These are smoke-level regression fixtures, not a claim that synthetic
stand-ins predict real-world separation quality — see
scripts/eval_singing_vs_speech.py's and scripts/eval_lead_vs_backing.py's
printed reports for the honest version of that caveat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SingingFixture:
    sample_rate: int
    mixture: np.ndarray  # (2, samples) — what the pipeline actually sees
    spoken_speech: np.ndarray  # (2, samples) — ground truth "talking" track
    sung_vocals: np.ndarray  # (2, samples) — ground truth "singing" track
    instruments: np.ndarray  # (2, samples) — ground truth instrument bed


@dataclass(frozen=True)
class LeadBackingFixture:
    sample_rate: int
    lead_only_seconds: float  # [0, lead_only_seconds) has no backing vocals at all
    mixture: np.ndarray  # (2, samples)
    lead_vocal: np.ndarray  # (2, samples) — ground truth lead melody, present throughout
    backing_vocals: np.ndarray  # (2, samples) — ground truth harmony stack, silent until lead_only_seconds
    instruments: np.ndarray  # (2, samples)


def make_speech_vs_singing_fixture(duration: float = 6.0, sr: int = 44100, seed: int = 0) -> SingingFixture:
    """Both voices overlap for the full clip — the harder, more honest case
    for measuring bleed than cutting between separate speech-only and
    singing-only sections would be."""
    rng = np.random.default_rng(seed)
    speech_mono = _formant_speech(duration, sr, rng)
    singing_mono = _sung_melody(duration, sr, notes_hz=(196.0, 220.0, 246.9, 220.0))  # G3 A3 B3 A3
    instruments_mono = _instrument_bed(duration, sr, rng)

    spoken_speech = _to_stereo(speech_mono, gain=0.5)
    sung_vocals = _to_stereo(singing_mono, gain=0.5)
    instruments = _to_stereo(instruments_mono, gain=0.35)
    mixture = spoken_speech + sung_vocals + instruments

    return SingingFixture(
        sample_rate=sr,
        mixture=mixture,
        spoken_speech=spoken_speech,
        sung_vocals=sung_vocals,
        instruments=instruments,
    )


def make_lead_vs_backing_fixture(
    lead_only_seconds: float = 6.0, harmony_seconds: float = 6.0, sr: int = 44100, seed: int = 0
) -> LeadBackingFixture:
    """First `lead_only_seconds`: lead melody alone, no backing vocals at
    all — a clean baseline. Next `harmony_seconds`: the same lead continues,
    joined by a stacked 3-voice harmony a third and a fifth above it (a
    "big harmony chorus") — the section that should show up in
    `backing_vocals`, not bleed into `lead_vocal`."""
    rng = np.random.default_rng(seed)
    duration = lead_only_seconds + harmony_seconds
    lead_notes = (196.0, 220.0, 246.9, 220.0, 196.0, 220.0)  # G3 A3 B3 A3 G3 A3, repeats to fill duration

    lead_mono = _sung_melody(duration, sr, notes_hz=lead_notes)
    harmony_mono = _harmony_stack(duration, sr, notes_hz=lead_notes, intervals=(1.25, 1.5))  # major third, perfect fifth
    instruments_mono = _instrument_bed(duration, sr, rng)

    # Gate the harmony stack to silence during the lead-only section.
    gate = np.ones(len(harmony_mono), dtype=np.float32)
    gate[: int(lead_only_seconds * sr)] = 0.0
    harmony_mono = harmony_mono * gate

    lead_vocal = _to_stereo(lead_mono, gain=0.5)
    backing_vocals = _to_stereo(harmony_mono, gain=0.5)
    instruments = _to_stereo(instruments_mono, gain=0.35)
    mixture = lead_vocal + backing_vocals + instruments

    return LeadBackingFixture(
        sample_rate=sr,
        lead_only_seconds=lead_only_seconds,
        mixture=mixture,
        lead_vocal=lead_vocal,
        backing_vocals=backing_vocals,
        instruments=instruments,
    )


def _harmony_stack(duration: float, sr: int, notes_hz: tuple[float, ...], intervals: tuple[float, ...]) -> np.ndarray:
    """Sum of the same melody transposed up by each ratio in `intervals` —
    e.g. 1.25 (~major third) and 1.5 (perfect fifth) stacked on the lead
    line produces the "big harmony chorus" this fixture needs."""
    voices = [_sung_melody(duration, sr, notes_hz=tuple(f * ratio for f in notes_hz)) for ratio in intervals]
    return (np.sum(voices, axis=0) / len(voices)).astype(np.float32)


def _to_stereo(mono: np.ndarray, gain: float) -> np.ndarray:
    normalized = gain * mono / (np.max(np.abs(mono)) + 1e-8)
    return np.stack([normalized, normalized], axis=0).astype(np.float32)


def _formant_speech(duration: float, sr: int, rng: np.random.Generator, f0: float = 140.0) -> np.ndarray:
    """A fundamental plus a couple of formant-ish harmonics, gated into
    syllable-length bursts (~4/sec) with silence between — the choppy
    envelope that (unlike singing's sustained notes) is the main thing
    naively distinguishing spoken cadence from sung cadence."""
    t = np.arange(int(duration * sr)) / sr
    voiced = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * f0 * 2.7 * t) + 0.25 * np.sin(2 * np.pi * f0 * 4.1 * t)
    syllable_rate = 4.0
    gate = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * syllable_rate * t))
    jitter = 0.6 + 0.4 * rng.random(len(t))
    return (voiced * gate * jitter).astype(np.float32)


def _sung_melody(duration: float, sr: int, notes_hz: tuple[float, ...]) -> np.ndarray:
    """A sequence of sustained tones with vibrato — the smooth, held-note
    envelope that (again, naively) distinguishes singing from speech's
    choppy bursts."""
    t = np.arange(int(duration * sr)) / sr
    n_notes = len(notes_hz)
    note_len = len(t) // n_notes
    out = np.zeros(len(t), dtype=np.float64)
    for i, f0 in enumerate(notes_hz):
        start = i * note_len
        end = len(t) if i == n_notes - 1 else (i + 1) * note_len
        seg_t = t[start:end] - t[start]
        vibrato_hz = f0 * (1 + 0.01 * np.sin(2 * np.pi * 5.5 * seg_t))
        phase = 2 * np.pi * np.cumsum(vibrato_hz) / sr
        tone = np.sin(phase) + 0.3 * np.sin(2 * phase)
        fade_samples = min(len(seg_t) // 4, int(0.05 * sr))
        fade = np.ones(len(seg_t))
        if fade_samples > 0:
            ramp = np.linspace(0, 1, fade_samples)
            fade[:fade_samples] = ramp
            fade[-fade_samples:] = ramp[::-1]
        out[start:end] = tone * fade
    return out.astype(np.float32)


def _instrument_bed(duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A slow chord plus soft noise, low-energy relative to the two voices
    so it doesn't dominate the mix."""
    t = np.arange(int(duration * sr)) / sr
    chord = sum(np.sin(2 * np.pi * f * t) for f in (110.0, 138.6, 164.8)) / 3.0
    texture = 0.15 * (rng.random(len(t)) - 0.5)
    return (0.4 * chord + texture).astype(np.float32)
