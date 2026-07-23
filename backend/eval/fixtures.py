"""Synthetic Speech-vs-Singing eval fixture. No real speech/singing
recordings ship in this repo — CLAUDE.md forbids redistributing third-party
audio, and licensing a real clip with both a talking voice and a singing
voice for a test fixture isn't something to do casually. This procedurally
generates stand-ins for "a talking voice" (choppy, syllable-gated bursts —
speech's rhythmic cadence) and "a singing voice" (sustained, vibrato'd
tones — singing's held-note cadence), mixed with a simple instrument bed.

This is a smoke-level regression fixture, not a claim that synthetic
stand-ins predict real-world vocal separation quality — see
scripts/eval_singing_vs_speech.py's printed report for the honest version of
that caveat.
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
