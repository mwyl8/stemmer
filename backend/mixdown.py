"""Server-side mixdown for "export a custom mix" (PRD Addendum §2.3): applies
the same volume/pan/mute/solo adjustments a user made in the player, but
against the full-quality WAV stems rather than the mp3 previews the browser
plays — downloads stay WAV-only, mirroring the existing per-stem/zip
downloads (app.py never hands out mixed audio derived from the lossy
preview).

The live player does this with a per-track Web Audio graph
(useMultitrackPlayer.js: source -> StereoPannerNode -> analyser). This module
reproduces the same effective-volume/exclusive-solo rules directly against
numpy arrays instead, since there's no browser here to host that graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from backend.separators.limiter import apply_peak_safety


@dataclass
class TrackAdjustment:
    name: str
    volume: float = 1.0
    pan: float = 0.0  # -1 (left) .. 1 (right)
    muted: bool = False
    solo: bool = False


@dataclass
class MixdownResult:
    audio: np.ndarray  # (channels, samples) float32
    sample_rate: int


def _balance_gains(pan: float) -> tuple[float, float]:
    """Equal-power stereo balance control — not a mono panner. pan=-1 keeps
    only the left channel, pan=1 keeps only the right, pan=0 is unity gain
    on both. Matches the perceptual shape of the player's StereoPannerNode
    closely enough for an offline export; it isn't a bit-exact reproduction
    of the Web Audio spec's stereo-input pan matrix."""
    pan = max(-1.0, min(1.0, pan))
    angle = (pan + 1.0) * (math.pi / 4.0)
    return math.cos(angle) * math.sqrt(2.0), math.sin(angle) * math.sqrt(2.0)


def mix_stems(
    stem_paths: dict[str, str],
    adjustments: list[TrackAdjustment],
    master_volume: float = 1.0,
    master_muted: bool = False,
) -> MixdownResult:
    """`stem_paths` maps stem name -> wav file path. Stems with no matching
    adjustment play at unity (volume=1, pan=0, unmuted) — the same default a
    freshly-loaded player row starts at."""
    by_name = {a.name: a for a in adjustments}
    any_solo = any(a.solo for a in adjustments)

    mixed: np.ndarray | None = None
    sample_rate: int | None = None
    for name, path in stem_paths.items():
        adj = by_name.get(name) or TrackAdjustment(name=name)
        effective_volume = adj.volume if (not any_solo or adj.solo) else 0.0
        if adj.muted or master_muted:
            effective_volume = 0.0
        effective_volume *= master_volume
        if effective_volume == 0.0:
            continue

        wav, file_sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
        wav = wav.T  # (channels, samples)
        if sample_rate is None:
            sample_rate = file_sr

        gain_l, gain_r = _balance_gains(adj.pan)
        if wav.shape[0] >= 2:
            wav = np.stack([wav[0] * gain_l, wav[1] * gain_r], axis=0)
        wav = wav * effective_volume

        if mixed is None:
            mixed = wav.copy()
            continue
        # Stems can differ by a handful of samples in length (per-model
        # padding) — pad the shorter one with silence rather than
        # truncating either, so nothing gets clipped off the end.
        if wav.shape[1] != mixed.shape[1]:
            target = max(wav.shape[1], mixed.shape[1])
            mixed = np.pad(mixed, ((0, 0), (0, target - mixed.shape[1])))
            wav = np.pad(wav, ((0, 0), (0, target - wav.shape[1])))
        mixed += wav

    if mixed is None or sample_rate is None:
        raise ValueError("no audible stems in this mix — everything is muted or lost to solo")

    safe_mixed, _peak, _limited = apply_peak_safety(mixed)
    return MixdownResult(audio=safe_mixed, sample_rate=sample_rate)
