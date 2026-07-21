"""Server-side waveform peaks (PRD Addendum §2.5): precompute a downsampled
amplitude series per stem so the browser player never has to fetch+decode a
full-resolution audio file just to draw a waveform. wavesurfer.js accepts a
`peaks` array per channel and treats it exactly like decoded PCM data
(Decoder.createBuffer) — it re-buckets whatever resolution it's given to fit
the current zoom level, so handing it an already-downsampled series (instead
of the true sample-accurate one) produces the same-looking waveform at a
fraction of the data, and — combined with the `duration` wavesurfer derives
itself — lets it skip the full blob fetch + AudioContext decode entirely and
just point the underlying <audio> element at the mp3 preview for streaming
playback (see wavesurfer.js's loadAudio: it only fetches+decodes when no
channelData is supplied).

Each bucket keeps its min AND max sample (not just one extreme) so the
waveform stays visually symmetric — a plain magnitude-only series would
render as a one-sided (all-top) waveform instead of the usual mirrored one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_NUM_BUCKETS = 4000


def compute_peaks(wav_path: Path, num_buckets: int = DEFAULT_NUM_BUCKETS) -> list[list[float]]:
    """Returns one list per channel, each `2 * num_buckets` floats long:
    [min0, max0, min1, max1, ...] in time order — fed straight to
    wavesurfer.js as its `peaks` option."""
    audio, _sr = sf.read(wav_path, dtype="float32", always_2d=True)  # (samples, channels)
    n, channels = audio.shape
    if n == 0:
        return [[] for _ in range(channels)]

    num_buckets = max(1, min(num_buckets, n))
    edges = np.unique(np.linspace(0, n, num_buckets + 1).astype(np.int64))
    if edges.size < 2:
        edges = np.array([0, n])
    starts = edges[:-1]

    peaks = []
    for ch in range(channels):
        col = audio[:, ch]
        maxes = np.maximum.reduceat(col, starts)
        mins = np.minimum.reduceat(col, starts)
        interleaved = np.empty(starts.size * 2, dtype=np.float32)
        interleaved[0::2] = mins
        interleaved[1::2] = maxes
        peaks.append(interleaved.tolist())
    return peaks
