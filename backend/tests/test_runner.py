"""runner.py's child-process entrypoint (`_main`), called directly in-process
(not via the real subprocess spawn — that would re-import backend.config
fresh from disk and lose this test's monkeypatched DB path). Covers the one
thing test_pool.py's mocking of `separate_in_subprocess` as a whole never
exercises: that `_main()` actually asks the separator for `runtime_info()`
and writes it onto the job row (backend/arch.py's whole point is surfaced
through exactly this wire).
"""

from __future__ import annotations

import sys

import numpy as np
import soundfile as sf

from backend import jobs, runner
from backend.config import MUSIC_SAMPLE_RATE
from backend.separators import router
from backend.separators.base import Separator


class FakeSeparator(Separator):
    def __init__(self, runtime_info_value):
        self._runtime_info_value = runtime_info_value

    def runtime_info(self):
        return self._runtime_info_value

    def separate(self, audio, on_chunk=None):
        if on_chunk is not None:
            on_chunk(1, 1)
        return {name: audio for name in ("vocals", "drums", "bass", "other")}


def _write_input_wav(path, seconds=1.0, sr=MUSIC_SAMPLE_RATE):
    data = (0.05 * np.random.default_rng(0).standard_normal((int(sr * seconds), 2))).astype(np.float32)
    sf.write(path, data, sr)


def test_main_writes_runtime_info_from_separator(db, tmp_path, monkeypatch):
    job_id = jobs.create_job("music", "balanced", "upload", "song.mp3")
    input_wav = tmp_path / "input.wav"
    _write_input_wav(input_wav)
    output_dir = tmp_path / "out"

    runtime_info = {"arch": "arm64", "provider": "CoreMLExecutionProvider", "model": "htdemucs_core"}
    monkeypatch.setattr(router, "select_separator", lambda mode, tier, stem_count=4: FakeSeparator(runtime_info))
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--input", str(input_wav), "--output-dir", str(output_dir), "--job-id", job_id],
    )

    runner._main()

    job = jobs.get_job(job_id)
    assert job.runtime_arch == "arm64"
    assert job.runtime_provider == "CoreMLExecutionProvider"
    assert job.runtime_model == "htdemucs_core"


def test_main_skips_runtime_info_write_when_separator_reports_none(db, tmp_path, monkeypatch):
    """Bandit-only (video mode) separators return None from runtime_info() —
    _main() must not blow up trying to unpack that into set_runtime_info."""
    job_id = jobs.create_job("video", "balanced", "upload", "song.mp3")
    input_wav = tmp_path / "input.wav"
    _write_input_wav(input_wav)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(router, "select_separator", lambda mode, tier, stem_count=4: FakeSeparator(None))
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--input", str(input_wav), "--output-dir", str(output_dir), "--job-id", job_id],
    )

    runner._main()

    job = jobs.get_job(job_id)
    assert job.runtime_arch is None
    assert job.runtime_provider is None
    assert job.runtime_model is None
