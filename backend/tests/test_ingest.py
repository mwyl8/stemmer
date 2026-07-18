"""URL validation, caps, ffmpeg normalize. Real cases land in Phase 2.

Phase 0 smoke test: the ingest modules import cleanly and are unimplemented stubs.
"""

import pytest

from backend.ingest import decode, fetch


def test_fetch_raises_not_implemented_before_phase_2():
    with pytest.raises(NotImplementedError):
        fetch.fetch("https://example.com/video")


def test_decode_raises_not_implemented_before_phase_2():
    with pytest.raises(NotImplementedError):
        decode.decode("in.mp4", sample_rate=44100)
