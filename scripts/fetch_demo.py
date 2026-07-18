"""Manual Phase 2 smoke check: eyeball that ingestion works end-to-end
against a REAL URL (or local file). Not part of the automated test suite —
this hits the real network via yt-dlp.

    uv run python scripts/fetch_demo.py "https://www.youtube.com/watch?v=<id>"
    uv run python scripts/fetch_demo.py --file data/test.mp3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingest import IngestError, ingest
from backend.ingest.fetch import FetchError, InvalidURLError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("url", nargs="?", help="a public YouTube/TikTok/Instagram URL")
    source.add_argument("--file", help="a local audio/video file instead of a URL")
    args = parser.parse_args()

    try:
        if args.file:
            print(f"ingesting local file: {args.file}")
            result = ingest(file=args.file)
        else:
            print(f"fetching + ingesting: {args.url}")
            result = ingest(url=args.url)
    except (InvalidURLError, FetchError, IngestError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    info = sf.info(result.wav_path)
    print("\nok:")
    print(f"  wav_path      {result.wav_path}")
    print(f"  content_hash  {result.content_hash}")
    print(f"  duration      {info.duration:.1f}s")
    print(f"  samplerate    {info.samplerate} Hz")
    print(f"  channels      {info.channels}")


if __name__ == "__main__":
    main()
