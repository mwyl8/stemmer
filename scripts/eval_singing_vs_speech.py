"""Speech-vs-Singing mode eval: how cleanly does a talking voice land in
spoken_speech and a singing voice land in sung_vocals, on a synthetic
fixture containing both simultaneously (backend/eval/fixtures.py)?

No real speech+singing recording ships in this repo (licensing — CLAUDE.md
forbids redistributing third-party audio), so the fixture is procedurally
generated stand-ins, not real vocals. Read this as a smoke-level regression
check on this pipeline's *routing* (does Bandit's speech stem end up in
spoken_speech, does Demucs's vocals stem end up in sung_vocals, and how much
does each leak into the other), not a benchmark of real-world separation
quality — real speech and singing have different spectra than these
stand-ins, and results will differ.

Singing-voice-vs-speech-voice is a known-hard, largely unsolved MSS problem:
neither Bandit nor Demucs was ever trained on "is this frame spoken or
sung", only on "what source type/instrument role is this" — a belted or
half-spoken line, or dialogue over a musical bed, can and will bleed across
both stems. The SIR numbers below are the honest measure of that; don't
round them up.

    uv run --group speech python scripts/eval_singing_vs_speech.py
    uv run --group speech python scripts/eval_singing_vs_speech.py --tier fast --duration 10

Needs the `speech` extras (torch/torchaudio/pytorch-lightning/spafe) and the
downloaded Bandit checkpoint (scripts/fetch_bandit_weights.py) — same
requirement as video/full mode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.eval.fixtures import make_speech_vs_singing_fixture
from backend.eval.metrics import sdr_sir_sar
from backend.separators.router import select_separator
from backend.separators.singing_sep import INSTRUMENTS_STEM_NAME, SPOKEN_STEM_NAME, SUNG_STEM_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--duration", type=float, default=6.0, help="fixture length in seconds")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fixture = make_speech_vs_singing_fixture(duration=args.duration, sr=44100, seed=args.seed)

    try:
        separator = select_separator("singing", args.tier)
    except NotImplementedError as exc:
        print(f"FAILED to build the singing-mode separator: {exc}", file=sys.stderr)
        print(
            "(Bandit needs the `speech` extras + a downloaded checkpoint — "
            "see scripts/fetch_bandit_weights.py)",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    stems = separator.separate(fixture.mixture)

    # Each stem's "own" ground truth vs. the one known interferer — the
    # other voice. That's the pair sdr_sir_sar needs to say how much of the
    # *other voice specifically* (not just generic noise) leaked in.
    voice_pairs = {
        SPOKEN_STEM_NAME: (fixture.spoken_speech, fixture.sung_vocals, "sung vocal"),
        SUNG_STEM_NAME: (fixture.sung_vocals, fixture.spoken_speech, "spoken voice"),
    }

    print(f"Speech-vs-Singing eval — synthetic fixture, tier={args.tier!r}, duration={args.duration}s, seed={args.seed}\n")
    print(f"{'stem':<16} {'SDR (dB)':<10} {'SIR (dB)':<10} {'SAR (dB)':<10}  meaning")
    print("-" * 76)
    for name, (target, interferer, other_label) in voice_pairs.items():
        metrics = sdr_sir_sar(stems[name], target, interferer)
        print(
            f"{name:<16} {metrics['sdr_db']:<10.2f} {metrics['sir_db']:<10.2f} {metrics['sar_db']:<10.2f}  "
            f"SIR = how much {other_label} bled in"
        )

    if INSTRUMENTS_STEM_NAME in stems:
        combined_voices = fixture.spoken_speech + fixture.sung_vocals
        instr_metrics = sdr_sir_sar(stems[INSTRUMENTS_STEM_NAME], fixture.instruments, combined_voices)
        print(f"{INSTRUMENTS_STEM_NAME:<16} {instr_metrics['sdr_db']:<10.2f} {'—':<10} {'—':<10}  vs. instrument bed")

    print(
        "\nHonest read: SIR is the number that matters for \"did the two voices\n"
        "land in separate stems\" — low or negative SIR means the other voice's\n"
        "energy dominates what should be a clean stem. This fixture overlaps\n"
        "both voices for the entire clip (the hard case, not the easy one), and\n"
        "is synthetic — real speech/singing spectra will differ. Treat this as\n"
        "a regression check on the pipeline's routing, not a quality benchmark."
    )


if __name__ == "__main__":
    main()
