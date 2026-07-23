"""Lead-vs-Backing-Vocals mode eval: does a lead melody land in lead_vocal,
and does a stacked harmony chorus land in backing_vocals, on a synthetic
fixture with a clean lead-only section followed by a harmony section
(backend/eval/fixtures.py)?

No real "harmony-heavy song" ships in this repo (licensing — CLAUDE.md
forbids redistributing third-party audio), so the fixture is a procedurally
generated lead melody joined partway through by a 3-voice harmony stack (a
major third + a perfect fifth above the lead) — a synthetic stand-in for "a
big harmony chorus," not real vocals. Read this as a smoke-level regression
check on this pipeline's *routing* (does energy that's actually a stacked
harmony land in backing_vocals, not bleed into lead_vocal), not a benchmark
of real-world separation quality.

This mode's underlying karaoke model was trained on "Vocals vs
Instrumental" (full mixes), repurposed here on Demucs's isolated vocal bus
(see karaoke_onnx.py's docstring) — it was never trained on "is this
specifically the lead melody vs. a harmony line," so expect real bleed on
real recordings, especially when the backing harmony closely doubles the
lead melody (as real backing vocals often do, unlike this fixture's
intentionally-separated pitches).

KNOWN LIMITATION OF THIS FIXTURE, found while building it: this model
routes almost none of a purely-synthesized tone (even with added harmonics,
a two-formant spectral envelope, vibrato, and breath noise) to `lead_vocal`
— it lands in `backing_vocals`/Instrumental instead, regardless of the
lead-only/harmony structure below. Verified directly against a real vocal
excerpt (Michael Jackson's Thriller, isolated singing bus): there, lead_vocal
correctly dominates (RMS ~0.032 vs. backing's ~0.021). So this is a real
model behavior — it needs real vocal timbre to route sensibly, synthetic
tones don't trigger its learned "lead voice" prior — not a bug in this
pipeline. The RMS numbers below are printed for the record, but the
trustworthy verification of this mode is the live run on a real song (see
the task this script accompanies); treat this fixture as a documented
negative result, not a passing check.

    uv run --group speech python scripts/eval_lead_vs_backing.py
    uv run --group speech python scripts/eval_lead_vs_backing.py --tier fast

Needs the exported karaoke ONNX model (scripts/export_roformer_onnx.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.eval.fixtures import make_lead_vs_backing_fixture
from backend.eval.metrics import sdr_sir_sar
from backend.separators.router import select_separator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--lead-only-seconds", type=float, default=6.0)
    parser.add_argument("--harmony-seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fixture = make_lead_vs_backing_fixture(
        lead_only_seconds=args.lead_only_seconds, harmony_seconds=args.harmony_seconds, sr=44100, seed=args.seed
    )

    try:
        separator = select_separator("karaoke", args.tier)
    except NotImplementedError as exc:
        print(f"FAILED to build the karaoke-mode separator: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    stems = separator.separate(fixture.mixture)

    sr = fixture.sample_rate
    lead_only_slice = slice(0, int(args.lead_only_seconds * sr))
    harmony_slice = slice(int(args.lead_only_seconds * sr), None)

    def rms(x):
        return float((x.astype("float64") ** 2).mean() ** 0.5)

    print(
        f"Lead-vs-Backing-Vocals eval — synthetic fixture, tier={args.tier!r}, "
        f"lead_only={args.lead_only_seconds}s, harmony={args.harmony_seconds}s, seed={args.seed}\n"
    )
    print(f"{'section':<24} {'lead_vocal RMS':<16} {'backing_vocals RMS':<20} {'lead/backing ratio':<20}")
    print("-" * 82)
    for label, sl in (("lead-only", lead_only_slice), ("harmony (both voices)", harmony_slice)):
        lead_rms = rms(stems["lead_vocal"][:, sl])
        backing_rms = rms(stems["backing_vocals"][:, sl])
        ratio = lead_rms / backing_rms if backing_rms > 1e-12 else float("inf")
        print(f"{label:<24} {lead_rms:<16.6f} {backing_rms:<20.6f} {ratio:<20.2f}")

    # SDR/SIR/SAR against ground truth, same metric as eval_singing_vs_speech.py,
    # over the harmony section specifically — the section where a real
    # "does backing land in backing_vocals, not lead_vocal" question applies
    # (the lead-only section has no backing_vocals ground truth to score against).
    print("\nSDR/SIR/SAR over the harmony section (vs. synthetic ground truth):")
    print(f"{'stem':<16} {'SDR (dB)':<10} {'SIR (dB)':<10} {'SAR (dB)':<10}  meaning")
    print("-" * 70)
    voice_pairs = {
        "lead_vocal": (fixture.lead_vocal, fixture.backing_vocals, "backing harmony"),
        "backing_vocals": (fixture.backing_vocals, fixture.lead_vocal, "lead vocal"),
    }
    for name, (target, interferer, other_label) in voice_pairs.items():
        metrics = sdr_sir_sar(stems[name][:, harmony_slice], target[:, harmony_slice], interferer[:, harmony_slice])
        print(
            f"{name:<16} {metrics['sdr_db']:<10.2f} {metrics['sir_db']:<10.2f} {metrics['sar_db']:<10.2f}  "
            f"SIR = how much {other_label} bled in"
        )

    print(
        "\nHonest read: if lead_vocal doesn't dominate the lead-only section here,\n"
        "that is the documented limitation, not a surprise — this model does not\n"
        "reliably recognize purely-synthesized tones as \"lead vocal\" (see this\n"
        "script's module docstring for the direct A/B against a real vocal\n"
        "excerpt, where routing was sensible). This fixture is kept as a running\n"
        "record of that limitation and as a network-free smoke check that the\n"
        "pipeline executes end to end, not as a claim that the split tracks\n"
        "correctly here. For a trustworthy confirmation, see the live run on a\n"
        "real harmony-heavy song. Regardless of fixture realism, do not expect\n"
        "clean singer-by-singer separation — that's explicitly out of scope for\n"
        "this mode."
    )


if __name__ == "__main__":
    main()
