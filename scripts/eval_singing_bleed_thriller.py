"""Speech-vs-Singing bleed regression check on a real cached job (not a
synthetic fixture): re-runs the given job's Speech-vs-Singing separation
through the actual job pipeline (backend.pool's "rerun" path — reuses the
original job's already-persisted source.wav, no re-fetch), then reports
backend.eval.metrics.cross_stem_leakage_fraction and window_rms_ratio
before (the job's existing, pre-fix stems) vs. after (the new run) for a
set of named windows.

This is the regression check for the cross-stem bleed fix in
_vocal_partition.py (see that module's and singing_sep.py's docstrings):
config.PIPELINE_VERSION was bumped so the "after" run can't be served from
the pre-fix cache, and the windows below are checked against known-good
baseline ratios so a future change can't quietly regress them.

    uv run --group speech python scripts/eval_singing_bleed_thriller.py <job_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from backend import jobs, pool
from backend.config import MUSIC_SAMPLE_RATE
from backend.eval.metrics import cross_stem_leakage_fraction, stem_envelope_correlation, window_rms_ratio
from backend.separators.singing_sep import SPOKEN_STEM_NAME, SUNG_STEM_NAME
from backend.storage import job_dir

# (label, start_seconds, end_seconds, target_stem, min_acceptable_ratio)
# Price monologue and verse are known-good baselines from before this fix —
# they must not regress. The bleed window is the bug report: one vocal
# performance split spectrally between both stems rather than routed to one
# (job c226c717..., listening test at 5:20-6:30) -- no established floor yet,
# just before-vs-after.
#
# "outro (held-out)" was picked by an automated scan of the *whole* track
# for a sustained, genuinely-active single-source stretch away from the
# other three windows (scripts/find_holdout*.py, not checked in) -- it was
# never used to tune _ARBITRATION_STRENGTH, the hysteresis thresholds, or
# the beat/harmony feature weights. Price monologue and sung verse were
# used for BOTH tuning and validation in earlier fixes, which is exactly
# the overfitting risk a held-out window exists to catch.
WINDOWS = [
    ("Price monologue (spoken)", 4 * 60 + 15, 4 * 60 + 52, SPOKEN_STEM_NAME, 15.2),
    ("sung verse", 2 * 60 + 20, 2 * 60 + 50, SUNG_STEM_NAME, 12.0),
    ("bleed region (5:20-6:30)", 5 * 60 + 20, 6 * 60 + 30, None, None),
    ("outro (held-out, sung)", 12 * 60 + 20, 12 * 60 + 55, SUNG_STEM_NAME, None),
]


def _read_stem(directory: Path, name: str) -> tuple:
    audio, sr = sf.read(directory / f"{name}.wav", dtype="float32", always_2d=True)
    assert sr == MUSIC_SAMPLE_RATE, f"{directory / f'{name}.wav'}: expected {MUSIC_SAMPLE_RATE} Hz, got {sr}"
    return audio.T  # (samples, channels) -> (channels, samples)


def _report(label: str, directory: Path) -> dict:
    spoken = _read_stem(directory, SPOKEN_STEM_NAME)
    sung = _read_stem(directory, SUNG_STEM_NAME)
    print(f"\n{label} ({directory})")
    print(f"  {'window':<28} {'leakage frac':<14} {'correlation':<13} {'target':<8} {'RMS ratio':<12}")
    report = {}
    for window_label, start, end, target_name, min_ratio in WINDOWS:
        target = spoken if target_name == SPOKEN_STEM_NAME else sung if target_name == SUNG_STEM_NAME else None
        other = sung if target is spoken else spoken if target is not None else None

        # Full (unsliced) arrays + start/end: cross_stem_leakage_fraction
        # measures "own peak" over the whole clip, not the queried slice --
        # slicing first would make a quiet residual bleed within the window
        # look identical to a loud one (see that function's docstring).
        leakage = cross_stem_leakage_fraction(spoken, sung, MUSIC_SAMPLE_RATE, start_seconds=start, end_seconds=end)
        # Distinguishes real duplication (one performance split spectrally,
        # both stems' envelopes rise/fall together) from mere co-activity,
        # which leakage_fraction alone can't tell apart -- see that
        # function's docstring.
        correlation = stem_envelope_correlation(spoken, sung, MUSIC_SAMPLE_RATE, start, end)
        ratio = window_rms_ratio(target, other, MUSIC_SAMPLE_RATE, start, end) if target is not None else None
        report[window_label] = {"leakage_fraction": leakage, "correlation": correlation, "rms_ratio": ratio, "min_ratio": min_ratio}
        ratio_str = f"{ratio:.2f}x" if ratio is not None else "—"
        target_str = target_name or "—"
        print(f"  {window_label:<28} {leakage:<14.3f} {correlation:<+13.3f} {target_str:<8} {ratio_str:<12}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job_id", help="the cached Speech-vs-Singing job to re-run and compare against")
    args = parser.parse_args()

    original = jobs.get_job(args.job_id)
    if original is None:
        print(f"job {args.job_id} not found", file=sys.stderr)
        raise SystemExit(1)
    if original.mode != "singing":
        print(f"job {args.job_id} is mode={original.mode!r}, not 'singing'", file=sys.stderr)
        raise SystemExit(1)

    before = _report("BEFORE (existing cached stems)", job_dir(args.job_id))

    new_job_id = jobs.create_job(
        mode=original.mode,
        tier=original.tier,
        source_type="rerun",
        source_ref=args.job_id,
        stem_count=original.stem_count,
    )
    print(f"\nre-running as job {new_job_id} ...")
    pool.process_job(new_job_id)
    new_job = jobs.get_job(new_job_id)
    if new_job.status != "done":
        print(f"re-run failed: {new_job.error}", file=sys.stderr)
        raise SystemExit(1)

    after = _report("AFTER (new run, fixed pipeline)", job_dir(new_job_id))

    print(f"\nnew job id: {new_job_id}")
    print(f"\n{'window':<28} {'before ratio':<14} {'after ratio':<14} {'min required':<14} {'regressed?'}")
    ok = True
    for window_label, _start, _end, _target, min_ratio in WINDOWS:
        b, a = before[window_label]["rms_ratio"], after[window_label]["rms_ratio"]
        if min_ratio is None:
            print(f"{window_label:<28} {'—' if b is None else f'{b:.2f}x':<14} {'—' if a is None else f'{a:.2f}x':<14} {'—':<14}")
            continue
        regressed = a < min_ratio
        ok = ok and not regressed
        print(f"{window_label:<28} {b:<14.2f} {a:<14.2f} {min_ratio:<14.2f} {'YES' if regressed else 'no'}")

    print(f"\n{'window':<28} {'before leakage':<16} {'after leakage':<16}")
    for window_label, *_rest in WINDOWS:
        b, a = before[window_label]["leakage_fraction"], after[window_label]["leakage_fraction"]
        print(f"{window_label:<28} {b:<16.3f} {a:<16.3f}")

    print(f"\n{'window':<28} {'before corr':<14} {'after corr':<14}")
    for window_label, *_rest in WINDOWS:
        b, a = before[window_label]["correlation"], after[window_label]["correlation"]
        print(f"{window_label:<28} {b:<+14.3f} {a:<+14.3f}")

    if not ok:
        print("\nREGRESSION: a verified window's RMS ratio dropped below its required floor.", file=sys.stderr)
        raise SystemExit(1)
    print("\nNo regressions on the verified windows.")


if __name__ == "__main__":
    main()
