#!/usr/bin/env python3
"""Check evaluation outputs against stored regression bounds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BOUNDS_FILE = Path(__file__).resolve().parent / "regression_bounds.json"
DEFAULT_TOLERANCE = 0.15


def load_rows(report: Path, name: str) -> list[dict]:
    path = report / f"{name}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("rows", []) if isinstance(payload, dict) else []


def collect(report: Path) -> dict[str, float]:
    measured: dict[str, float] = {}
    for row in load_rows(report, "corners"):
        if row.get("edge refine") != "no" or row.get("TTA") != "yes":
            continue
        model = "heatmap" if "heatmap" in row.get("Model", "") else "regression"
        dataset = row.get("Set", "?")
        if "own" in dataset:
            tag = "real_own"
        elif "MIDV" in dataset:
            tag = "real_midv"
        elif "course only" in dataset:
            tag = "synthetic_course"
        elif "synthetic" in dataset:
            tag = "synthetic"
        else:
            continue
        for metric, key in (("MCE (px)", "mce_px"), ("median (px)", "median_px")):
            if metric in row:
                measured[f"corners.{model}.{tag}.{key}"] = float(row[metric])
        if "quad IoU" in row:
            measured[f"corners.{model}.{tag}.neg_iou"] = -float(row["quad IoU"])

    for row in load_rows(report, "end_to_end"):
        if "predicted" in row.get("Rectified with", ""):
            if "corner error (px)" in row:
                measured["end_to_end.corner_error_px"] = float(row["corner error (px)"])
            if "OCR confidence" in row:
                measured["end_to_end.neg_ocr_confidence"] = -float(row["OCR confidence"])

    for row in load_rows(report, "enhancement_enhance_main"):
        if str(row.get("Split", "")).strip() == "test":
            if "PSNR" in row:
                measured["enhancement.test.neg_psnr_db"] = -float(row["PSNR"])
            if "SSIM" in row:
                measured["enhancement.test.neg_ssim"] = -float(row["SSIM"])

    return measured


def _markdown(measured: dict[str, float], baseline: dict[str, float],
              tolerance: float) -> str:
    lines = ["| Measurement | Baseline | This run | Change | |",
             "|---|---:|---:|---:|:--|"]
    for key, now in sorted(measured.items()):
        display_key = key
        was = baseline.get(key)
        flip = -1.0 if ".neg_" in key else 1.0
        if flip < 0:
            display_key = key.replace(".neg_", ".")
        if was is None:
            lines.append(f"| `{display_key}` | — | {flip * now:.3f} | new | |")
            continue
        ceiling = was + max(abs(was) * tolerance, 0.5)
        delta = now - was
        if now > ceiling:
            mark, note = "FAIL", "regressed"
        elif delta < -max(abs(was) * 0.02, 0.05):
            mark, note = "OK", "improved"
        else:
            mark, note = "--", "unchanged"
        lines.append(f"| `{display_key}` | {flip * was:.3f} | {flip * now:.3f} | "
                     f"{-delta if flip < 0 else delta:+.3f} | {mark} {note} |")
    lines += ["",
              f"Bounds are ceilings with a {tolerance:.0%} tolerance, held in "
              "`scripts/regression_bounds.json`.",
              "Corner errors are in pixels on the frozen sets — the *set* is "
              "part of the number, since the same detector measures around a "
              "pixel on synthetic pages and tens of pixels on photographs."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "Check evaluation outputs against regression bounds.").splitlines()[0]
    )
    ap.add_argument("--report", type=Path, default=Path("outputs/report"),
                    help="directory holding the evaluation JSON")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="fractional worsening allowed before failing")
    ap.add_argument("--update", action="store_true",
                    help="rewrite the baseline from this run — do this on purpose")
    ap.add_argument("--markdown", type=Path, default=None,
                    help="also write a table of the comparison here")
    args = ap.parse_args(argv)

    measured = collect(args.report)
    if not measured:
        print(f"error: no measurements found under {args.report}. "
              f"Run `amazingscanner eval --all` first.", file=sys.stderr)
        return 1

    if args.update or not BOUNDS_FILE.exists():
        BOUNDS_FILE.write_text(json.dumps(
            {"tolerance": args.tolerance,
             "note": "Ceilings, not targets. Every value is 'lower is better'; "
                     "metrics where higher is better are stored negated.",
             "baseline": {k: round(v, 4) for k, v in sorted(measured.items())}},
            indent=2) + "\n")
        verb = "Updated" if args.update else "Created"
        try:
            shown = BOUNDS_FILE.relative_to(Path.cwd())
        except ValueError:
            shown = BOUNDS_FILE
        print(f"{verb} {shown} with {len(measured)} baselines.")
        return 0

    spec = json.loads(BOUNDS_FILE.read_text())
    baseline = spec.get("baseline", {})
    tolerance = float(spec.get("tolerance", args.tolerance))

    regressions: list[str] = []
    improvements: list[str] = []
    unseen: list[str] = []

    for key, now in sorted(measured.items()):
        if key not in baseline:
            unseen.append(key)
            continue
        was = float(baseline[key])
        ceiling = was + max(abs(was) * tolerance, 0.5)
        delta = now - was
        if now > ceiling:
            regressions.append(
                f"  {key}\n      was {was:+.3f}  now {now:+.3f}  "
                f"({delta:+.3f}, ceiling {ceiling:+.3f})")
        elif delta < -max(abs(was) * 0.02, 0.05):
            improvements.append(f"  {key}: {was:+.3f} → {now:+.3f} ({delta:+.3f})")

    print(f"Checked {len(measured)} measurements against "
          f"{len(baseline)} baselines (tolerance {tolerance:.0%})\n")

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(measured, baseline, tolerance))

    if improvements:
        print("Improved:")
        print("\n".join(improvements))
        print("\n  Bank these with `python scripts/check_regression.py --update`,"
              "\n  and only once the run that produced them is the one you ship.\n")

    if unseen:
        print(f"New measurements with no baseline ({len(unseen)}):")
        for k in unseen:
            print(f"  {k}")
        print()

    if regressions:
        print(f"REGRESSED ({len(regressions)}):")
        print("\n".join(regressions))
        print("\nIf this is a deliberate trade — better on photographs, worse on"
              "\nsynthetic pages, say — re-baseline with --update and say so in"
              "\nthe pull request. Do not widen the tolerance to make it pass.")
        return 1

    print("No regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
