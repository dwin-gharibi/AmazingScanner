from __future__ import annotations

import argparse
import time
import traceback

__all__ = ["STEPS", "run", "main"]


def _export(args) -> int:
    from .engine.export_models import main as m
    return m(["--out", args.models, "--runs", args.runs])


def _evaluate(args) -> int:
    from .eval.evaluate import main as m
    argv = ["--runs", args.runs, "--all"]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    return m(argv)


def _charts(args) -> int:
    from .eval.charts import main as m
    return m([])


def _figures(args) -> int:
    from .eval.figures import main as m
    return m(["--charts", "--images"] if args.quick else ["--all"])


def _stages(args) -> int:
    from .eval.stages import main as m
    argv = []
    if args.quick:
        argv += ["--limit", "6"]
    return m(argv)


def _gallery(args) -> int:
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "build_demo_gallery.py"
    if not script.exists():
        print("  [skip] gallery: scripts/build_demo_gallery.py not found")
        return 0
    return subprocess.call([sys.executable, str(script)])


def _report(args) -> int:
    from .eval.report import main as m
    return m([])


def _tests(args) -> int:
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    return subprocess.call(cmd)


STEPS: dict[str, tuple] = {
    "export":   (_export,   "pick the best run per model, write fp16 weights", False),
    "evaluate": (_evaluate, "every benchmark table", False),
    "charts":   (_charts,   "one chart per measured question", False),
    "figures":  (_figures,  "galleries, qualitative sheets, demo GIFs", False),
    "stages":   (_stages,   "per-stage inputs, outputs and accuracy", False),
    "gallery":  (_gallery,  "publish the per-photo strips into docs/assets", False),
    "report":   (_report,   "docs/REPORT.md and the README results block", False),
    "tests":    (_tests,    "the full test suite", True),
}


def run(args) -> int:
    skip = set(args.skip or [])
    if args.only:
        wanted = [s for s in args.only if s in STEPS]
    else:
        wanted = [n for n in STEPS if n not in skip]
    if args.quick:
        wanted = [n for n in wanted if not STEPS[n][2]]

    print(f"[refresh] {len(wanted)} step(s): {', '.join(wanted)}")
    results: list[tuple[str, str, float]] = []
    t_all = time.time()

    for name in wanted:
        fn, desc, _ = STEPS[name]
        print(f"\n=== {name} — {desc} ===", flush=True)
        t0 = time.time()
        try:
            rc = fn(args) or 0
            status = "ok" if rc == 0 else f"exit {rc}"
        except Exception as exc:
            status = f"{type(exc).__name__}: {exc}"
            if args.traceback:
                traceback.print_exc()
        results.append((name, status, time.time() - t0))

    print("\n" + "=" * 60)
    print(f"{'step':<10} {'status':<34} {'seconds':>9}")
    for name, status, secs in results:
        print(f"{name:<10} {status:<34} {secs:>9.1f}")
    failed = [n for n, s, _ in results if s != "ok"]
    print("=" * 60)
    print(f"total {time.time() - t_all:.1f}s"
          + (f" — {len(failed)} step(s) failed: {', '.join(failed)}" if failed
             else " — everything refreshed"))
    if not failed:
        print("\nUpdated:")
        for p in ("models/*.pt", "outputs/report/*.json", "docs/assets/charts/*.png",
                  "docs/assets/*.jpg", "docs/assets/*.gif", "outputs/stages/",
                  "docs/assets/stages/", "docs/REPORT.md", "docs/CHARTS.md",
                  "README.md"):
            print(f"  {p}")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate every derived artefact from the trained weights")
    ap.add_argument("--runs", default="runs", help="where the training runs live")
    ap.add_argument("--models", default="models", help="where to write fp16 weights")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images per evaluation set (useful for a smoke run)")
    ap.add_argument("--quick", action="store_true",
                    help="skip the GIFs and the test suite")
    ap.add_argument("--only", nargs="*", choices=list(STEPS),
                    help="run only these steps")
    ap.add_argument("--skip", nargs="*", choices=list(STEPS), help="skip these steps")
    ap.add_argument("--traceback", action="store_true")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
