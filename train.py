from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    task, rest = argv[0], argv[1:]
    if task == "enhance":
        from docscanner.engine.train_enhance import main as run
        return run(rest)
    if task == "corners":
        from docscanner.engine.train_corners import main as run
        if not any(a.startswith("--approach") for a in rest):
            rest = ["--approach", "heatmap", *rest]
        return run(rest)
    if task == "all":
        from docscanner.all_in_one import main as run
        return run(rest)

    print(f"unknown task {task!r}\n{__doc__}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
