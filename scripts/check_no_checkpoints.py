#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

MAX_EXPORT_MB = 40.0
ALLOWED_DIR = "models"
JUNK_DIRS = (".ipynb_checkpoints", "__pycache__", ".pytest_cache", ".DS_Store")


def main(argv: list[str]) -> int:
    problems: list[str] = []

    for name in argv:
        path = Path(name)
        parts = path.parts
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0.0

        if parts and parts[0] == ALLOWED_DIR:
            if size_mb > MAX_EXPORT_MB:
                problems.append(
                    f"{path} is {size_mb:.0f} MB — exported weights should be fp16 and "
                    f"well under {MAX_EXPORT_MB:.0f} MB. Re-run `amazingscanner export`.")
            continue

        problems.append(
            f"{path} ({size_mb:.0f} MB) is a training checkpoint. "
            f"Only exported weights under {ALLOWED_DIR}/ belong in git — "
            f"run `amazingscanner export` and commit that instead.")

    problems += _tracked_junk()

    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    return 1 if problems else 0


def _tracked_junk() -> list[str]:
    import subprocess
    try:
        out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                             check=True).stdout.splitlines()
    except Exception:
        return []
    hits = sorted({f for f in out
                   if any(f"/{d}/" in f or f.startswith(f"{d}/") for d in JUNK_DIRS)})
    if not hits:
        return []
    shown = ", ".join(hits[:3]) + (f" (+{len(hits) - 3} more)" if len(hits) > 3 else "")
    return [f"{len(hits)} tracked editor/notebook file(s): {shown}. "
            f"Remove with `git rm -r --cached` — these are stale copies that "
            f"never regenerate."]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
