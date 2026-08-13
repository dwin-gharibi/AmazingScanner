#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REF_RE = re.compile(r'(?:!\[[^\]]*\]\(|(?<!\!)\[[^\]]*\]\(|src=")([^)"\s#]+)')


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return set(out.stdout.splitlines())


def check_references(problems: list[str]) -> None:
    for md in [ROOT / "README.md", *(ROOT / "docs").rglob("*.md"),
               *(ROOT / "deploy").rglob("*.md")]:
        text = md.read_text(encoding="utf-8")
        for m in REF_RE.finditer(text):
            ref = m.group(1)
            if ref.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            if not (md.parent / ref).resolve().exists():
                problems.append(f"{md.relative_to(ROOT)} references missing file: {ref}")


def check_generated_docs(problems: list[str]) -> None:
    if not (ROOT / "outputs" / "report" / "corners.json").exists():
        return
    before = {p: (ROOT / p).read_bytes() for p in ("docs/REPORT.md", "README.md")}
    gen = subprocess.run([sys.executable, "-m", "docscanner.eval.report"],
                        cwd=ROOT, capture_output=True, text=True)
    if gen.returncode != 0:
        problems.append("docs generator failed: "
                        + (gen.stderr or gen.stdout).strip().splitlines()[-1])
        return
    for rel, old in before.items():
        new = (ROOT / rel).read_bytes()
        if new != old:
            import difflib
            diff = list(difflib.unified_diff(
                old.decode("utf-8", "replace").splitlines(),
                new.decode("utf-8", "replace").splitlines(),
                fromfile=f"{rel} (committed)", tofile=f"{rel} (regenerated)",
                lineterm="", n=1))[:16]
            problems.append(
                f"{rel} is stale: regenerating it from outputs/report/*.json "
                f"produces different text. Commit the regenerated file, or "
                f"re-run the evaluation the numbers came from.\n  "
                + "\n  ".join(diff))
            (ROOT / rel).write_bytes(old)


def check_orphans(problems: list[str]) -> None:
    hay = ""
    for pat in ("README.md", "docs/**/*.md", "src/**/*.py", "scripts/*.py",
                "notebooks/*.ipynb"):
        for f in ROOT.glob(pat):
            hay += f.read_text(encoding="utf-8", errors="ignore")
    for asset in sorted(_tracked()):
        p = Path(asset)
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
            continue
        if not asset.startswith("docs/assets/"):
            continue
        name = p.name.replace("_dark", "")
        if p.parent.name in {"stages", "examples"}:
            continue
        if name not in hay:
            problems.append(f"orphaned asset nothing references: {asset}")


def check_notebooks(problems: list[str]) -> None:
    for nb_path in (ROOT / "notebooks").glob("*.ipynb"):
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            assert isinstance(nb.get("cells"), list) and nb["cells"]
        except Exception as exc:
            problems.append(f"{nb_path.relative_to(ROOT)} does not parse: {exc}")


def check_report_inputs_are_tracked(problems: list[str]) -> None:
    report_py = ROOT / "src" / "docscanner" / "eval" / "report.py"
    if not report_py.exists():
        return
    names = set(re.findall(r'_md\(\s*"([a-z0-9_]+)"', report_py.read_text()))
    tracked = _tracked()
    for name in sorted(names):
        rel = f"outputs/report/{name}.md"
        if not (ROOT / rel).exists():
            continue
        if rel not in tracked:
            problems.append(
                f"{rel} is referenced by the REPORT template but is not tracked "
                f"by git (outputs/ is ignored). Commit it with `git add -f {rel}` "
                f"and its .json, or CI will regenerate REPORT.md without it.")


def main() -> int:
    problems: list[str] = []
    check_references(problems)
    check_report_inputs_are_tracked(problems)
    check_generated_docs(problems)
    check_orphans(problems)
    check_notebooks(problems)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if not problems:
        print("documentation is consistent with the repository")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
