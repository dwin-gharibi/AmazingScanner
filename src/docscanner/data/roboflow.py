from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["RoboflowProject", "parse_universe_url", "sync", "main"]

API = "https://api.roboflow.com"
DEFAULT_DEST = Path("data/real/own")
FORMATS = ("coco-segmentation", "coco")


@dataclass
class RoboflowProject:
    workspace: str
    project: str
    version: int | str = 1
    api_key: str | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("ROBOFLOW_API_KEY")

    @property
    def slug(self) -> str:
        return f"{self.workspace}/{self.project}/{self.version}"

    def _get(self, url: str, params: dict[str, Any] | None = None):
        import requests

        params = dict(params or {})
        if self.api_key:
            params["api_key"] = self.api_key
        r = requests.get(url, params=params, timeout=60)
        if r.status_code in (401, 403):
            raise PermissionError(
                f"Roboflow refused the request for {self.slug} ({r.status_code}). "
                "Set ROBOFLOW_API_KEY to a key with access to this workspace; a "
                "public dataset still needs a key for the export endpoint.")
        if r.status_code == 404:
            raise FileNotFoundError(
                f"{self.slug} not found. Check the workspace/project slugs and that "
                "the version number exists.")
        r.raise_for_status()
        return r

    def info(self) -> dict[str, Any]:
        return self._get(f"{API}/{self.workspace}/{self.project}").json()

    def export_url(self, fmt: str) -> str:
        data = self._get(f"{API}/{self.workspace}/{self.project}/{self.version}/{fmt}").json()
        link = (data.get("export") or {}).get("link") or data.get("link")
        if not link:
            raise RuntimeError(
                f"Roboflow returned no download link for {self.slug} as {fmt}: "
                f"{json.dumps(data)[:300]}")
        return link

    def download(self, dest: Path, fmt: str | None = None) -> Path:
        import requests

        formats = (fmt,) if fmt else FORMATS
        last: Exception | None = None
        for f in formats:
            try:
                url = self.export_url(f)
            except Exception as exc:
                last = exc
                continue
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(dest)
            (dest / ".roboflow.json").write_text(json.dumps(
                {"slug": self.slug, "format": f}, indent=1), encoding="utf-8")
            print(f"  downloaded {self.slug} as {f} -> {dest}")
            return dest
        raise RuntimeError(f"could not export {self.slug} in any of {formats}: {last}")


def parse_universe_url(url: str) -> tuple[str, str, int]:
    m = re.search(r"roboflow\.com/([^/\s]+)/([^/\s?#]+)(?:/(?:dataset/)?(\d+))?", url)
    if not m:
        raise ValueError(f"could not parse a Roboflow project out of {url!r}")
    return m.group(1), m.group(2), int(m.group(3) or 1)


def _find_coco(root: Path) -> Path | None:
    cands = sorted(root.rglob("*.json"))
    cands.sort(key=lambda p: (0 if "annotations" in p.name else 1, len(p.parts)))
    return cands[0] if cands else None


def sync(project: RoboflowProject, dest: Path = DEFAULT_DEST,
         check_only: bool = False) -> dict[str, Any]:
    from .check_labels import check_manifest
    from .real import from_coco

    dest = Path(dest)
    staging = dest / ("export_rf_check" if check_only else "export_rf")
    if staging.exists():
        shutil.rmtree(staging)
    project.download(staging)

    coco = _find_coco(staging)
    if coco is None:
        raise RuntimeError(f"no COCO json inside the export at {staging}")
    images_dir = coco.parent if any(coco.parent.glob("*.jpg")) else staging

    out_root = dest / "_rf_check" if check_only else dest
    manifest = from_coco(coco, images_dir, out_root, name="own-photos")
    report = check_manifest(manifest)
    data = json.loads(Path(manifest).read_text())

    summary: dict[str, Any] = {
        "slug": project.slug,
        "annotated": data["count"],
        "clean": report.ok,
        "problems": len(report.problems),
    }

    if check_only:
        existing = dest / "annotations.json"
        if existing.exists():
            have = json.loads(existing.read_text())
            summary["in_repo"] = have["count"]
            summary["drifted"] = _drift(have, data)
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(out_root, ignore_errors=True)
    return summary


def _drift(a: dict, b: dict, tol: float = 1.0) -> list[str]:
    import numpy as np

    left = {Path(i["file"]).stem: i.get("corners") for i in a.get("items", [])}
    right = {Path(i["file"]).stem: i.get("corners") for i in b.get("items", [])}
    out = []
    for k in sorted(set(left) | set(right)):
        p, q = left.get(k), right.get(k)
        if p is None or q is None:
            out.append(f"{k}: {'only in repo' if q is None else 'only in Roboflow'}")
            continue
        if np.abs(np.asarray(p, float) - np.asarray(q, float)).max() > tol:
            out.append(f"{k}: corners moved")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sync the real test set with Roboflow")
    ap.add_argument("--url", default=None,
                    help="Roboflow project URL (workspace/project[/version])")
    ap.add_argument("--workspace", default="dwin-gharibi")
    ap.add_argument("--project", default="amazingscanner-ncqzb")
    ap.add_argument("--version", default=1)
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--sync", action="store_true", help="download and convert in place")
    ap.add_argument("--check", action="store_true",
                    help="compare against the checked-in manifest, write nothing")
    ap.add_argument("--info", action="store_true", help="print project metadata only")
    args = ap.parse_args(argv)

    ws, proj, ver = args.workspace, args.project, args.version
    if args.url:
        ws, proj, ver = parse_universe_url(args.url)
    rf = RoboflowProject(ws, proj, ver)

    if not rf.api_key:
        print("ROBOFLOW_API_KEY is not set.\n"
              "  The repository already contains a converted export, so nothing here\n"
              "  is required: `python -m docscanner.data.prepare --own` works offline.\n"
              "  Set the key only to re-download or to check for drift.")
        return 2

    try:
        if args.info or not (args.sync or args.check):
            print(json.dumps(rf.info(), indent=1)[:2000])
            return 0
        summary = sync(rf, Path(args.dest), check_only=args.check)
    except (PermissionError, FileNotFoundError, RuntimeError) as exc:
        print(f"  {type(exc).__name__}: {exc}")
        return 1

    print(f"  {summary['slug']}: {summary['annotated']} annotated, "
          f"{summary['clean']} clean, {summary['problems']} with problems")
    drift = summary.get("drifted")
    if drift:
        print(f"  DRIFT against the checked-in manifest ({len(drift)}):")
        for line in drift[:15]:
            print(f"    - {line}")
        return 1
    if args.check:
        print("  no drift: the checked-in annotations match the Roboflow project")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
