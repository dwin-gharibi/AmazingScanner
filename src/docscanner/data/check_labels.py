from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils.geometry import (estimate_page_size, order_corners, quad_area,
                              quad_is_convex)
from ..utils.imageio import imread_rgb, imwrite_rgb
from ..utils.viz import draw_quad, grid, label_bar

__all__ = ["check_manifest", "LabelReport"]


@dataclass
class LabelReport:
    total: int = 0
    ok: int = 0
    problems: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.problems)

    def summary(self) -> str:
        lines = [f"checked {self.total} annotations: {self.ok} clean, "
                 f"{self.failed} with problems, {len(self.warnings)} with warnings"]
        for p in self.problems:
            lines.append(f"  ERROR  {p['file']}: {p['issue']}")
        for w in self.warnings:
            lines.append(f"  warn   {w['file']}: {w['issue']}")
        return "\n".join(lines)


def check_manifest(
    manifest: str | Path,
    contact_sheet: str | Path | None = None,
    min_area_frac: float = 0.02,
    max_aspect: float = 6.0,
) -> LabelReport:
    manifest = Path(manifest)
    data = json.loads(manifest.read_text())
    root = manifest.parent
    report = LabelReport(total=len(data.get("items", [])))
    tiles = []

    for item in data.get("items", []):
        name = item["file"]
        issues: list[str] = []
        warns: list[str] = []

        pts = np.asarray(item.get("corners", []), np.float64)
        if pts.shape != (4, 2) or not np.isfinite(pts).all():
            report.problems.append({"file": name, "issue": "not four finite points"})
            continue

        path = root / name
        if not path.exists():
            report.problems.append({"file": name, "issue": "image file missing"})
            continue
        img = imread_rgb(path)
        h, w = img.shape[:2]

        if not quad_is_convex(pts):
            issues.append("quad is not convex (corners likely clicked out of order)")

        area_frac = quad_area(pts) / float(w * h)
        if area_frac < min_area_frac:
            issues.append(f"page covers only {area_frac * 100:.1f}% of the frame")
        elif area_frac > 1.4:
            warns.append(f"page covers {area_frac * 100:.0f}% of the frame")

        margin = 0.08
        if (pts[:, 0] < -margin * w).any() or (pts[:, 0] > (1 + margin) * w).any() \
           or (pts[:, 1] < -margin * h).any() or (pts[:, 1] > (1 + margin) * h).any():
            warns.append("a corner lies well outside the image")

        canonical = order_corners(pts)
        if not np.allclose(canonical, pts.astype(np.float32), atol=1.5):
            warns.append("stored order differs from canonical TL,TR,BR,BL "
                         "(it will be re-ordered on load)")

        pw, ph = estimate_page_size(canonical, img.shape[:2])
        aspect = max(pw, ph) / max(min(pw, ph), 1)
        if aspect > max_aspect:
            warns.append(f"implied page aspect {aspect:.1f}:1 is unusual")

        if issues:
            report.problems.append({"file": name, "issue": "; ".join(issues)})
        else:
            report.ok += 1
        if warns:
            report.warnings.append({"file": name, "issue": "; ".join(warns)})

        status = "OK" if not issues else "PROBLEM"
        tiles.append(label_bar(draw_quad(img, canonical),
                               f"{Path(name).name}  [{status}]"))

    if contact_sheet and tiles:
        sheet = grid(tiles, cols=4, cell=340)
        imwrite_rgb(contact_sheet, sheet, quality=92)
        print(f"  contact sheet -> {contact_sheet}")

    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate corner annotations")
    ap.add_argument("--manifest", default="data/real/own/annotations.json")
    ap.add_argument("--sheet", default="docs/assets/label_review.jpg",
                    help="contact sheet written for human review")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any annotation has a problem")
    args = ap.parse_args(argv)

    path = Path(args.manifest)
    if not path.exists():
        print(f"no manifest at {path}\n"
              f"Annotate your photos first (see data/real/own/README.md), then run:\n"
              f"  python -m docscanner.data.prepare --own")
        return 0

    report = check_manifest(path, contact_sheet=args.sheet)
    print(report.summary())
    return 1 if (args.strict and report.failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
