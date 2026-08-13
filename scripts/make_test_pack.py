#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
SEED_BASE = 7000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate unseen test photographs")
    ap.add_argument("--out", default="data/test_pack")
    ap.add_argument("--n", type=int, default=18, help="total photographs")
    ap.add_argument("--seed", type=int, default=SEED_BASE)
    ap.add_argument("--long-side", type=int, default=1280)
    args = ap.parse_args(argv)

    from docscanner.data.degrade import DegradationConfig
    from docscanner.data.prepare import load_splits, make_generator
    from docscanner.utils.imageio import imwrite_rgb

    out = Path(args.out)
    photos, truth = out / "photos", out / "ground_truth"
    for d in (photos, truth / "targets"):
        d.mkdir(parents=True, exist_ok=True)

    styles = {
        "clean": DegradationConfig.for_enhancement(),
        "corner-hard": DegradationConfig.for_corners(),
        "ood": DegradationConfig.hard(),
    }
    splits = load_splits()
    labels: list[dict] = []
    style_names = list(styles)
    per = dict.fromkeys(style_names, 0)
    for k in range(args.n):
        style = style_names[k % len(style_names)]
        gen = make_generator("test", cfg=styles[style], splits=splits,
                             photo_long_side=args.long_side)
        s = gen.generate(np.random.default_rng((args.seed, k)),
                         want_rectified=True)
        stem = f"pack_{k:02d}_{style}"
        imwrite_rgb(photos / f"{stem}.jpg", s.photo, quality=92)
        imwrite_rgb(truth / "targets" / f"{stem}.png", s.target)
        labels.append({
            "file": f"photos/{stem}.jpg",
            "style": style,
            "corners": np.asarray(s.corners, float).round(2).tolist(),
            "target": f"ground_truth/targets/{stem}.png",
        })
        per[style] += 1

    (truth / "corners.json").write_text(json.dumps(
        {"note": "TL,TR,BR,BL in photo pixels; generated from TEST-split "
                 "scans at seeds >= 7000 - never seen by any model here",
         "count": len(labels), "items": labels}, indent=1))
    (out / "styles.json").write_text(json.dumps(per, indent=1))
    (out / "README.md").write_text(
        "# Unseen test pack\n\n"
        "Photographs generated for **testing only** — every source page is "
        "from the held-out test split (no model here has trained on it in any "
        "form) and every composite uses seeds no training or frozen-set "
        "tooling draws from.\n\n"
        f"{len(labels)} photographs across three styles: `clean` (the "
        "enhancement policy), `corner-hard` (full circle of orientations, "
        "small pages), and `ood` (curl, glare, motion blur — degradations no "
        "model trained on). `ground_truth/corners.json` has the exact quads "
        "and `ground_truth/targets/` the clean pages, so runs can be scored:\n\n"
        "```bash\n"
        "amazingscanner scan data/test_pack/photos --out /tmp/pack_scans\n"
        "python scripts/make_test_pack.py --n 18   # regenerate / extend\n"
        "```\n\n"
        "Regenerating with a different `--seed` (staying above 7000) mints a "
        "fresh pack the models have still never seen.\n")
    print(f"wrote {len(labels)} photographs -> {out}  ({per})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
