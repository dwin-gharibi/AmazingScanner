#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SHIPPED = "models/corner_heatmap.pt"
FINETUNED = "runs/e2e_finetune/best.pt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", default=SHIPPED)
    ap.add_argument("--finetuned", default=FINETUNED)
    ap.add_argument("--out", default="outputs/report/finetune.json")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    from docscanner.data.datasets import RealPhotoSet
    from docscanner.eval.metrics import corner_metrics
    from docscanner.eval.ocr import ocr_available, readability_report
    from docscanner.pipeline.scanner import DocumentScanner
    from docscanner.utils.geometry import order_corners

    manifest = Path("data/real/own/annotations.json")
    if not manifest.exists():
        print("  [skip] no annotated real photographs")
        return 0
    ds = RealPhotoSet(manifest)
    n = len(ds)
    arms = {"baseline (trained on coordinates)": args.baseline,
            "fine-tuned through the warp": args.finetuned}

    rows, gap_rows, per_image = [], [], {}
    t0 = time.time()

    ceiling = {"conf": [], "words": []}
    scanner0 = DocumentScanner(args.baseline, "models/enhance.pt")
    for i in range(n):
        photo, gt, _ = ds.raw(i)
        if ocr_available():
            rep = readability_report(
                {"gt": scanner0.scan(photo, corners=gt, auto_orientation=False).image})
            ceiling["conf"].append(rep["gt"]["confidence"])
            ceiling["words"].append(rep["gt"]["words"])

    for label, ckpt in arms.items():
        if not Path(ckpt).exists():
            print(f"  [skip] {label}: {ckpt} not found")
            continue
        scanner = DocumentScanner(ckpt, "models/enhance.pt")
        preds, gts, sizes, conf, words = [], [], [], [], []
        for i in range(n):
            photo, gt, _ = ds.raw(i)
            res = scanner.scan(photo, auto_orientation=False)
            preds.append(res.corners)
            gts.append(order_corners(gt))
            sizes.append((photo.shape[1], photo.shape[0]))
            if ocr_available():
                rep = readability_report({"pred": res.image})
                conf.append(rep["pred"]["confidence"])
                words.append(rep["pred"]["words"])

        sc = corner_metrics(np.asarray(preds), np.asarray(gts), np.asarray(sizes))
        rows.append({"Detector": label, "n": n, "MCE (px)": sc.mce_px,
                     "median (px)": sc.median_px, "MCE (% diag)": sc.mce_norm * 100,
                     "quad IoU": sc.iou,
                     "success@16px": sc.success[16.0] * 100,
                     "success@32px": sc.success[32.0] * 100,
                     "worst (px)": sc.worst_px})
        per_image[label] = sc.per_image.tolist()

        if ocr_available():
            c_gt, c_pr = float(np.mean(ceiling["conf"])), float(np.mean(conf))
            w_gt, w_pr = float(np.mean(ceiling["words"])), float(np.mean(words))
            gap_rows.append({
                "Detector": label,
                "OCR conf (annotated corners)": c_gt,
                "OCR conf (predicted corners)": c_pr,
                "confidence gap": c_pr - c_gt,
                "words (annotated)": w_gt,
                "words (predicted)": w_pr,
                "words gap": w_pr - w_gt,
            })
        print(f"  {label:34s} MCE={sc.mce_px:6.2f}px  median={sc.median_px:6.2f}px  "
              f"IoU={sc.iou:.3f}")

    def _table(rs: list[dict]) -> str:
        if not rs:
            return "_not measured_\n"
        cols = list(rs[0])
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rs:
            out.append("| " + " | ".join(
                f"{r[c]:+.2f}" if isinstance(r[c], float) and "gap" in c
                else f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])
                for c in cols) + " |")
        return "\n".join(out) + "\n"

    md = ("### Corner accuracy\n\n" + _table(rows)
          + "\n### The downstream gap the brief asks about\n\n" + _table(gap_rows))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"rows": rows, "gap_rows": gap_rows, "per_image": per_image,
         "n": n, "seconds": round(time.time() - t0, 1)}, indent=1))
    out_path.with_suffix(".md").write_text(md)
    print("\n" + md)
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
