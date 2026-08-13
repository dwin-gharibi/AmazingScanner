#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/report/ocr_real.json")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    from docscanner.data.datasets import RealPhotoSet
    from docscanner.eval.ocr import ocr_available, readability_report
    from docscanner.pipeline.enhance_pipeline import EnhancementPipeline
    from docscanner.utils.geometry import order_corners, rectify
    from docscanner.utils.imageio import imread_rgb

    if not ocr_available():
        print("[ocr-real] no OCR engine — skipped")
        return 0

    root = Path("data/real/own")
    manifest = root / "annotations.json"
    if not manifest.exists():
        print("  [skip] no annotated real photographs")
        return 0

    ds = RealPhotoSet(manifest)
    enhancer = EnhancementPipeline("models/enhance.pt", max_side=1600)
    per_image, t0 = [], time.time()

    for i in range(len(ds)):
        photo, gt, item = ds.raw(i)
        page = rectify(photo, order_corners(np.asarray(gt, np.float32)),
                       max_side=1600)[0]
        ours = enhancer(page, mode="color").image

        images = {"rectified input": page, "ours": ours}
        ref_rel = item.get("reference")
        ref_path = root / ref_rel if ref_rel else None
        if ref_path and ref_path.exists():
            images["commercial reference"] = imread_rgb(ref_path)

        rep = readability_report(images)
        rec = {"image": Path(item["file"]).name.split("_")[0]}
        for key, r in rep.items():
            rec[f"{key} conf"] = r["confidence"]
            rec[f"{key} words"] = r["words"]
        per_image.append(rec)
        print(f"  {rec['image']:8} input {rec.get('rectified input conf', 0):5.1f} "
              f"| ours {rec.get('ours conf', 0):5.1f} "
              f"| ref {rec.get('commercial reference conf', 0):5.1f}")

    def _mean(key: str) -> float:
        vals = [r[key] for r in per_image if key in r and not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    rows = []
    for label in ("rectified input", "ours", "commercial reference"):
        if f"{label} conf" not in per_image[0]:
            continue
        rows.append({"Image": label, "n": len(per_image),
                     "OCR confidence": _mean(f"{label} conf"),
                     "words read": _mean(f"{label} words")})

    verdict = {}
    if len(rows) == 3:
        ours_w = np.array([r["ours words"] for r in per_image], float)
        ref_w = np.array([r["commercial reference words"] for r in per_image], float)
        inp_w = np.array([r["rectified input words"] for r in per_image], float)
        ours_c = np.array([r["ours conf"] for r in per_image], float)
        ref_c = np.array([r["commercial reference conf"] for r in per_image], float)
        verdict = {
            "pages where ours beats the raw input (words)": int((ours_w > inp_w).sum()),
            "pages where ours matches or beats the app (words)": int((ours_w >= ref_w).sum()),
            "pages where ours matches or beats the app (confidence)":
                int((ours_c >= ref_c).sum()),
            "n": len(per_image),
        }

    payload = {"rows": rows, "per_image": per_image, "verdict": verdict,
               "seconds": round(time.time() - t0, 1)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))

    cols = list(rows[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(
            f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    if verdict:
        md += ["", "| Question | Answer |", "|---|---|"]
        for k, v in verdict.items():
            if k != "n":
                md.append(f"| {k} | **{v} / {verdict['n']}** |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
