#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

WORD_RATIO_OK = 0.55
ASPECT_TOL = 0.35


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/report/reference_check.json")
    ap.add_argument("--sheet", default="docs/assets/reference_check.jpg")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)

    import cv2
    import numpy as np
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    from docscanner.data.datasets import RealPhotoSet
    from docscanner.eval.ocr import ocr_available, readability_report
    from docscanner.pipeline.scanner import DocumentScanner
    from docscanner.utils.imageio import imread_rgb
    from docscanner.utils.viz import label_bar

    root = Path("data/real/own")
    ds = RealPhotoSet(root / "annotations.json")
    scanner = DocumentScanner("models/corner_heatmap.pt", "models/enhance.pt")

    rows, tiles, t0 = [], [], time.time()
    for i in range(len(ds)):
        photo, _, item = ds.raw(i)
        stem = Path(item["file"]).name.split(".")[0]
        res = scanner.scan(photo)
        ours = res.image
        ref_rel = item.get("reference")
        ref = imread_rgb(root / ref_rel) if ref_rel and (root / ref_rel).exists() else None

        rec: dict = {"image": stem,
                     "aspect_ours": round(ours.shape[1] / ours.shape[0], 3),
                     "rotation": res.rotation_applied,
                     "source": res.corner_result.source if res.corner_result else "supplied",
                     "seconds": round(res.seconds, 2)}
        if ref is not None:
            rec["aspect_ref"] = round(ref.shape[1] / ref.shape[0], 3)
        if ocr_available():
            imgs = {"ours": ours} | ({"ref": ref} if ref is not None else {})
            rep = readability_report(imgs)
            for k, r in rep.items():
                rec[f"{k}_conf"] = round(r["confidence"], 1)
                rec[f"{k}_words"] = r["words"]

        verdict = "ok"
        ow, rw = rec.get("ours_words", 0), rec.get("ref_words", 0)
        if ow <= 2:
            verdict = "unread"
        elif ref is not None and abs(rec["aspect_ours"] - rec["aspect_ref"]) > ASPECT_TOL:
            verdict = "geometry"
        elif rw and ow < rw * WORD_RATIO_OK:
            verdict = "low-ocr"
        rec["verdict"] = verdict
        rows.append(rec)
        print(f"  {stem:8} {verdict:9} ours {ow:4d} words / {rec.get('ours_conf', 0):5.1f}   "
              f"ref {rw:4d} / {rec.get('ref_conf', 0):5.1f}   rot {rec['rotation']:>4}")

        h = 460
        panel = [cv2.resize(x, (int(round(x.shape[1] * h / x.shape[0])), h))
                 for x in ([photo, ours] + ([ref] if ref is not None else []))]
        names = [f"{stem} - photo", f"ours ({ow} words)"] + (
            [f"reference ({rw} words)"] if ref is not None else [])
        strip = [label_bar(p, n) for p, n in zip(panel, names)]
        hmax = max(s.shape[0] for s in strip)
        tiles.append(np.concatenate(
            [cv2.copyMakeBorder(s, 0, hmax - s.shape[0], 0, 8,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255)) for s in strip],
            axis=1))

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "counts": counts, "n": len(rows),
                               "seconds": round(time.time() - t0, 1)}, indent=1))

    cols = ["image", "verdict", "ours_words", "ours_conf", "ref_words", "ref_conf",
            "rotation", "source"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
    md += ["", "| verdict | photographs |", "|---|---|"]
    for k in sorted(counts):
        md.append(f"| {k} | **{counts[k]} / {len(rows)}** |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")

    if tiles:
        wmax = max(t.shape[1] for t in tiles)
        sheet = np.concatenate(
            [cv2.copyMakeBorder(t, 0, 10, 0, wmax - t.shape[1],
                                cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles],
            axis=0)
        p = Path(args.sheet)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
        print(f"\ncontact sheet -> {p}")

    print("\n".join(md[-len(counts) - 3:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
