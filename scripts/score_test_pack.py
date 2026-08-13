#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


def _score_pack(pack: Path, threads: int | None):
    import cv2
    import numpy as np
    import torch
    if threads:
        torch.set_num_threads(threads)

    from docscanner.eval.metrics import match_cyclic, psnr, ssim
    from docscanner.pipeline.corner_pipeline import CornerPipeline
    from docscanner.pipeline.enhance_pipeline import EnhancementPipeline
    from docscanner.utils.geometry import order_corners, rectify
    from docscanner.utils.imageio import imread_rgb

    manifest = json.loads((pack / "ground_truth" / "corners.json").read_text())
    corners_pipe = CornerPipeline("models/corner_heatmap.pt")
    enhancer = EnhancementPipeline("models/enhance.pt", max_side=1400)

    items: list[dict] = []
    per_style: dict[str, dict[str, list]] = defaultdict(
        lambda: {"err_pct": [], "iou": [], "psnr": [], "ssim": [], "sources": []})
    t0 = time.time()
    for it in manifest["items"]:
        img = imread_rgb(pack / it["file"])
        h, w = img.shape[:2]
        diag = float(np.hypot(h, w))
        gt = order_corners(np.asarray(it["corners"], np.float32))

        res = corners_pipe(img)
        p = order_corners(np.asarray(res.corners, np.float32))
        gm = match_cyclic(p[None], gt[None])[0]
        err = float(np.linalg.norm(p - gm, axis=1).mean())
        buf_p = np.zeros((256, 256), np.uint8)
        buf_g = np.zeros((256, 256), np.uint8)
        cv2.fillConvexPoly(buf_p, np.round(p / [w, h] * 255).astype(np.int32), 1)
        cv2.fillConvexPoly(buf_g, np.round(gm / [w, h] * 255).astype(np.int32), 1)
        union = int((buf_p | buf_g).sum())
        iou = float((buf_p & buf_g).sum() / union) if union else 0.0

        page = rectify(img, gt, max_side=1400)[0]
        out = enhancer(page, mode="raw").network_output
        tgt = imread_rgb(pack / it["target"])
        if out.shape != tgt.shape:
            out = cv2.resize(out, (tgt.shape[1], tgt.shape[0]))

        rec = {"file": it["file"], "style": it["style"],
               "err_pct": err / diag * 100, "iou": iou,
               "psnr": psnr(out, tgt), "ssim": ssim(out, tgt),
               "source": res.source}
        items.append(rec)
        s = per_style[it["style"]]
        for k in ("err_pct", "iou", "psnr", "ssim"):
            s[k].append(rec[k])
        s["sources"].append(res.source)

    rows = []
    for style, s in sorted(per_style.items()):
        e = np.asarray(s["err_pct"])
        rows.append({
            "Style": style,
            "n": int(len(e)),
            "corner err (% diag)": float(e.mean()),
            "median (% diag)": float(np.median(e)),
            "success@2%": float((e < 2).mean() * 100),
            "success@4%": float((e < 4).mean() * 100),
            "quad IoU": float(np.mean(s["iou"])),
            "PSNR": float(np.mean(s["psnr"])),
            "SSIM": float(np.mean(s["ssim"])),
            "detected by network": f"{s['sources'].count('network')}/{len(e)}",
        })
    all_e = np.concatenate([np.asarray(s["err_pct"]) for s in per_style.values()])
    rows.append({
        "Style": "ALL", "n": int(len(all_e)),
        "corner err (% diag)": float(all_e.mean()),
        "median (% diag)": float(np.median(all_e)),
        "success@2%": float((all_e < 2).mean() * 100),
        "success@4%": float((all_e < 4).mean() * 100),
        "quad IoU": float(np.mean([v for s in per_style.values() for v in s["iou"]])),
        "PSNR": float(np.mean([v for s in per_style.values() for v in s["psnr"]])),
        "SSIM": float(np.mean([v for s in per_style.values() for v in s["ssim"]])),
        "detected by network":
            f"{sum(s['sources'].count('network') for s in per_style.values())}/{len(all_e)}",
    })
    return {"pack": str(pack), "count": int(len(all_e)),
            "seconds": round(time.time() - t0, 1), "rows": rows, "items": items}


def _pick_exhibits(payload: dict) -> list[tuple[str, dict]]:
    import numpy as np

    by_style: dict[str, list[dict]] = defaultdict(list)
    for rec in payload["items"]:
        by_style[rec["style"]].append(rec)
    picks = []
    for style in sorted(by_style):
        recs = sorted(by_style[style], key=lambda r: r["err_pct"])
        picks.append(("median", recs[int(np.floor((len(recs) - 1) / 2))]))
        picks.append(("hardest", recs[-1]))
    return picks


def render_figure(pack: Path, payload: dict, path: Path) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from docscanner.pipeline.scanner import DocumentScanner
    from docscanner.utils.imageio import imread_rgb

    scanner = DocumentScanner("models/corner_heatmap.pt", "models/enhance.pt")
    gt_by_file = {it["file"]: it["corners"]
                  for it in json.loads(
                      (pack / "ground_truth" / "corners.json").read_text())["items"]}

    picks = _pick_exhibits(payload)
    fig, axes = plt.subplots(3, 2, figsize=(15, 15.5))
    for ax, (tag, rec) in zip(axes.ravel(), picks):
        img = imread_rgb(pack / rec["file"])
        result = scanner.scan(img)
        shown = img.copy()
        for quad, colour in ((gt_by_file[rec["file"]], (46, 204, 113)),
                             (result.corners, (231, 76, 60))):
            pts = np.round(np.asarray(quad)).astype(np.int32)
            cv2.polylines(shown, [pts], True, colour,
                          max(2, int(0.004 * max(img.shape[:2]))))
        h = 640
        pair = [cv2.resize(x, (int(round(x.shape[1] * h / x.shape[0])), h))
                for x in (shown, result.image)]
        gap = np.full((h, 14, 3), 255, np.uint8)
        ax.imshow(np.concatenate([pair[0], gap, pair[1]], axis=1))
        ax.set_title(f"{rec['style']} — {tag} · corner err {rec['err_pct']:.2f}% "
                     f"of diagonal · quad IoU {rec['iou']:.2f}", fontsize=11)
        ax.axis("off")
    fig.suptitle("The unseen pack, end to end — green truth, red detection; "
                 "median and hardest photo of each style", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)
    print(f"figure -> {path}")


def render_chart(payload: dict) -> None:
    from docscanner.eval.charts import chart_pack_success
    chart_pack_success()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a generated test pack")
    ap.add_argument("--pack", default="data/test_pack")
    ap.add_argument("--out", default="outputs/report/test_pack.json")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--figure", default=None,
                    help="also render a photo-to-scan montage JPEG here "
                         "(for example: docs/assets/pack_benchmark.jpg)")
    ap.add_argument("--chart", action="store_true",
                    help="also redraw docs/assets/charts/pack_success.png")
    ap.add_argument("--from-json", default=None,
                    help="reuse a previous run's JSON (with items) instead of rescoring")
    args = ap.parse_args(argv)

    pack = Path(args.pack)
    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text())
        if "items" not in payload:
            ap.error("--from-json file has no per-item records; rescore instead")
    else:
        payload = _score_pack(pack, args.threads)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=1))

        rows = payload["rows"]
        cols = list(rows[0])
        md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            md.append("| " + " | ".join(
                f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])
                for c in cols) + " |")
        out_path.with_suffix(".md").write_text("\n".join(md) + "\n")
        print("\n".join(md))
        print(f"\n{payload['count']} photographs scored in "
              f"{payload['seconds']}s -> {out_path}")

    if args.figure:
        render_figure(pack, payload, Path(args.figure))
    if args.chart:
        render_chart(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
