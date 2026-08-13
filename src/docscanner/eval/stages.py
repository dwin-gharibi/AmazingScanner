from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.geometry import estimate_page_size, order_corners, quad_iou
from ..utils.imageio import imread_rgb, imwrite_rgb
from ..utils.viz import PALETTE, draw_corners, draw_quad, side_by_side

OUT = Path("outputs/stages")

__all__ = ["scan_stages", "run_set", "main"]


def _ocr(img: np.ndarray) -> dict[str, float]:
    try:
        from .ocr import ocr_available, run_ocr
        if not ocr_available():
            return {"confidence": float("nan"), "words": float("nan")}
        r = run_ocr(img)
        return {"confidence": float(r.mean_confidence), "words": float(r.words)}
    except Exception:
        return {"confidence": float("nan"), "words": float("nan")}


def scan_stages(scanner, photo: np.ndarray, out_dir: Path,
                gt_corners: np.ndarray | None = None,
                mode: str = "color", ocr: bool = True,
                reference: np.ndarray | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = photo.shape[:2]
    diag = float(np.hypot(h, w))

    t0 = time.time()
    res = scanner.scan(photo, mode=mode)
    total_seconds = time.time() - t0

    stages: list[dict[str, Any]] = []

    imwrite_rgb(out_dir / "1_input.jpg", photo, quality=92)
    stages.append({
        "stage": "1. input", "file": "1_input.jpg",
        "what": "the raw photograph, exactly as the camera produced it",
        "size": [w, h],
        "metrics": _ocr(photo) if ocr else {},
    })

    if gt_corners is not None:
        overlay = draw_corners(photo, gt=order_corners(gt_corners), pred=res.corners,
                               legend=("annotated", "predicted", ""))
    else:
        overlay = draw_quad(photo, res.corners, PALETTE["pred"])
    imwrite_rgb(out_dir / "2_detected.jpg", overlay, quality=92)

    detect: dict[str, Any] = {
        "seconds": round(res.timings.get("detect", 0.0), 4),
        "source": res.meta.get("corner_source", "network"),
        "confidence": round(float(res.meta.get("confidence", float("nan"))), 4),
        "edge_refined": bool(res.meta.get("refined", False)),
    }
    if gt_corners is not None:
        gt = order_corners(np.asarray(gt_corners, np.float32))
        err = float(np.linalg.norm(res.corners - gt, axis=1).mean())
        detect.update({
            "corner_error_px": round(err, 3),
            "corner_error_pct_diag": round(100 * err / max(diag, 1e-6), 3),
            "quad_iou": round(float(quad_iou(res.corners, gt)), 4),
            "all_within_16px": bool(np.linalg.norm(res.corners - gt, axis=1).max() <= 16),
        })
    stages.append({
        "stage": "2. detect", "file": "2_detected.jpg",
        "what": "four page corners, refined onto the page border",
        "metrics": detect,
    })

    imwrite_rgb(out_dir / "3_rectified.jpg", res.rectified, quality=92)
    rh, rw = res.rectified.shape[:2]
    rect: dict[str, Any] = {
        "seconds": round(res.timings.get("rectify", 0.0), 4),
        "page_size": [rw, rh],
        "aspect": round(max(rw, rh) / max(min(rw, rh), 1), 4),
    }
    if gt_corners is not None:
        gw, gh = estimate_page_size(order_corners(np.asarray(gt_corners, np.float32)),
                                    (h, w))
        want = max(gw, gh) / max(min(gw, gh), 1)
        rect["aspect_from_annotation"] = round(float(want), 4)
        rect["aspect_error_pct"] = round(100 * abs(rect["aspect"] - want) / max(want, 1e-6), 3)
    stages.append({
        "stage": "3. rectify", "file": "3_rectified.jpg",
        "what": "homography from those corners; the page flattened",
        "metrics": rect,
    })

    enhanced = res.image
    imwrite_rgb(out_dir / "4_enhanced.jpg", enhanced, quality=94)
    enh: dict[str, Any] = {"seconds": round(res.timings.get("enhance", 0.0), 4)}
    if ocr:
        before, after = _ocr(res.rectified), _ocr(enhanced)
        enh.update({
            "ocr_confidence_before": round(before["confidence"], 2),
            "ocr_confidence_after": round(after["confidence"], 2),
            "ocr_confidence_gain": round(after["confidence"] - before["confidence"], 2),
            "words_before": before["words"],
            "words_after": after["words"],
        })
    stages.append({
        "stage": "4. enhance", "file": "4_enhanced.jpg",
        "what": "the enhancement network: shadow, cast and blur removed",
        "metrics": enh,
    })

    imwrite_rgb(out_dir / "5_final.jpg", res.image, quality=94)
    stages.append({
        "stage": "5. finish", "file": "5_final.jpg",
        "what": f"auto-orientation and the '{mode}' output style",
        "metrics": {"seconds": round(res.timings.get("orient", 0.0), 4),
                    "rotation_applied_deg": int(res.rotation_applied),
                    "mode": mode},
    })

    panels = [photo, overlay, res.rectified, enhanced, res.image]
    titles = ["1. input", "2. detected", "3. rectified", "4. enhanced", "5. final"]
    if reference is not None:
        imwrite_rgb(out_dir / "6_reference.jpg", reference, quality=94)
        ref_metrics: dict[str, Any] = {}
        if ocr:
            ours, theirs = _ocr(res.image), _ocr(reference)
            ref_metrics = {
                "ocr_confidence_ours": round(ours["confidence"], 2),
                "ocr_confidence_commercial": round(theirs["confidence"], 2),
                "ocr_confidence_delta": round(ours["confidence"] - theirs["confidence"], 2),
                "words_ours": ours["words"],
                "words_commercial": theirs["words"],
                "at_least_as_readable": bool(
                    np.isfinite(ours["confidence"]) and np.isfinite(theirs["confidence"])
                    and ours["confidence"] >= theirs["confidence"]),
            }
        stages.append({
            "stage": "6. commercial baseline", "file": "6_reference.jpg",
            "what": "the same document from a scanning app -- a baseline, not a target",
            "metrics": ref_metrics,
        })
        panels.append(reference)
        titles.append("6. commercial app")

    strip = side_by_side(panels, titles, cell=300)
    imwrite_rgb(out_dir / "strip.jpg", strip, quality=92)

    summary: dict[str, Any] = {
        "stages": stages,
        "total": {
            "seconds": round(total_seconds, 3),
            "per_stage_seconds": {k: round(v, 4) for k, v in res.timings.items()},
        },
    }
    if ocr:
        a, b = _ocr(photo), _ocr(res.image)
        summary["total"].update({
            "ocr_confidence_raw_photo": round(a["confidence"], 2),
            "ocr_confidence_final": round(b["confidence"], 2),
            "ocr_confidence_gain": round(b["confidence"] - a["confidence"], 2),
            "words_raw_photo": a["words"], "words_final": b["words"],
        })
    if gt_corners is not None:
        summary["total"]["corner_error_px"] = detect.get("corner_error_px")

    (out_dir / "stages.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def stat(values: list[float]) -> dict[str, float] | None:
        vals = [v for v in values if isinstance(v, (int, float))
                and not isinstance(v, bool) and np.isfinite(v)]
        if not vals:
            return None
        return {"mean": round(float(np.mean(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "worst": round(float(np.max(vals)), 4), "n": len(vals)}

    out: dict[str, Any] = {"n_images": len(runs), "per_stage": {}, "total": {}}
    if not runs:
        return out
    for i, s in enumerate(runs[0]["stages"]):
        name = s["stage"]
        keys = {k for r in runs for k in r["stages"][i]["metrics"]}
        agg = {}
        for k in sorted(keys):
            st = stat([r["stages"][i]["metrics"].get(k) for r in runs])
            if st:
                agg[k] = st
        for k in sorted(keys):
            vals = [r["stages"][i]["metrics"].get(k) for r in runs]
            if any(isinstance(v, bool) for v in vals):
                agg[k] = {"rate_pct": round(100 * float(np.mean(
                    [bool(v) for v in vals if v is not None])), 2)}
        out["per_stage"][name] = {"what": s["what"], "metrics": agg}
    for k in sorted({k for r in runs for k in r["total"]}):
        st = stat([r["total"].get(k) for r in runs])
        if st:
            out["total"][k] = st
    return out


def run_set(limit: int | None = None, mode: str = "color",
            out_root: Path = OUT) -> dict[str, Any]:
    from ..data.datasets import RealPhotoSet
    from ..pipeline.scanner import DocumentScanner

    manifest = Path("data/real/own/annotations.json")
    if not manifest.exists():
        print("  [skip] no real photographs prepared")
        return {}
    enh, cor = Path("models/enhance.pt"), Path("models/corner_heatmap.pt")
    if not (enh.exists() and cor.exists()):
        print("  [skip] models/ is missing exported weights")
        return {}

    ds = RealPhotoSet(manifest)
    scanner = DocumentScanner(cor, enh, max_side=1100)
    n = len(ds) if limit is None else min(limit, len(ds))
    runs, n_ref = [], 0
    root = manifest.parent
    for i in range(n):
        img, gt, item = ds.raw(i)
        name = Path(item["file"]).stem
        ref = None
        if item.get("reference"):
            ref_path = root / item["reference"]
            if ref_path.exists():
                ref = imread_rgb(ref_path)
                n_ref += 1
        summary = scan_stages(scanner, img, out_root / name, gt_corners=gt,
                              mode=mode, reference=ref)
        runs.append(summary)
        det = summary["stages"][1]["metrics"]
        print(f"  [{i + 1:2d}/{n}] {name:28s} "
              f"corner {det.get('corner_error_px', float('nan')):7.2f} px  "
              f"IoU {det.get('quad_iou', float('nan')):.3f}  "
              f"{summary['total']['seconds']:.2f}s")

    agg = _aggregate(runs)
    agg["n_with_commercial_reference"] = n_ref
    if n_ref == 0:
        print("  note: no commercial reference scans found. Drop imgN_scanned.jpg\n"
              "        into data/real/own/reference/ and re-run `prepare --own`;\n"
              "        the baseline comparison then appears automatically.")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")
    _write_markdown(agg, out_root / "STAGES.md")
    return agg


def _write_markdown(agg: dict[str, Any], path: Path) -> None:
    if not agg.get("per_stage"):
        return
    lines = [
        "# Stage-by-stage accuracy",
        "",
        f"Every stage of the scanner, scored on its own, over "
        f"**{agg['n_images']} real photographs**. Each stage's output is written "
        f"as a separate image under `outputs/stages/<photo>/`.",
        "",
    ]
    for name, block in agg["per_stage"].items():
        lines += [f"## {name}", "", f"_{block['what']}_", ""]
        if not block["metrics"]:
            lines += ["No metric applies to this stage.", ""]
            continue
        lines += ["| metric | mean | median | worst | n |", "|---|---|---|---|---|"]
        for k, st in block["metrics"].items():
            if "rate_pct" in st:
                lines.append(f"| {k} | {st['rate_pct']:.1f}% | | | |")
            else:
                lines.append(f"| {k} | {st['mean']:.3f} | {st['median']:.3f} | "
                             f"{st['worst']:.3f} | {st['n']} |")
        lines.append("")
    lines += ["## Total", "", "| metric | mean | median | worst | n |",
              "|---|---|---|---|---|"]
    for k, st in agg["total"].items():
        lines.append(f"| {k} | {st['mean']:.3f} | {st['median']:.3f} | "
                     f"{st['worst']:.3f} | {st['n']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-stage inputs, outputs and accuracy")
    ap.add_argument("--image", default=None, help="one photograph instead of the set")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="color")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args(argv)

    print("[stages]")
    if args.image:
        from ..pipeline.scanner import DocumentScanner
        scanner = DocumentScanner("models/corner_heatmap.pt", "models/enhance.pt")
        out = Path(args.out) / Path(args.image).stem
        s = scan_stages(scanner, imread_rgb(args.image), out,
                        mode=args.mode, ocr=not args.no_ocr)
        print(json.dumps(s, indent=1))
        print(f"  wrote {out}")
        return 0

    agg = run_set(limit=args.limit, mode=args.mode, out_root=Path(args.out))
    if agg:
        print(json.dumps(agg["total"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
