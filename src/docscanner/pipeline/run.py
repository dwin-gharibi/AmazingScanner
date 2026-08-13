from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..utils.imageio import imread_rgb, imwrite_rgb
from ..utils.viz import PALETTE, draw_quad, side_by_side

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
DEFAULT_ENHANCE = ["models/enhance.pt", "runs/enhance_main/best.pt"]
DEFAULT_CORNERS = ["models/corner_heatmap.pt", "runs/corner_heatmap/best.pt",
                   "models/corner_regression.pt", "runs/corner_regression/best.pt"]


def _resolve(explicit: str | None, candidates: list[str], what: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"checkpoint not found: {p}")
        return p
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    sys.exit(f"no {what} checkpoint found (looked in: {', '.join(candidates)}).\n"
             f"Train one with  python -m docscanner.engine.train_{what} ...")


def _inputs(path: str) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        if not files:
            sys.exit(f"no images found under {p}")
        return files
    if not p.exists():
        sys.exit(f"input not found: {p}")
    return [p]


def cmd_enhance(args) -> int:
    from .enhance_pipeline import EnhancementPipeline

    ckpt = _resolve(args.checkpoint, DEFAULT_ENHANCE, "enhance")
    pipe = EnhancementPipeline(ckpt, max_side=args.max_side, threads=args.threads,
                               preserve_tone=args.preserve_tone)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for path in _inputs(args.input):
        img = imread_rgb(path)
        res = pipe(img, mode=args.mode)
        stem = path.stem
        imwrite_rgb(out_dir / f"{stem}_enhanced.png", res.image)
        imwrite_rgb(out_dir / f"{stem}_compare.jpg",
                    side_by_side([res.input_image, res.image],
                                 ["input (rectified)", f"enhanced ({args.mode})"],
                                 cell=700), quality=93)
        records.append({"input": str(path), "seconds": res.seconds,
                        "size": list(res.image.shape[:2][::-1]), "mode": args.mode})
        print(f"  {path.name}: {res.seconds:.2f}s -> {out_dir / f'{stem}_enhanced.png'}")

    (out_dir / "enhance_report.json").write_text(json.dumps(records, indent=1))
    return 0


def cmd_corners(args) -> int:
    from .corner_pipeline import CornerPipeline

    ckpt = _resolve(args.checkpoint, DEFAULT_CORNERS, "corners")
    pipe = CornerPipeline(ckpt, threads=args.threads,
                          refine=bool(getattr(args, "refine", False)),
                          tta=not args.no_tta)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for path in _inputs(args.input):
        img = imread_rgb(path)
        res = pipe(img)
        stem = path.stem
        overlay = draw_quad(img, res.corners, PALETTE["pred"])
        imwrite_rgb(out_dir / f"{stem}_corners.jpg", overlay, quality=93)
        if res.heatmaps is not None:
            from ..utils.viz import heatmap_overlay
            imwrite_rgb(out_dir / f"{stem}_heatmap.jpg",
                        heatmap_overlay(img, res.heatmaps.max(axis=0)), quality=90)
        records.append({
            "input": str(path), "approach": res.approach,
            "corners_tl_tr_br_bl": np.round(res.corners, 2).tolist(),
            "confidence": res.confidence, "edge_refined": res.refined,
            "source": res.source, "tta": res.meta.get("tta", False),
            "seconds": res.seconds,
        })
        print(f"  {path.name}: {res.approach}, conf={res.confidence:.2f}, "
              f"refined={res.refined}, source={res.source}, "
              f"{res.seconds * 1000:.0f}ms")

    (out_dir / "corners_report.json").write_text(json.dumps(records, indent=1))
    return 0


def cmd_scan(args) -> int:
    from .scanner import DocumentScanner, save_pdf

    enh = _resolve(args.checkpoint, DEFAULT_ENHANCE, "enhance")
    cor = _resolve(args.corner_checkpoint, DEFAULT_CORNERS, "corners")
    scanner = DocumentScanner(cor, enh, threads=args.threads, max_side=args.max_side,
                              refine_corners=bool(getattr(args, "refine", False)),
                              tta=not args.no_tta,
                              preserve_tone=getattr(args, "preserve_tone", 0.0))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, pages = [], []

    for path in _inputs(args.input):
        img = imread_rgb(path)
        t0 = time.time()
        res = scanner.scan(img, mode=args.mode, auto_orientation=not args.no_orient)
        stem = path.stem
        imwrite_rgb(out_dir / f"{stem}_scan.png", res.image)
        imwrite_rgb(out_dir / f"{stem}_steps.jpg", side_by_side(
            [draw_quad(res.photo, res.corners, PALETTE["pred"]), res.rectified, res.image],
            ["1. detected page", "2. rectified", f"3. enhanced ({args.mode})"],
            cell=560), quality=93)
        pages.append(res.image)
        records.append({
            "input": str(path),
            "corners_tl_tr_br_bl": np.round(res.corners, 2).tolist(),
            "confidence": res.meta.get("confidence"),
            "rotation_applied": res.rotation_applied,
            "timings": res.timings, "seconds": time.time() - t0,
        })
        print(f"  {path.name}: {res.seconds:.2f}s "
              f"(detect {res.timings.get('detect', 0) * 1000:.0f}ms, "
              f"enhance {res.timings.get('enhance', 0):.2f}s) "
              f"-> {out_dir / f'{stem}_scan.png'}")

    if args.pdf and pages:
        p = save_pdf(pages, out_dir / args.pdf)
        print(f"  PDF -> {p}")
    (out_dir / "scan_report.json").write_text(json.dumps(records, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="docscanner.pipeline.run",
        description="Run a trained pipeline on an image or a folder of images",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("input", help="image file or directory")
        p.add_argument("-o", "--output", default="outputs/pipeline")
        p.add_argument("-c", "--checkpoint", default=None)
        p.add_argument("--threads", type=int, default=4)

    def _tone_flag(p):
        p.add_argument("--preserve-tone", type=float, default=0.0, metavar="0..1",
                       help="how much of a non-paper page's own colour and tone "
                            "to keep (default 1.0). Pass 0 for the plain network "
                            "output: every page whitened, book covers included, "
                            "and 10-14.5%% of pixels clipped to pure white on the "
                            "real photographs")

    p_enh = sub.add_parser("enhance", help="enhance an already rectified document")
    common(p_enh)
    p_enh.add_argument("--mode", default="color",
                       choices=["color", "gray", "bw", "whiteboard", "raw"])
    p_enh.add_argument("--max-side", type=int, default=1600)
    _tone_flag(p_enh)
    p_enh.set_defaults(func=cmd_enhance)

    p_cor = sub.add_parser("corners", help="detect the page corners in a raw photo")
    common(p_cor)
    p_cor.add_argument("--refine", action="store_true",
                       help="classical sub-pixel edge refinement. Off by "
                            "default: it helps synthetic pages and costs ~2px "
                            "on real photographs, where the page border "
                            "competes with bindings, desk edges and shadows")
    p_cor.add_argument("--no-refine", action="store_true",
                       help=argparse.SUPPRESS)
    p_cor.add_argument("--no-tta", action="store_true",
                       help="single view instead of four quarter turns "
                            "(~3x faster; costs ~5%% mean accuracy on real "
                            "photographs, but slightly *fewer* gross misses)")
    p_cor.set_defaults(func=cmd_corners)

    p_scan = sub.add_parser("scan", help="end-to-end: raw photo -> clean page")
    common(p_scan)
    p_scan.add_argument("--corner-checkpoint", default=None)
    p_scan.add_argument("--mode", default="color",
                        choices=["color", "gray", "bw", "whiteboard", "raw"])
    p_scan.add_argument("--max-side", type=int, default=1600)
    _tone_flag(p_scan)
    p_scan.add_argument("--pdf", default=None, help="also write all pages to this PDF")
    p_scan.add_argument("--refine", action="store_true",
                        help="classical edge refinement (off by default; see "
                             "`corners --help`)")
    p_scan.add_argument("--no-refine", action="store_true",
                        help=argparse.SUPPRESS)
    p_scan.add_argument("--no-tta", action="store_true",
                        help="single-view corner detection (~3x faster detect)")
    p_scan.add_argument("--no-orient", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
