from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.datasets import FrozenCornerSet, FrozenPairSet, RealPhotoSet
from ..data.prepare import DataNotPreparedError, load_splits, make_generator
from ..pipeline.corner_pipeline import CornerPipeline
from ..pipeline.enhance_pipeline import EnhancementPipeline
from ..pipeline.scanner import DocumentScanner
from ..utils.geometry import order_corners
from .metrics import corner_metrics, psnr, ssim, summarize_corner_errors
from .ocr import ocr_available, ocr_backends, readability_report

OUT = Path("outputs/report")
ASSETS = Path("docs/assets")

__all__ = ["evaluate_enhancement", "evaluate_corners", "evaluate_ocr",
           "evaluate_end_to_end", "markdown_table"]


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None,
                   floatfmt: str = "{:.4f}") -> str:
    if not rows:
        return "_(no rows)_"
    cols = list(columns or rows[0].keys())

    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "-"
            return floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _write(name: str, data: Any, markdown: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=float))
    if markdown is not None:
        (OUT / f"{name}.md").write_text(markdown)
    print(f"  wrote {OUT / name}.json" + (" (+.md)" if markdown else ""))


def _score_pairs(pipeline: EnhancementPipeline | None, pairs, limit: int | None = None,
                 label: str = "") -> dict[str, float]:
    ps, ss = [], []
    t0 = time.time()
    n = len(pairs) if limit is None else min(limit, len(pairs))
    for i in range(n):
        inp, tgt = pairs.page(i)
        out = inp if pipeline is None else pipeline(inp, mode="raw").network_output
        ps.append(psnr(out, tgt))
        ss.append(ssim(out, tgt))
    return {"n": n, "PSNR": float(np.mean(ps)), "SSIM": float(np.mean(ss)),
            "seconds/page": (time.time() - t0) / max(n, 1)}


def _train_split_pairs(n: int, seed: int = 9090, page_long_side: int = 768):
    from ..data.datasets import freeze_pairs
    from ..data.degrade import DegradationConfig
    root = Path("data/frozen/enhance_train_probe")
    if not (root / "manifest.json").exists():
        gen = make_generator("train", cfg=DegradationConfig.for_enhancement(),
                             splits=load_splits(), photo_long_side=1280)
        freeze_pairs(gen, root, n=n, seed=seed, page_long_side=page_long_side)
    return FrozenPairSet(root)


def evaluate_enhancement(checkpoint: str | Path, limit: int | None = None,
                         name: str = "enhance_main") -> dict:
    print(f"[enhancement] {checkpoint}")
    pipe = EnhancementPipeline(checkpoint, max_side=1600)

    sets = {}
    try:
        sets["train"] = _train_split_pairs(24)
    except DataNotPreparedError:
        print("  [skip] train split: datasets not built (frozen sets only)")
    for split, root in (("validation", "enhance_val"), ("test", "enhance_test"),
                        ("test (course scans only)", "enhance_test_course"),
                        ("pseudo-real (harder, OOD)", "pseudo_real")):
        p = Path("data/frozen") / root
        if (p / "manifest.json").exists():
            sets[split] = FrozenPairSet(p)
        else:
            print(f"  [skip] {split}: data/frozen/{root} not built")
    if not sets:
        print("  [skip] enhancement: no pair sets present at all")
        return {"rows": []}

    rows = []
    if "test" in sets:
        baseline_test = _score_pairs(None, sets["test"], limit, "baseline")
        rows.append({"Split": "test - degraded input (no model)", **baseline_test})
    for split, ds in sets.items():
        base = _score_pairs(None, ds, limit)
        got = _score_pairs(pipe, ds, limit)
        rows.append({
            "Split": split,
            "n": got["n"],
            "PSNR": got["PSNR"],
            "SSIM": got["SSIM"],
            "PSNR (input)": base["PSNR"],
            "SSIM (input)": base["SSIM"],
            "dPSNR": got["PSNR"] - base["PSNR"],
            "dSSIM": got["SSIM"] - base["SSIM"],
            "seconds/page": got["seconds/page"],
        })

    cols = ["Split", "n", "PSNR", "SSIM", "PSNR (input)", "SSIM (input)",
            "dPSNR", "dSSIM", "seconds/page"]
    md = markdown_table(rows, cols)
    _write(f"enhancement_{name}", {"checkpoint": str(checkpoint), "rows": rows}, md)
    print(md)
    return {"rows": rows}


def compare_enhancement_runs(runs: dict[str, str | Path], limit: int = 24) -> dict:
    print("[enhancement ablation]")
    test = FrozenPairSet(Path("data/frozen/enhance_test"))
    rows = [{"Run": "degraded input (no model)", **_score_pairs(None, test, limit)}]
    for name, ckpt in runs.items():
        if not Path(ckpt).exists():
            print(f"  [skip] {name}: {ckpt} not found")
            continue
        pipe = EnhancementPipeline(ckpt, max_side=1600)
        r = _score_pairs(pipe, test, limit)
        cfg = pipe.ckpt_meta.get("config", {}).get("extra", {})
        rows.append({"Run": name, "loss": cfg.get("loss", "-"),
                     "bg prior": cfg.get("bg_prior", "-"),
                     "dropout": cfg.get("dropout", 0.0), **r})
    md = markdown_table(rows, ["Run", "loss", "bg prior", "dropout", "n", "PSNR", "SSIM"])
    _write("enhancement_ablation", {"rows": rows}, md)
    print(md)
    return {"rows": rows}


@torch.no_grad()
def _corner_predictions(pipe: CornerPipeline, dataset, use_refine: bool,
                        use_tta: bool = False,
                        limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds, gts, sizes = [], [], []
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(n):
        img, gt = dataset.raw(i)[:2]
        res = pipe(img, refine=use_refine, tta=use_tta)
        preds.append(res.corners)
        gts.append(order_corners(gt))
        sizes.append((img.shape[1], img.shape[0]))
    return np.asarray(preds), np.asarray(gts), np.asarray(sizes)


CORNER_VARIANTS: tuple[tuple[bool, bool], ...] = (
    (False, False), (True, False), (False, True), (True, True))


def evaluate_corners(checkpoints: dict[str, str | Path], limit: int | None = None,
                     refine_variants: tuple[tuple[bool, bool], ...] = CORNER_VARIANTS,
                     out_name: str = "corners") -> dict:
    print("[corners]")
    datasets = {}
    for ds_name, root in (("synthetic test", "corners_test"),
                          ("synthetic test (course only)", "corners_test_course"),
                          ("pseudo-real (OOD)", "pseudo_real_corners")):
        p = Path("data/frozen") / root
        if (p / "annotations.json").exists():
            datasets[ds_name] = FrozenCornerSet(p, size=512)
        else:
            print(f"  [skip] {ds_name}: data/frozen/{root} not built")
    real_manifest = Path("data/real/midv500/annotations.json")
    if real_manifest.exists():
        datasets["real photos (MIDV-500)"] = RealPhotoSet(real_manifest)
    own = Path("data/real/own/annotations.json")
    if own.exists():
        datasets["real photos (own)"] = RealPhotoSet(own)

    rows, detail = [], {}
    for name, ckpt in checkpoints.items():
        if not Path(ckpt).exists():
            print(f"  [skip] {name}: {ckpt} not found")
            continue
        pipe = CornerPipeline(ckpt, refine=True)
        for ds_name, ds in datasets.items():
            for use_refine, use_tta in refine_variants:
                preds, gts, sizes = _corner_predictions(pipe, ds, use_refine,
                                                        use_tta, limit)
                sc = corner_metrics(preds, gts, sizes)
                row = {"Model": name, "Set": ds_name,
                       "edge refine": "yes" if use_refine else "no",
                       "TTA": "yes" if use_tta else "no",
                       **sc.as_row()}
                rows.append(row)
                key = f"{name}|{ds_name}|{use_refine}|{use_tta}"
                detail[key] = {
                    "per_corner_px": summarize_corner_errors(preds, gts).tolist(),
                    "per_image_px": sc.per_image.tolist(),
                }
                print(f"  {name:26s} {ds_name:24s} refine={int(use_refine)} "
                      f"tta={int(use_tta)} "
                      f"MCE={sc.mce_px:6.2f}px  succ@16={sc.success[16.0] * 100:5.1f}%  "
                      f"IoU={sc.iou:.3f}")

    cols = ["Model", "Set", "edge refine", "TTA", "n", "MCE (px)", "median (px)",
            "MCE (% diag)", "quad IoU", "success@8px", "success@16px",
            "success@32px", "worst (px)"]
    md = markdown_table(rows, cols)
    _write(out_name, {"rows": rows, "detail": detail}, md)
    return {"rows": rows, "detail": detail}



def _merge_corner_tables(*tables: dict) -> dict:
    rows, detail = [], {}
    for t in tables:
        if not t:
            continue
        rows += t.get("rows", [])
        detail.update(t.get("detail", {}))
    cols = ["Model", "Set", "edge refine", "n", "MCE (px)", "median (px)",
            "MCE (% diag)", "quad IoU", "success@8px", "success@16px",
            "success@32px", "worst (px)"]
    md = markdown_table(rows, cols)
    _write("corners", {"rows": rows, "detail": detail}, md)
    return {"rows": rows, "detail": detail}


def evaluate_ocr(checkpoint: str | Path, limit: int = 20) -> dict:
    if not ocr_available():
        print("[ocr] no OCR engine - skipped "
              "(pip install easyocr, or apt-get install tesseract-ocr)")
        return {"available": False}
    engines = ocr_backends()
    print(f"[ocr] engine: {engines[0]} (available: {', '.join(engines)})")
    pipe = EnhancementPipeline(checkpoint, max_side=1600)
    test = FrozenPairSet(Path("data/frozen/enhance_test"))

    acc: dict[str, list[dict[str, float]]] = {"degraded input": [], "enhanced": [],
                                              "clean target": []}
    n = min(limit, len(test))
    for i in range(n):
        inp, tgt = test.page(i)
        out = pipe(inp, mode="raw").network_output
        rep = readability_report(
            {"degraded input": inp, "enhanced": out, "clean target": tgt},
            reference_key="clean target")
        for k, v in rep.items():
            acc[k].append(v)

    rows = []
    for k, vals in acc.items():
        if not vals:
            continue
        rows.append({
            "Image": k,
            "n": len(vals),
            "OCR confidence": float(np.mean([v["confidence"] for v in vals])),
            "words read": float(np.mean([v["words"] for v in vals])),
            "CER vs clean scan": float(np.nanmean([v.get("cer", np.nan) for v in vals])),
            "WER vs clean scan": float(np.nanmean([v.get("wer", np.nan) for v in vals])),
        })
    md = markdown_table(rows)
    _write("ocr", {"rows": rows, "engine": engines[0],
                   "engines_available": list(engines)}, md)
    print(md)
    return {"rows": rows, "engine": engines[0]}


def evaluate_end_to_end(corner_ckpt: str | Path, enhance_ckpt: str | Path,
                        limit: int = 24) -> dict:
    print("[end-to-end]")
    manifest = Path("data/real/own/annotations.json")
    if not manifest.exists():
        manifest = Path("data/real/midv500/annotations.json")
    if not manifest.exists():
        print("  [skip] no annotated real photos found")
        return {}

    ds = RealPhotoSet(manifest)
    scanner = DocumentScanner(corner_ckpt, enhance_ckpt)
    n = min(limit, len(ds))

    rows: list[dict[str, Any]] = []
    per_image = []
    for i in range(n):
        photo, gt_corners, item = ds.raw(i)
        gt_scan = scanner.scan(photo, corners=gt_corners, auto_orientation=False)
        pred_scan = scanner.scan(photo, auto_orientation=False)
        err = float(np.linalg.norm(pred_scan.corners - gt_corners, axis=1).mean())

        entry: dict[str, Any] = {"i": i, "corner_error_px": err,
                                 "seconds": pred_scan.seconds}
        if ocr_available():
            rep = readability_report({
                "annotated corners": gt_scan.image,
                "predicted corners": pred_scan.image,
            })
            entry["conf_gt"] = rep["annotated corners"]["confidence"]
            entry["conf_pred"] = rep["predicted corners"]["confidence"]
            entry["words_gt"] = rep["annotated corners"]["words"]
            entry["words_pred"] = rep["predicted corners"]["words"]
        per_image.append(entry)

    def _mean(key):
        vals = [e[key] for e in per_image if key in e and not np.isnan(e[key])]
        return float(np.mean(vals)) if vals else float("nan")

    rows.append({"Rectified with": "annotated corners (ground truth)",
                 "n": n, "corner error (px)": 0.0,
                 "OCR confidence": _mean("conf_gt"), "words read": _mean("words_gt")})
    rows.append({"Rectified with": "predicted corners (fully automatic)",
                 "n": n, "corner error (px)": _mean("corner_error_px"),
                 "OCR confidence": _mean("conf_pred"), "words read": _mean("words_pred")})

    md = markdown_table(rows)
    _write("end_to_end", {"rows": rows, "per_image": per_image}, md)
    print(md)
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate trained models")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--models", default="models",
                    help="where the shipped weights live; the report describes\n"
                         "these, because they are what a user actually runs")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--enhancement", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--corners", action="store_true")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--end-to-end", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", choices=["all", "shipped"], default="all",
                    help="corner (refine, TTA) combinations to measure: 'all' "
                         "climbs the full inference ladder for the report; "
                         "'shipped' measures only the configuration users get "
                         "(refine off, TTA on) -- what CI's accuracy guard "
                         "bounds, at a quarter of the cost")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    if not any([args.all, args.enhancement, args.ablation, args.corners,
                args.ocr, args.end_to_end]):
        args.all = True

    torch.set_num_threads(args.threads)
    runs = Path(args.runs)

    from ..engine.export_models import CANDIDATES, _newer_capability, _would_regress, pick_best

    def _flagship(key: str, fallback: str) -> Path | None:
        cands = tuple(str(Path(args.runs) / Path(c).relative_to("runs"))
                      for c in CANDIDATES[key])
        picked, why = pick_best(cands, name=key, runs_root=args.runs)
        shipped = Path(args.models) / f"{key}.pt"
        if (shipped.exists() and picked is not None
                and (_would_regress(picked, shipped)
                     or _newer_capability(shipped, picked))):
            print(f"  [{key}] {shipped}  (the shipped weights; "
                  f"{picked.parent.name} in {args.runs} is older or weaker)")
            return shipped
        if picked is not None:
            print(f"  [{key}] {picked}  ({why})")
            return picked
        legacy = runs / fallback / "best.pt"
        if legacy.exists():
            return legacy
        shipped = Path("models") / f"{key}.pt"
        if shipped.exists():
            print(f"  [{key}] {shipped}  (no training run here; the shipped weights)")
            return shipped
        print(f"  [{key}] no checkpoint anywhere — skipping what needs it")
        return None

    enh = _flagship("enhance", "enhance_main")
    heat = _flagship("corner_heatmap", "corner_heatmap")
    reg = _flagship("corner_regression", "corner_regression")

    if not any((enh, heat, reg)):
        print("\nNothing to evaluate: no training runs and no exported weights.\n"
              "  Train first:  amazingscanner all\n"
              "  Or export an existing run:  amazingscanner export")
        return 1

    if (args.all or args.enhancement) and enh:
        evaluate_enhancement(enh, limit=args.limit)
    if (args.all or args.ablation) and enh:
        compare_enhancement_runs({
            "flagship (full budget)": enh,
            "control (combined loss, no dropout)": runs / "abl_enhance_ctrl" / "best.pt",
            "+ dropout 0.15 (Section 6)": runs / "abl_enhance_drop" / "best.pt",
            "loss: MSE": runs / "abl_loss_mse" / "best.pt",
            "loss: L1": runs / "abl_loss_l1" / "best.pt",
            "with background prior": runs / "abl_bg_prior" / "best.pt",
        }, limit=args.limit or 24)
    if (args.all or args.corners) and (heat or reg):
        pair = {k: v for k, v in (("A: regression", reg), ("B: heatmap", heat)) if v}
        variants = (CORNER_VARIANTS if args.variants == "all"
                    else ((False, True),))
        flagship = evaluate_corners(pair,
                                    limit=args.limit,
                                    refine_variants=variants)
        if args.variants == "all":
            ablation = evaluate_corners({
                "A: regression (control)": runs / "abl_reg_ctrl" / "best.pt",
                "A: regression + dropout": runs / "abl_reg_drop" / "best.pt",
                "B: heatmap (control)": runs / "abl_heat_ctrl" / "best.pt",
                "B: heatmap + dropout": runs / "abl_heat_drop" / "best.pt",
            }, limit=args.limit, refine_variants=((True, False),),
                out_name="corners_dropout")
            _merge_corner_tables(flagship, ablation)
            evaluate_corners({
                "B: heatmap + page-mask head": runs / "abl_seg_on" / "best.pt",
                "B: heatmap, no mask head": runs / "abl_seg_off" / "best.pt",
            }, limit=args.limit, refine_variants=((True, False),),
                out_name="corners_seg_head")
    if (args.all or args.ocr) and enh:
        evaluate_ocr(enh, limit=args.limit or 20)
    if (args.all or args.end_to_end) and heat and enh:
        evaluate_end_to_end(heat, enh, limit=args.limit or 24)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
