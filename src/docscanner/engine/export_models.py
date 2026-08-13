from __future__ import annotations

import argparse
from pathlib import Path

import torch

CANDIDATES: dict[str, tuple[str, ...]] = {
    "enhance": ("runs/enhance_main_long/best.pt", "runs/enhance_main/best.pt",
                "runs/abl_loss_l1/best.pt", "runs/abl_loss_mse/best.pt",
                "runs/abl_enhance_ctrl/best.pt", "runs/abl_enhance_drop/best.pt",
                "runs/abl_no_bg_prior/best.pt"),
    "corner_heatmap": ("runs/corner_heatmap_v3/best.pt",
                       "runs/corner_heatmap_v2/best.pt",
                       "runs/corner_heatmap_long/best.pt",
                       "runs/corner_heatmap/best.pt",
                       "runs/abl_heat_drop/best.pt",
                       "runs/abl_heat_ctrl/best.pt"),
    "corner_regression": ("runs/corner_regression_long/best.pt",
                          "runs/corner_regression/best.pt",
                          "runs/abl_reg_ctrl/best.pt",
                          "runs/abl_reg_drop/best.pt"),
}

METRICS = {"val_ssim": True, "val_psnr": True, "val_mce_px": False}

PREFERRED: dict[str, tuple[str, str]] = {
    "corner_heatmap": (
        "runs/corner_heatmap_v3/best.pt",
        "41.96 px on the 24 real photographs vs 61.15 for the best frozen-set "
        "score; the frozen set has no distractor pages so it cannot measure this"),
}


def _score(path: Path) -> tuple[str, float] | None:
    import json

    hist = path.parent / "history.json"
    if not hist.exists():
        return None
    try:
        data = json.loads(hist.read_text())
    except (OSError, ValueError):
        return None
    epochs = data if isinstance(data, list) else data.get("epochs", [])
    for key, higher_better in METRICS.items():
        values = [e[key] for e in epochs
                  if isinstance(e, dict) and isinstance(e.get(key), (int, float))]
        if values:
            return key, (max(values) if higher_better else min(values))
    return None


def trained_at_all(path: Path) -> tuple[bool, str]:
    import json

    hist = path.parent / "history.json"
    if not hist.exists():
        return True, ""
    try:
        rows = json.loads(hist.read_text())
    except (OSError, ValueError):
        return True, ""
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return True, ""

    last = rows[-1]
    if "val_psnr" in last and "val_psnr_input" in last:
        gain = float(last["val_psnr"]) - float(last["val_psnr_input"])
        if gain <= 0.05:
            return False, (f"val_psnr {last['val_psnr']:.4f} does not beat the "
                           f"untouched input ({last['val_psnr_input']:.4f}); "
                           "this checkpoint is the identity function")

    losses = [r["train_loss"] for r in rows if isinstance(r.get("train_loss"), (int, float))]
    if len(losses) >= 3 and max(losses) - min(losses) < 1e-6:
        return False, (f"train_loss never changed across {len(losses)} epochs "
                       f"(constant {losses[0]:.6f}); no optimiser step landed")
    return True, ""


def _has_mask_head(path: Path) -> bool:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return False
    state = ckpt.get("ema") or ckpt.get("model") or {}
    return any(str(k).startswith("seg.") for k in state)


def _newer_capability(a: Path, b: Path) -> bool:
    return _has_mask_head(a) and not _has_mask_head(b)


def _usable_mask_head(path: Path) -> bool:
    from ..pipeline.corner_pipeline import DEFAULT_MIN_MASK_IOU

    if not _has_mask_head(path):
        return False
    try:
        extra = torch.load(str(path), map_location="cpu",
                           weights_only=False).get("extra", {})
    except Exception:
        return False
    iou = extra.get("val_mask_iou")
    return isinstance(iou, (int, float)) and float(iou) >= DEFAULT_MIN_MASK_IOU


def _outranked_by_newer_capability(preferred: Path,
                                   candidates: tuple[str, ...]) -> bool:
    if _has_mask_head(preferred):
        return False
    for rel in candidates:
        p = Path(rel)
        if p.exists() and p != preferred and _has_mask_head(p):
            print(f"  [note] preference for {preferred.parent.name} stands down: "
                  f"{p.parent.name} has a page-mask head, which is what the "
                  f"preference was measuring")
            return True
    return False


def pick_best(candidates: tuple[str, ...], name: str | None = None,
              runs_root: str | Path = "runs") -> tuple[Path | None, str]:
    if name and name in PREFERRED:
        rel, why = PREFERRED[name]
        p = Path(runs_root) / Path(rel).relative_to("runs")
        if p.exists() and not _outranked_by_newer_capability(p, candidates):
            return p, f"explicitly preferred: {why}"
    scored: list[tuple[Path, str, float]] = []
    for rel in candidates:
        p = Path(rel)
        if not p.exists():
            continue
        ok, why = trained_at_all(p)
        if not ok:
            print(f"  [reject] {p.parent.name}: {why}")
            continue
        s = _score(p)
        if s is None:
            scored.append((p, "", float("nan")))
        else:
            scored.append((p, s[0], s[1]))
    if not scored:
        return None, "no candidate exists"

    with_mask = [t for t in scored if _usable_mask_head(t[0])]
    if with_mask and len(with_mask) < len(scored):
        dropped = [t[0].parent.name for t in scored if t not in with_mask]
        print(f"  [note] ranking only the page-mask candidates "
              f"({', '.join(t[0].parent.name for t in with_mask)}); "
              f"val_mce_px cannot compare them against {', '.join(dropped)}")
        scored = with_mask

    rated = [t for t in scored if t[1]]
    if not rated:
        return scored[0][0], "only candidate (no recorded metric)"
    key = rated[0][1]
    higher = METRICS[key]
    rated.sort(key=lambda t: t[2], reverse=higher)
    best = rated[0]
    if len(rated) == 1:
        return best[0], f"only candidate ({key}={best[2]:.4f})"
    runner = rated[1]
    return best[0], (f"{key}={best[2]:.4f} beats {runner[0].parent.name} "
                     f"({runner[2]:.4f})")


_KEEP_FP32_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
_FP16_SAFE_MAX = 1024.0


def _should_halve(name: str, tensor: torch.Tensor) -> bool:
    if tensor.dtype != torch.float32:
        return False
    if name.endswith(_KEEP_FP32_SUFFIXES):
        return False
    return float(tensor.abs().max()) < _FP16_SAFE_MAX


def export_one(src: str | Path, dst: str | Path, half: bool = True) -> dict:
    src, dst = Path(src), Path(dst)
    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]

    out_state, max_err, kept = {}, 0.0, 0
    for k, v in state.items():
        if half and _should_halve(k, v):
            h = v.half()
            max_err = max(max_err, float((h.float() - v).abs().max()))
            out_state[k] = h
        else:
            kept += 1
            out_state[k] = v

    payload = {
        "model": out_state,
        "config": ckpt.get("config", {}),
        "extra": ckpt.get("extra", {}),
        "exported_from": str(src),
        "half": half,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)

    reloaded = torch.load(str(dst), map_location="cpu", weights_only=False)["model"]
    round_trip_err = 0.0
    for k, v in state.items():
        got = reloaded[k].to(v.dtype)
        if not torch.isfinite(got).all():
            raise ValueError(f"{dst}: non-finite values in {k} after export")
        round_trip_err = max(round_trip_err, float((got - v).abs().max()))

    return {
        "src": str(src), "dst": str(dst),
        "src_mb": src.stat().st_size / 1e6,
        "dst_mb": dst.stat().st_size / 1e6,
        "max_weight_error": max_err,
        "round_trip_error": round_trip_err,
        "kept_fp32": kept,
        "val": {k: v for k, v in ckpt.get("extra", {}).items()
                if isinstance(v, (int, float))},
    }


def _checkpoint_score(path: Path) -> tuple[str, float] | None:
    if not path.exists():
        return None
    try:
        extra = torch.load(str(path), map_location="cpu",
                           weights_only=False).get("extra", {})
    except Exception:
        return None
    for key in METRICS:
        if isinstance(extra.get(key), (int, float)):
            return key, float(extra[key])
    return None


def _epochs(path: Path) -> int | None:
    import json

    hist = path.parent / "history.json"
    if hist.exists():
        try:
            data = json.loads(hist.read_text())
            epochs = data if isinstance(data, list) else data.get("epochs", [])
            if epochs:
                return len(epochs)
        except (OSError, ValueError):
            pass
    try:
        extra = torch.load(str(path), map_location="cpu",
                           weights_only=False).get("extra", {})
    except Exception:
        return None
    e = extra.get("epoch")
    return int(e) if isinstance(e, (int, float)) else None


def _would_regress(candidate: Path, existing: Path) -> str | None:
    if _usable_mask_head(candidate) and not _has_mask_head(existing):
        print(f"  [note] {candidate.parent.name} adds a page-mask head the "
              f"exported weights lack; the frozen-set score cannot compare "
              f"them, so it is not used as a veto.")
        return None

    if _has_mask_head(existing) and not _has_mask_head(candidate):
        return (f"the exported weights have a page-mask head and "
                f"{candidate.parent.name} does not — most likely `runs/` predates "
                f"the model in `models/` (runs/ is git-ignored, so a pull brings "
                f"the weights without them). Retrain, or pass --force if the "
                f"downgrade is deliberate")

    new = _score(candidate) or _checkpoint_score(candidate)
    old = _checkpoint_score(existing)
    if new is None or old is None or new[0] != old[0]:
        return None
    key = new[0]
    better = new[1] > old[1] if METRICS[key] else new[1] < old[1]
    if better:
        return None

    ne, oe = _epochs(candidate), _epochs(existing)
    if ne is not None and oe is not None and ne < oe:
        return (f"{key}={new[1]:.4f} is worse than the exported {key}={old[1]:.4f}, "
                f"and it ran {ne} epochs against {oe} — that is the shape of a "
                f"smoke run, not a retrain")
    print(f"  [note] {candidate.parent.name}: {key}={new[1]:.4f} vs the exported "
          f"{old[1]:.4f} — exporting anyway"
          + (f" ({ne} epochs vs {oe})" if ne and oe else "")
          + ".\n         The frozen set has no distractor pages, so it cannot "
            "measure the\n         real-photo failure this model is aimed at. "
            "Confirm with `eval --corners`.")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export deployable weights")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="models")
    ap.add_argument("--fp32", action="store_true", help="keep float32 weights")
    ap.add_argument("--force", action="store_true",
                    help="export even when it would replace a better model")
    args = ap.parse_args(argv)

    out = Path(args.out)
    total_before = total_after = 0.0
    held = 0
    for name, candidates in CANDIDATES.items():
        if args.runs != "runs":
            candidates = tuple(str(Path(args.runs) / Path(c).relative_to("runs"))
                               for c in candidates)
        src, why = pick_best(candidates, name, runs_root=args.runs)
        if src is None:
            print(f"  [skip] {name}: {why}")
            continue

        pin = out / f"{name}.pt.pin"
        if pin.exists() and (out / f"{name}.pt").exists() and not args.force:
            held += 1
            reason = pin.read_text().strip().splitlines()
            print(f"  [pin]  {name:18s} keeping the exported weights — "
                  f"{reason[0] if reason else 'pinned'}")
            continue
        regress = None if args.force else _would_regress(src, out / f"{name}.pt")
        if regress:
            held += 1
            print(f"  [hold] {name:18s} keeping the exported weights — {regress}")
            continue
        print(f"  {name:18s} <- {src.parent.name}  ({why})")
        info = export_one(src, out / f"{name}.pt", half=not args.fp32)
        total_before += info["src_mb"]
        total_after += info["dst_mb"]
        val = " ".join(f"{k}={v:.4g}" for k, v in info["val"].items())
        print(f"  {name:18s} {info['src_mb']:6.1f} MB -> {info['dst_mb']:5.1f} MB "
              f"(max delta {info['max_weight_error']:.2e}, "
              f"{info['kept_fp32']} tensors kept fp32)  {val}")

    if total_after:
        print(f"  total {total_before:.1f} MB -> {total_after:.1f} MB "
              f"({total_after / max(total_before, 1e-9) * 100:.0f}%)")
    if held:
        print(f"\n  {held} model(s) held back — they scored worse AND ran fewer\n"
              f"  epochs than what is already exported, which is what a smoke run\n"
              f"  looks like. If you meant to replace them:\n\n"
              f"      amazingscanner export --force\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
