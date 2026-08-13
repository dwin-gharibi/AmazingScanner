from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.datasets import CornerDataset, FrozenCornerSet
from ..data.degrade import DegradationConfig
from ..data.prepare import load_splits, make_generator
from ..models.corner_nets import CornerHeatmapNet, CornerRegressor, soft_argmax_2d
from ..models.losses import CornerCoordLoss, HeatmapLoss
from .common import (
    DeviceLoader,
    TrainConfig,
    Trainer,
    default_workers,
    resolve_device,
    set_seed,
)

PAGE_AUGMENTATION = (
    "p_page_tint", "p_page_band", "p_ruled_paper",
    "p_page_stack", "p_binding", "p_gutter",
    "p_round_corners", "p_dog_ear", "p_corner_occluder",
    "p_backlit", "p_pale_background",
)


def build_model(args):
    if args.approach == "regression":
        return CornerRegressor(base=args.base, depth=args.depth,
                               dropout=args.dropout, input_size=args.size,
                               head_dim=args.head_dim)
    return CornerHeatmapNet(base=args.base, depth=args.depth, dropout=args.dropout,
                            out_stride=args.out_stride, heatmap_pad=args.pad,
                            seg_head=args.seg_weight > 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train a corner detector")
    ap.add_argument("--approach", choices=["regression", "heatmap"], required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--out-stride", type=int, default=4)
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--sigma", type=float, default=1.6)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--coord-loss", default="wing", choices=["wing", "l1", "l2"])
    ap.add_argument("--coord-weight", type=float, default=0.15)
    ap.add_argument("--no-page-aug", action="store_true",
                    help="disable page/boundary/corner augmentation (the control "
                         "arm; see PAGE_AUGMENTATION)")
    ap.add_argument("--seg-weight", type=float, default=0.5,
                    help="weight on the page-interior mask head (heatmap approach "
                         "only); 0 disables the head entirely")
    ap.add_argument("--perm-invariant", action="store_true",
                    help="match the loss against the best cyclic corner ordering "
                         "(off by default: it destabilises the channel assignment)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--photo-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=None,
                    help="loader processes (default: one per core, less one)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:N | mps")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast + grad scaler; a CUDA win, ignored on CPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--auto-steps", action="store_true",
                    help="accepted for compatibility; every epoch is now capped "
                         "at its share of --minutes automatically")
    args = ap.parse_args(argv)

    if args.depth is None:
        args.depth = 5 if args.approach == "regression" else 4
    name = args.name or f"corner_{args.approach}"

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    splits = load_splits()

    model = build_model(args)
    train_cfg = DegradationConfig.for_corners()
    if args.no_page_aug:
        for knob in PAGE_AUGMENTATION:
            setattr(train_cfg, knob, 0.0)
    gen = make_generator("train", cfg=train_cfg,
                         splits=splits, photo_long_side=args.photo_size)

    args.workers = default_workers(args.workers)

    def _loader(steps: int):
        ds = CornerDataset(gen, length=steps * args.batch, size=args.size,
                           seed=args.seed)
        return DataLoader(ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, drop_last=True,
                          pin_memory=True,
                          persistent_workers=args.workers > 0,
                          prefetch_factor=4 if args.workers > 0 else None)

    train_loader = _loader(args.steps)
    val_ds = FrozenCornerSet(Path("data/frozen/corners_val"), size=args.size)
    device = resolve_device(args.device)
    val_loader = DeviceLoader(
        DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0),
        device)

    is_heatmap = args.approach == "heatmap"
    coord_loss = CornerCoordLoss(args.coord_loss,
                                 permutation_invariant=args.perm_invariant).to(device)
    heat_loss = HeatmapLoss(coord_weight=args.coord_weight, pad=args.pad,
                            sigma=args.sigma, seg_weight=args.seg_weight,
                            permutation_invariant=args.perm_invariant).to(device)
    use_seg = is_heatmap and args.seg_weight > 0

    def _forward(model, x):
        if use_seg:
            return model.forward_with_seg(x)
        return model(x), None

    cfg = TrainConfig(
        name=name, epochs=args.epochs, steps_per_epoch=args.steps,
        batch_size=args.batch, lr=args.lr, weight_decay=args.weight_decay,
        num_workers=args.workers, seed=args.seed, ema_decay=args.ema,
        max_minutes=args.minutes, device=args.device, amp=args.amp,
        extra={"approach": args.approach, "size": args.size, "base": args.base,
               "depth": args.depth, "dropout": args.dropout, "pad": args.pad,
               "out_stride": args.out_stride, "coord_loss": args.coord_loss,
               "head_dim": args.head_dim, "sigma": args.sigma,
               "seg_weight": args.seg_weight, "seg_head": use_seg,
               "page_aug": not args.no_page_aug},
    )

    def step_fn(model, batch):
        x, c = batch
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if is_heatmap:
            out, seg = _forward(model, x)
            return heat_loss(out, c, seg)
        loss = coord_loss(model(x), c)
        return loss, {"coord": loss.detach()}

    @torch.no_grad()
    def validate_fn(model) -> dict[str, float]:
        model.eval()
        t0 = time.time()
        errs, losses, preds, gts, mious = [], [], [], [], []
        for x, c in val_loader:
            if cfg.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            if is_heatmap:
                out, seg = _forward(model, x)
                loss, _ = heat_loss(out, c, seg)
                pred = soft_argmax_2d(torch.sigmoid(out), pad=args.pad)
                if seg is not None:
                    from ..models.corner_nets import coords_to_mask
                    p = torch.sigmoid(seg) > 0.5
                    t = coords_to_mask(c, seg.shape[-1], pad=args.pad) > 0.5
                    inter = (p & t).flatten(1).sum(1).float()
                    union = (p | t).flatten(1).sum(1).float().clamp(min=1)
                    mious.append((inter / union).cpu().numpy())
            else:
                out = model(x)
                loss = coord_loss(out, c)
                pred = out
            losses.append(float(loss))
            errs.append((torch.norm(pred - c, dim=-1) * args.size).cpu().numpy())
            preds.append(pred.cpu().numpy() * args.size)
            gts.append(c.cpu().numpy() * args.size)
        model.train()
        e = np.concatenate(errs, axis=0) if errs else np.zeros((1, 4))
        per_image = e.mean(axis=1)
        from ..eval.metrics import match_cyclic
        p_all = np.concatenate(preds, axis=0)
        g_all = match_cyclic(p_all, np.concatenate(gts, axis=0))
        cyc = np.linalg.norm(p_all - g_all, axis=2).mean(axis=1)
        out_metrics = {
            "val_loss": float(np.mean(losses)),
            "val_mce_px": float(per_image.mean()),
            "val_mce_cyclic_px": float(cyc.mean()),
            "val_median_px": float(np.median(per_image)),
            "val_success_8px": float((e.max(axis=1) <= 8).mean()) * 100.0,
            "val_seconds": time.time() - t0,
        }
        if mious:
            out_metrics["val_mask_iou"] = float(np.concatenate(mious).mean())
        return out_metrics

    trainer = Trainer(model, cfg, train_loader, step_fn, validate_fn,
                      out_dir=Path(args.out) / name,
                      monitor="val_mce_px", monitor_mode="min")
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
