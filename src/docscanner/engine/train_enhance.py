from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.datasets import EnhancementDataset, FrozenPairSet
from ..data.degrade import DegradationConfig
from ..data.prepare import load_splits, make_generator
from ..eval.metrics import psnr_torch, ssim_torch
from ..models.enhance_unet import DocEnhanceNet
from ..models.losses import build_enhancement_loss
from .common import (
    DeviceLoader,
    TrainConfig,
    Trainer,
    default_workers,
    resolve_device,
    set_seed,
)


def build_model(args) -> DocEnhanceNet:
    return DocEnhanceNet(
        base=args.base,
        depth=args.depth,
        dropout=args.dropout,
        use_bg_prior=not args.no_bg_prior,
        bilinear=True,
        se=True,
    )


def make_loaders(args, splits):
    gen = make_generator("train", cfg=DegradationConfig.for_enhancement(),
                         splits=splits,
                         photo_long_side=args.photo_size,
                         page_long_side=args.page_size)

    args.workers = default_workers(args.workers)

    def _loader(steps: int):
        ds = EnhancementDataset(
            gen, length=steps * args.batch, crop=args.crop,
            crops_per_photo=args.crops_per_photo, page_long_side=args.page_size,
            page_size_range=(args.page_min, args.page_max), seed=args.seed,
        )
        return DataLoader(
            ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
            pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
            prefetch_factor=4 if args.workers > 0 else None,
        )

    train_loader = _loader(args.steps)
    val_ds = FrozenPairSet(Path("data/frozen/enhance_val"), crop=args.crop,
                           crops_per_page=args.val_crops)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    return train_loader, val_loader


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the document enhancement network")
    ap.add_argument("--name", default="enhance_main")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--loss", default="combined",
                    choices=["mse", "l1", "l1_ssim", "combined"])
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--no-bg-prior", action="store_true")
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--steps", type=int, default=180, help="optimizer steps per epoch")
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--crops-per-photo", type=int, default=5)
    ap.add_argument("--photo-size", type=int, default=1440)
    ap.add_argument("--page-size", type=int, default=896)
    ap.add_argument("--page-min", type=int, default=640,
                    help="scale augmentation: smallest rectified page raster")
    ap.add_argument("--page-max", type=int, default=1152)
    ap.add_argument("--val-crops", type=int, default=2)
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
    ap.add_argument("--resume", default=None)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    splits = load_splits()

    model = build_model(args)
    if args.resume:
        from .common import load_checkpoint
        load_checkpoint(args.resume, model, prefer_ema=False)
        print(f"resumed from {args.resume}")

    device = resolve_device(args.device)
    criterion = build_enhancement_loss(args.loss).to(device)
    train_loader, val_loader = make_loaders(args, splits)
    val_loader = DeviceLoader(val_loader, device)

    cfg = TrainConfig(
        name=args.name, epochs=args.epochs, steps_per_epoch=args.steps,
        batch_size=args.batch, lr=args.lr, weight_decay=args.weight_decay,
        num_workers=args.workers, seed=args.seed, ema_decay=args.ema,
        max_minutes=args.minutes, device=args.device, amp=args.amp,
        extra={"loss": args.loss, "crop": args.crop, "base": args.base,
               "depth": args.depth, "dropout": args.dropout,
               "bg_prior": not args.no_bg_prior,
               "crops_per_photo": args.crops_per_photo,
               "page_size": args.page_size, "photo_size": args.photo_size},
    )

    def step_fn(model, batch):
        x, y = batch
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
            y = y.contiguous(memory_format=torch.channels_last)
        pred = model(x)
        return criterion(pred, y)

    @torch.no_grad()
    def validate_fn(model) -> dict[str, float]:
        model.eval()
        t0 = time.time()
        tot_loss = tot_psnr = tot_ssim = 0.0
        tot_psnr_in = 0.0
        n = 0
        for x, y in val_loader:
            if cfg.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            pred = model(x)
            loss, _ = criterion(pred, y)
            p = pred.clamp(0, 1)
            tot_loss += float(loss) * x.shape[0]
            tot_psnr += psnr_torch(p, y) * x.shape[0]
            tot_ssim += ssim_torch(p, y) * x.shape[0]
            tot_psnr_in += psnr_torch(x, y) * x.shape[0]
            n += x.shape[0]
        model.train()
        return {
            "val_loss": tot_loss / max(n, 1),
            "val_psnr": tot_psnr / max(n, 1),
            "val_ssim": tot_ssim / max(n, 1),
            "val_psnr_input": tot_psnr_in / max(n, 1),
            "val_seconds": time.time() - t0,
        }

    trainer = Trainer(
        model, cfg, train_loader, step_fn, validate_fn,
        out_dir=Path(args.out) / args.name,
        monitor="val_psnr", monitor_mode="max",
    )
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
