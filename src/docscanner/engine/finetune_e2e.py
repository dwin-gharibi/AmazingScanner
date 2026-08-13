from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.datasets import DOC_MEAN, DOC_STD, resize_with_corners, to_chw
from ..data.degrade import DegradationConfig
from ..data.prepare import load_splits, make_generator
from ..models.corner_nets import CornerHeatmapNet, CornerRegressor, soft_argmax_2d
from ..models.enhance_unet import DocEnhanceNet
from ..models.losses import CornerCoordLoss, build_enhancement_loss
from ..models.warp import rectify_batch
from .common import DeviceLoader, TrainConfig, Trainer, resolve_device, set_seed


class JointDataset(Dataset):
    def __init__(self, generator, length: int, photo_size: int = 256,
                 page_hw: tuple[int, int] = (224, 224), seed: int = 0):
        self.gen = generator
        self.length = int(length)
        self.photo_size = int(photo_size)
        self.page_hw = page_hw
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng((self.seed, self._epoch, int(index)))
        h, w = self.page_hw
        s = self.gen.generate(rng, want_rectified=True, page_size=(w, h))
        photo, corners = resize_with_corners(s.photo, s.corners_content, self.photo_size)
        return (
            to_chw(photo),
            to_chw(s.target),
            torch.from_numpy((corners / self.photo_size).astype(np.float32)),
        )


def _load_corner_model(path: Path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    extra = ckpt.get("config", {}).get("extra", {})
    approach = extra.get("approach", "heatmap")
    state = ckpt.get("ema") or ckpt["model"]
    if approach == "regression":
        model = CornerRegressor(base=extra.get("base", 24), depth=extra.get("depth", 5),
                                head_dim=extra.get("head_dim", 256), dropout=0.0,
                                input_size=extra.get("size", 256))
    else:
        model = CornerHeatmapNet(base=extra.get("base", 24), depth=extra.get("depth", 4),
                                 out_stride=extra.get("out_stride", 4),
                                 heatmap_pad=extra.get("pad", 0.12), dropout=0.0,
                                 seg_head=any(k.startswith("seg.") for k in state))
    model.load_state_dict(state)
    return model, approach, extra


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end fine-tuning of the scanner")
    from .export_models import CANDIDATES, pick_best
    _corner = pick_best(CANDIDATES["corner_heatmap"], name="corner_heatmap")[0]
    _enh = pick_best(CANDIDATES["enhance"], name="enhance")[0]
    ap.add_argument("--corner-checkpoint",
                    default=str(_corner) if _corner else "runs/corner_heatmap/best.pt")
    ap.add_argument("--enhance-checkpoint",
                    default=str(_enh) if _enh else "runs/enhance_main/best.pt")
    ap.add_argument("--name", default="e2e_finetune")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--photo-size", type=int, default=256)
    ap.add_argument("--page", type=int, default=224)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--coord-weight", type=float, default=2.0,
                    help="anchor on the coordinate loss; 0 = reconstruction only")
    ap.add_argument("--unfreeze-enhancer", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:N | mps")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast + grad scaler; a CUDA win, ignored on CPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=None)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    corner_model, approach, extra = _load_corner_model(Path(args.corner_checkpoint))
    enh_ckpt = torch.load(str(args.enhance_checkpoint), map_location="cpu",
                          weights_only=False)
    e_extra = enh_ckpt.get("config", {}).get("extra", {})
    enhancer = DocEnhanceNet(base=e_extra.get("base", 32), depth=e_extra.get("depth", 4),
                             dropout=0.0, use_bg_prior=e_extra.get("bg_prior", True))
    enhancer.load_state_dict(enh_ckpt.get("ema") or enh_ckpt["model"])

    if not args.unfreeze_enhancer:
        enhancer.eval()
        for p in enhancer.parameters():
            p.requires_grad_(False)

    splits = load_splits()
    gen = make_generator("train", cfg=DegradationConfig.for_corners(), splits=splits,
                         photo_long_side=640, page_long_side=args.page)
    train_ds = JointDataset(gen, length=args.steps * args.batch,
                            photo_size=args.photo_size,
                            page_hw=(args.page, args.page), seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, drop_last=True,
                              persistent_workers=args.workers > 0)

    val_gen = make_generator("val", cfg=DegradationConfig.for_corners(), splits=splits,
                             photo_long_side=640, page_long_side=args.page)
    val_ds = JointDataset(val_gen, length=48, photo_size=args.photo_size,
                          page_hw=(args.page, args.page), seed=9999)
    device = resolve_device(args.device)
    val_loader = DeviceLoader(
        DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0),
        device)

    enhancer = enhancer.to(device)
    corner_model = corner_model.to(device)
    enh_loss = build_enhancement_loss("combined").to(device)
    coord_loss = CornerCoordLoss("wing").to(device)
    mean = torch.tensor(DOC_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DOC_STD, device=device).view(1, 3, 1, 1)

    def predict_corners(model, photo01):
        x = (photo01 - mean) / std
        if approach == "heatmap":
            return soft_argmax_2d(torch.sigmoid(model(x)), pad=extra.get("pad", 0.12))
        return model(x)

    def forward(model, batch):
        photo, target, gt = batch
        pred = predict_corners(model, photo)
        safe = pred.clamp(-0.2, 1.2)
        rect = rectify_batch(photo, safe, (args.page, args.page))
        out = enhancer(rect)
        recon, comps = enh_loss(out, target)
        total = recon
        if args.coord_weight:
            c = coord_loss(pred, gt)
            comps["coord"] = c.detach()
            total = total + args.coord_weight * c
        comps["recon"] = recon.detach()
        return total, comps

    @torch.no_grad()
    def validate_fn(model):
        model.eval()
        t0 = time.time()
        recon, errs = [], []
        for batch in val_loader:
            photo, target, gt = batch
            pred = predict_corners(model, photo).clamp(-0.2, 1.2)
            rect = rectify_batch(photo, pred, (args.page, args.page))
            out = enhancer(rect)
            r, _ = enh_loss(out, target)
            recon.append(float(r))
            errs.append(float((torch.norm(pred - gt, dim=-1) * args.photo_size).mean()))
        model.train()
        return {"val_loss": float(np.mean(recon)),
                "val_mce_px": float(np.mean(errs)),
                "val_seconds": time.time() - t0}

    cfg = TrainConfig(name=args.name, epochs=args.epochs, steps_per_epoch=args.steps,
                      batch_size=args.batch, lr=args.lr, num_workers=args.workers,
                      seed=args.seed, ema_decay=0.999, max_minutes=args.minutes,
                      device=args.device, amp=args.amp,
                      extra={**extra, "approach": approach, "finetune": "end-to-end",
                             "coord_weight": args.coord_weight,
                             "enhancer_frozen": not args.unfreeze_enhancer})

    print(f"[{args.name}] baseline before fine-tuning:", validate_fn(corner_model))
    trainer = Trainer(corner_model, cfg, train_loader, forward, validate_fn,
                      out_dir=Path(args.out) / args.name,
                      monitor="val_mce_px", monitor_mode="min")
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
