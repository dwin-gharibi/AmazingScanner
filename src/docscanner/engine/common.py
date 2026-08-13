from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

__all__ = ["set_seed", "History", "EMA", "cosine_lr", "Trainer", "TrainConfig",
           "save_checkpoint", "load_checkpoint", "plot_history", "count_parameters",
           "resolve_device", "to_device", "DeviceLoader",
           "measure_step_time", "calibrate_steps_per_epoch", "default_workers"]


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(False)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def resolve_device(spec: str = "auto") -> torch.device:
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested but torch.cuda.is_available() is False. "
            "Use the CUDA image (Dockerfile.gpu) and `docker run --gpus all`, or "
            "pass --device cpu.")
    if dev.type == "mps" and not (getattr(torch.backends, "mps", None)
                                  and torch.backends.mps.is_available()):
        raise RuntimeError("device='mps' was requested but MPS is not available.")
    return dev


def to_device(batch, device: torch.device):
    if device.type == "cpu":
        return batch
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch


def default_workers(requested: int | None = None) -> int:
    if requested is not None and requested >= 0:
        return int(requested)
    return max(1, (os.cpu_count() or 2) - 1)


def measure_step_time(loader, n: int = 6, warmup: int = 2) -> float:
    it = iter(loader)
    for _ in range(warmup):
        try:
            next(it)
        except StopIteration:
            return 0.0
    t0, taken = time.time(), 0
    for _ in range(n):
        try:
            next(it)
        except StopIteration:
            break
        taken += 1
    return (time.time() - t0) / max(taken, 1)


def calibrate_steps_per_epoch(step_time: float, *, epochs: int,
                              minutes: float | None, reserve: float = 0.3,
                              floor: int = 20, ceiling: int = 100_000) -> int | None:
    if not minutes or minutes <= 0 or step_time <= 0:
        return None
    usable_seconds = minutes * 60.0 * (1.0 - reserve)
    steps = int(usable_seconds / max(step_time, 1e-6) // max(epochs, 1))
    return max(floor, min(steps, ceiling))


class DeviceLoader:
    def __init__(self, loader, device: torch.device):
        self.loader, self.device = loader, device

    def __iter__(self):
        for batch in self.loader:
            yield to_device(batch, self.device)

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, item):
        return getattr(self.loader, item)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class TrainConfig:
    name: str = "run"
    epochs: int = 20
    steps_per_epoch: int = 200
    batch_size: int = 8
    lr: float = 2e-3
    min_lr_ratio: float = 0.02
    warmup_frac: float = 0.05
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    num_workers: int = 3
    seed: int = 0
    ema_decay: float = 0.999
    channels_last: bool = True
    max_minutes: float | None = None
    amp: bool = False
    device: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, default=str)


class History:
    def __init__(self, path: str | Path | None = None):
        self.rows: list[dict[str, float]] = []
        self.path = Path(path) if path else None

    def append(self, **row: float) -> None:
        self.rows.append({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in row.items()})
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.rows, indent=1))

    def column(self, key: str) -> list[float]:
        return [r[key] for r in self.rows if key in r]

    @classmethod
    def load(cls, path: str | Path) -> History:
        h = cls(path)
        p = Path(path)
        if p.exists():
            h.rows = json.loads(p.read_text())
        return h

    def best(self, key: str, mode: str = "min") -> dict[str, float] | None:
        vals = [r for r in self.rows if key in r]
        if not vals:
            return None
        return (min if mode == "min" else max)(vals, key=lambda r: r[key])


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                s.copy_(m)

    def state_dict(self):
        return self.shadow.state_dict()


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 0,
              min_ratio: float = 0.02) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def save_checkpoint(path: str | Path, model: nn.Module, cfg: TrainConfig,
                    extra: dict[str, Any] | None = None,
                    ema: EMA | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "config": asdict(cfg),
        "arch": {"class": type(model).__name__, **getattr(model, "hparams", {})},
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, model: nn.Module, prefer_ema: bool = True,
                    strict: bool = True) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("ema") if (prefer_ema and ckpt.get("ema")) else ckpt["model"]
    model.load_state_dict(state, strict=strict)
    return ckpt


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        train_loader: Iterable,
        step_fn: Callable[[nn.Module, Any], tuple[torch.Tensor, dict]],
        validate_fn: Callable[[nn.Module], dict[str, float]],
        out_dir: str | Path,
        monitor: str = "val_loss",
        monitor_mode: str = "min",
        log_every: int = 25,
    ):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.use_amp = bool(cfg.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.loader = train_loader
        self.step_fn = step_fn
        self.validate_fn = validate_fn
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.log_every = log_every

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.99), eps=1e-8,
        )
        self.ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
        self.history = History(self.out_dir / "history.json")
        self.total_steps = cfg.epochs * cfg.steps_per_epoch
        self.warmup = int(self.total_steps * cfg.warmup_frac)
        self.global_step = 0
        self.skipped_steps = 0
        self.taken_steps = 0
        self.best_value = math.inf if monitor_mode == "min" else -math.inf
        (self.out_dir / "config.json").write_text(cfg.to_json())

    def _is_better(self, value: float) -> bool:
        return value < self.best_value if self.monitor_mode == "min" else value > self.best_value

    def _set_lr(self) -> float:
        lr = cosine_lr(self.global_step, self.total_steps, self.cfg.lr,
                       self.warmup, self.cfg.min_lr_ratio)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    SKIP_TOLERANCE = 0.5

    def _check_progress(self, epoch: int, seen: int) -> None:
        if not self.skipped_steps:
            return
        share = self.skipped_steps / max(self.skipped_steps + self.taken_steps, 1)
        print(f"  [warning] {self.skipped_steps} of "
              f"{self.skipped_steps + self.taken_steps} steps skipped "
              f"({share:.0%}) -- non-finite loss or gradients", flush=True)
        if share > self.SKIP_TOLERANCE and epoch == 0:
            raise RuntimeError(
                f"[{self.cfg.name}] {share:.0%} of the first epoch's optimiser "
                "steps were skipped because the gradients were not finite, so "
                "the weights are not moving. Training would burn the whole "
                "budget and produce the untrained model. Common causes: an fp16 "
                "overflow in a loss term (try --no-amp to confirm), a learning "
                "rate that diverged, or a NaN in the data.")

    def fit(self) -> History:
        cfg = self.cfg
        start = time.time()
        dev = str(self.device)
        if self.device.type == "cuda":
            dev += f" ({torch.cuda.get_device_name(self.device)})"
        print(f"[{cfg.name}] {count_parameters(self.model) / 1e6:.2f}M params | "
              f"{cfg.epochs} epochs x {cfg.steps_per_epoch} steps | batch {cfg.batch_size} | "
              f"budget {cfg.max_minutes or float('inf'):.0f} min | device {dev}"
              + (" | amp fp16" if self.use_amp else ""), flush=True)

        stopped = False
        for epoch in range(cfg.epochs):
            if hasattr(self.loader, "dataset") and hasattr(self.loader.dataset, "set_epoch"):
                self.loader.dataset.set_epoch(epoch)
            self.model.train()
            running: dict[str, float] = {}
            seen = 0
            t_epoch = time.time()

            epoch_deadline = None
            if cfg.max_minutes:
                remaining = cfg.max_minutes * 60 - (time.time() - start)
                epochs_left = cfg.epochs - epoch
                epoch_deadline = time.time() + max(1.0, remaining / epochs_left * 0.85)

            for batch in self.loader:
                if seen >= cfg.steps_per_epoch:
                    break
                if epoch_deadline and time.time() > epoch_deadline and seen > 0:
                    break
                lr = self._set_lr()
                self.optimizer.zero_grad(set_to_none=True)
                batch = to_device(batch, self.device)
                if self.use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss, comps = self.step_fn(self.model, batch)
                    self.scaler.scale(loss).backward()
                    if cfg.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       cfg.grad_clip)
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scaler.get_scale() < scale_before:
                        self.skipped_steps += 1
                else:
                    loss, comps = self.step_fn(self.model, batch)
                    if not torch.isfinite(loss):
                        self.skipped_steps += 1
                        continue
                    loss.backward()
                    if cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       cfg.grad_clip)
                    self.optimizer.step()
                self.taken_steps += 1
                if self.ema is not None:
                    self.ema.update(self.model, self.global_step)

                running["loss"] = running.get("loss", 0.0) + float(loss.detach())
                for k, v in comps.items():
                    running[k] = running.get(k, 0.0) + float(v)
                seen += 1
                self.global_step += 1

                if seen % self.log_every == 0:
                    el = time.time() - t_epoch
                    print(f"  e{epoch + 1:02d} {seen:4d}/{cfg.steps_per_epoch} "
                          f"loss={running['loss'] / seen:.4f} lr={lr:.2e} "
                          f"{el / seen:.2f}s/step", flush=True)

                if cfg.max_minutes and (time.time() - start) / 60 > cfg.max_minutes:
                    print(f"  [time budget reached at epoch {epoch + 1}, step {seen}]",
                          flush=True)
                    stopped = True
                    break

            self._check_progress(epoch, seen)
            train_metrics = {f"train_{k}": v / max(seen, 1) for k, v in running.items()}
            eval_model = self.ema.shadow if self.ema is not None else self.model
            val_metrics = self.validate_fn(eval_model)

            row = {"epoch": epoch + 1, "lr": lr, "seconds": time.time() - t_epoch,
                   **train_metrics, **val_metrics}
            self.history.append(**row)

            msg = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"[{cfg.name}] epoch {epoch + 1}/{cfg.epochs} "
                  f"train_loss={train_metrics.get('train_loss', float('nan')):.4f} {msg} "
                  f"({row['seconds']:.0f}s)", flush=True)

            if self.monitor in val_metrics and self._is_better(val_metrics[self.monitor]):
                self.best_value = val_metrics[self.monitor]
                save_checkpoint(self.out_dir / "best.pt", self.model, cfg,
                                extra={"epoch": epoch + 1, **val_metrics}, ema=self.ema)
                print(f"  -> new best {self.monitor}={self.best_value:.4f}", flush=True)
            save_checkpoint(self.out_dir / "last.pt", self.model, cfg,
                            extra={"epoch": epoch + 1, **val_metrics}, ema=self.ema)

            if stopped:
                break

        total = (time.time() - start) / 60
        print(f"[{cfg.name}] done in {total:.1f} min | best {self.monitor}="
              f"{self.best_value:.4f}", flush=True)
        try:
            plot_history(self.history, self.out_dir / "curves.png", title=cfg.name)
        except Exception as exc:
            print(f"  [plot failed: {exc}]")
        return self.history


def plot_history(history: History, path: str | Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = history.rows
    if not rows:
        raise ValueError("empty history")
    epochs = [r["epoch"] for r in rows]

    metric_keys = [k for k in rows[0]
                   if k.startswith("val_") and k not in ("val_loss",)]
    ncols = 2 if metric_keys else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.2), dpi=130)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    if "train_loss" in rows[0]:
        ax.plot(epochs, [r["train_loss"] for r in rows], "-o", ms=3,
                label="train", color="#2b6cb0")
    if "val_loss" in rows[0]:
        ax.plot(epochs, [r["val_loss"] for r in rows], "-o", ms=3,
                label="validation", color="#c53030")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{title} - loss")
    ax.grid(alpha=0.25)
    ax.legend()

    if metric_keys:
        ax = axes[1]
        for i, k in enumerate(sorted(metric_keys)):
            vals = [r.get(k, np.nan) for r in rows]
            ax.plot(epochs, vals, "-o", ms=3, label=k.replace("val_", ""),
                    color=plt.cm.viridis(i / max(len(metric_keys) - 1, 1)))
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} - validation metrics")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
