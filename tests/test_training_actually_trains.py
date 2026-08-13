from __future__ import annotations

import json

import pytest
import torch

from docscanner.engine.common import TrainConfig, Trainer
from docscanner.engine.export_models import trained_at_all
from docscanner.models.enhance_unet import DocEnhanceNet
from docscanner.models.losses import build_enhancement_loss


def _page(batch: int = 2, size: int = 96) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.full((batch, 3, size, size), 0.93)
    for k in range(24):
        y0 = int(torch.randint(0, size - 12, (1,), generator=g))
        x0 = int(torch.randint(0, size - 12, (1,), generator=g))
        if k % 2:
            x[:, :, y0:y0 + 1, x0:x0 + 12] = 0.12
        else:
            x[:, :, y0:y0 + 12, x0:x0 + 1] = 0.12
    return x


@pytest.mark.parametrize("preset", ["combined", "l1", "mse", "l1_ssim"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_enhancement_loss_gradients_are_finite(preset, dtype):
    target = _page()
    pred = (target + 0.06 * torch.randn(target.shape, generator=torch.Generator().manual_seed(1)))
    pred = pred.clamp(0, 1).to(dtype).requires_grad_(True)

    loss, comps = build_enhancement_loss(preset)(pred, target.to(dtype))
    loss.backward()

    assert torch.isfinite(loss), f"{preset}/{dtype}: loss itself is not finite"
    assert torch.isfinite(pred.grad).all(), (
        f"{preset}/{dtype}: non-finite gradients -- a GradScaler would skip "
        "every step and the model would never move")
    assert float(pred.grad.abs().max()) > 0, f"{preset}/{dtype}: gradients are all zero"
    for name, value in comps.items():
        assert torch.isfinite(value), f"{preset}/{dtype}: component {name} is not finite"


def test_fp16_and_fp32_losses_agree():
    target = _page()
    pred = (target + 0.05).clamp(0, 1)
    crit = build_enhancement_loss("combined")
    a, _ = crit(pred.float(), target.float())
    b, _ = crit(pred.half(), target.half())
    assert abs(float(a) - float(b)) < 2e-3


def test_a_few_steps_move_the_weights():
    torch.manual_seed(0)
    model = DocEnhanceNet(base=8, depth=2)
    crit = build_enhancement_loss("combined")
    before = model.head.weight.detach().clone()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = _page(batch=2, size=64)
    degraded = (target * 0.75 + 0.1).clamp(0, 1)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss, _ = crit(model(degraded), target)
        loss.backward()
        opt.step()

    moved = float((model.head.weight - before).abs().max())
    assert moved > 0, "three optimiser steps left the residual head untouched"


def test_trained_at_all_rejects_an_identity_run(tmp_path):
    dead = tmp_path / "dead"
    dead.mkdir()
    (dead / "history.json").write_text(json.dumps([
        {"epoch": e, "train_loss": 0.3672, "val_psnr": 10.6319,
         "val_psnr_input": 10.6319} for e in range(1, 6)
    ]))
    ok, why = trained_at_all(dead / "best.pt")
    assert not ok
    assert "identity" in why or "never changed" in why

    alive = tmp_path / "alive"
    alive.mkdir()
    (alive / "history.json").write_text(json.dumps([
        {"epoch": 1, "train_loss": 0.09, "val_psnr": 19.4, "val_psnr_input": 10.6},
        {"epoch": 2, "train_loss": 0.06, "val_psnr": 20.8, "val_psnr_input": 10.6},
    ]))
    ok, why = trained_at_all(alive / "best.pt")
    assert ok, why


def test_trainer_raises_when_every_step_is_skipped(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 3, 1)
    batches = [(torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)) for _ in range(6)]

    def poisoned_step(m, batch):
        x, y = batch
        return m(x).mean() * float("nan"), {}

    trainer = Trainer(
        model,
        TrainConfig(name="poisoned", epochs=1, steps_per_epoch=6, batch_size=1,
                    ema_decay=0.0, channels_last=False, device="cpu"),
        batches, poisoned_step, lambda m: {"val_loss": 0.0},
        out_dir=tmp_path / "run", monitor="val_loss",
    )
    with pytest.raises(RuntimeError, match="not finite"):
        trainer.fit()


def test_trainer_completes_a_healthy_run(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 3, 1)
    batches = [(torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)) for _ in range(4)]

    def step(m, batch):
        x, y = batch
        return torch.nn.functional.mse_loss(m(x), y), {}

    trainer = Trainer(
        model,
        TrainConfig(name="healthy", epochs=1, steps_per_epoch=4, batch_size=1,
                    ema_decay=0.0, channels_last=False, device="cpu"),
        batches, step, lambda m: {"val_loss": 0.0},
        out_dir=tmp_path / "run", monitor="val_loss",
    )
    trainer.fit()
    assert trainer.skipped_steps == 0
    assert trainer.taken_steps == 4
