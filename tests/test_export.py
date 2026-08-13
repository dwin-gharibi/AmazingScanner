from __future__ import annotations

import torch

from docscanner.engine.common import TrainConfig, save_checkpoint
from docscanner.engine.export_models import _should_halve, export_one
from docscanner.models.enhance_unet import DocEnhanceNet


def _checkpoint(tmp_path, running_var_max: float = 1.0):
    model = DocEnhanceNet(base=8, depth=3)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_var.fill_(running_var_max)
    cfg = TrainConfig(name="t", extra={"base": 8, "depth": 3, "bg_prior": True})
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, cfg)
    return model, path


def test_export_is_smaller_but_faithful(tmp_path):
    model, src = _checkpoint(tmp_path)
    info = export_one(src, tmp_path / "out.pt")
    assert info["dst_mb"] < info["src_mb"]
    assert info["round_trip_error"] < 1e-3


def test_batchnorm_statistics_stay_float32(tmp_path):
    _, src = _checkpoint(tmp_path, running_var_max=63000.0)
    dst = tmp_path / "out.pt"
    info = export_one(src, dst)

    state = torch.load(str(dst), map_location="cpu", weights_only=False)["model"]
    big = [(k, v) for k, v in state.items() if k.endswith("running_var")]
    assert big, "no BatchNorm buffers found"
    for k, v in big:
        assert v.dtype == torch.float32, f"{k} was stored as {v.dtype}"
        assert torch.isfinite(v).all()
    assert info["max_weight_error"] < 1e-2


def test_should_halve_rejects_buffers_and_large_tensors():
    assert _should_halve("stem.conv1.0.weight", torch.randn(4, 4))
    assert not _should_halve("stem.conv1.1.running_var", torch.rand(4))
    assert not _should_halve("stem.conv1.1.running_mean", torch.rand(4))
    assert not _should_halve("some.weight", torch.full((4,), 5e4))
    assert not _should_halve("counter", torch.tensor([3]))


def test_exported_model_produces_the_same_output(tmp_path):
    model, src = _checkpoint(tmp_path, running_var_max=40000.0)
    dst = tmp_path / "out.pt"
    export_one(src, dst)

    rebuilt = DocEnhanceNet(base=8, depth=3)
    state = torch.load(str(dst), map_location="cpu", weights_only=False)["model"]
    rebuilt.load_state_dict({k: v.float() for k, v in state.items()})

    x = torch.rand(1, 3, 64, 64)
    model.eval()
    rebuilt.eval()
    with torch.no_grad():
        a, b = model(x, clamp=True), rebuilt(x, clamp=True)
    assert torch.isfinite(b).all()
    assert float((a - b).abs().max()) < 2e-3


def _corner_checkpoint(path, *, seg: bool, mce: float, mask_iou=None, epoch=10):
    from docscanner.models.corner_nets import CornerHeatmapNet

    model = CornerHeatmapNet(base=8, depth=4, seg_head=seg)
    cfg = TrainConfig(name=path.parent.name,
                      extra={"approach": "heatmap", "base": 8, "depth": 4})
    extra = {"epoch": epoch, "val_mce_px": mce}
    if mask_iou is not None:
        extra["val_mask_iou"] = mask_iou
    save_checkpoint(path, model, cfg, extra=extra)
    return path


def test_a_pinned_preference_does_not_outlive_a_retrain(tmp_path):
    from docscanner.engine.export_models import pick_best

    runs = tmp_path / "runs"
    (runs / "corner_heatmap_v3").mkdir(parents=True)
    (runs / "corner_heatmap").mkdir(parents=True)
    _corner_checkpoint(runs / "corner_heatmap_v3" / "best.pt", seg=False, mce=10.0)
    _corner_checkpoint(runs / "corner_heatmap" / "best.pt", seg=True, mce=40.0,
                       mask_iou=0.9)

    cands = (str(runs / "corner_heatmap_v3" / "best.pt"),
             str(runs / "corner_heatmap" / "best.pt"))
    best, why = pick_best(cands, name="corner_heatmap", runs_root=runs)
    assert best.parent.name == "corner_heatmap", why


def test_a_mask_head_too_weak_to_use_does_not_hijack_the_export(tmp_path):
    from docscanner.engine.export_models import pick_best

    runs = tmp_path / "runs"
    (runs / "corner_heatmap_v3").mkdir(parents=True)
    (runs / "corner_heatmap").mkdir(parents=True)
    _corner_checkpoint(runs / "corner_heatmap_v3" / "best.pt", seg=False, mce=10.0)
    _corner_checkpoint(runs / "corner_heatmap" / "best.pt", seg=True, mce=40.0,
                       mask_iou=0.5)

    cands = (str(runs / "corner_heatmap_v3" / "best.pt"),
             str(runs / "corner_heatmap" / "best.pt"))
    best, why = pick_best(cands, name="corner_heatmap", runs_root=runs)
    assert best.parent.name == "corner_heatmap_v3", why


def test_the_regression_guard_does_not_veto_a_new_capability(tmp_path):
    from docscanner.engine.export_models import _would_regress

    runs = tmp_path / "runs"
    (runs / "new").mkdir(parents=True)
    shipped = tmp_path / "corner_heatmap.pt"
    _corner_checkpoint(runs / "new" / "best.pt", seg=True, mce=40.0,
                       mask_iou=0.9, epoch=40)
    _corner_checkpoint(shipped, seg=False, mce=10.0, epoch=23)

    assert _would_regress(runs / "new" / "best.pt", shipped) is None


def test_the_pipeline_refuses_a_mask_head_it_cannot_vouch_for(tmp_path):
    import warnings

    from docscanner.pipeline.corner_pipeline import CornerPipeline

    weak = _corner_checkpoint(tmp_path / "weak.pt", seg=True, mce=20.0, mask_iou=0.5)
    unknown = _corner_checkpoint(tmp_path / "unknown.pt", seg=True, mce=20.0)
    good = _corner_checkpoint(tmp_path / "good.pt", seg=True, mce=20.0, mask_iou=0.95)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert CornerPipeline(weak).use_mask is False
        assert CornerPipeline(unknown).use_mask is False
        assert CornerPipeline(good).use_mask is True


def test_a_pin_outranks_every_export_heuristic(tmp_path):
    from docscanner.engine import export_models as em

    runs = tmp_path / "runs"; (runs / "enhance_main").mkdir(parents=True)
    out = tmp_path / "models"; out.mkdir()
    model = DocEnhanceNet(base=8, depth=3)
    cfg = TrainConfig(name="enhance_main", extra={"base": 8, "depth": 3})
    save_checkpoint(runs / "enhance_main" / "best.pt", model, cfg,
                    extra={"epoch": 40, "val_psnr": 99.0})
    save_checkpoint(out / "enhance.pt", model, cfg,
                    extra={"epoch": 24, "val_psnr": 21.0})
    (out / "enhance.pt.pin").write_text("user preference: keep this one\n")
    before = (out / "enhance.pt").read_bytes()

    rc = em.main(["--runs", str(runs), "--out", str(out)])
    assert rc == 0
    assert (out / "enhance.pt").read_bytes() == before, "pin was ignored"

    rc = em.main(["--runs", str(runs), "--out", str(out), "--force"])
    assert rc == 0
    assert (out / "enhance.pt").read_bytes() != before, "--force must override a pin"


def test_a_stale_run_cannot_downgrade_a_shipped_mask_model(tmp_path):
    from docscanner.engine.export_models import _would_regress

    runs = tmp_path / "runs"; (runs / "old").mkdir(parents=True)
    shipped = tmp_path / "corner_heatmap.pt"
    _corner_checkpoint(runs / "old" / "best.pt", seg=False, mce=10.0, epoch=40)
    _corner_checkpoint(shipped, seg=True, mce=15.0, mask_iou=0.92, epoch=29)

    why = _would_regress(runs / "old" / "best.pt", shipped)
    assert why and "page-mask head" in why, why
