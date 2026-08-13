from __future__ import annotations

import numpy as np
import pytest
import torch

from docscanner.eval.metrics import corner_metrics, psnr, ssim
from docscanner.eval.ocr import cer, normalize_text, wer
from docscanner.models.corner_nets import (
    CornerHeatmapNet,
    CornerRegressor,
    coords_to_heatmaps,
    soft_argmax_2d,
)
from docscanner.models.enhance_unet import DocEnhanceNet, background_prior
from docscanner.models.losses import (
    MSSSIM,
    SSIM,
    CharbonnierLoss,
    CornerCoordLoss,
    HeatmapLoss,
    SobelGradientLoss,
    build_enhancement_loss,
    cyclic_permutations,
)

torch.manual_seed(0)


@pytest.mark.parametrize("size", [64, 96, 128])
def test_enhance_net_preserves_resolution(size):
    net = DocEnhanceNet(base=8, depth=3).eval()
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        y = net(x, clamp=True)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_enhance_net_is_fully_convolutional_on_rectangles():
    net = DocEnhanceNet(base=8, depth=3).eval()
    with torch.no_grad():
        y = net(torch.rand(1, 3, 96, 160))
    assert y.shape == (1, 3, 96, 160)


def test_enhance_net_starts_as_identity():
    net = DocEnhanceNet(base=8, depth=3).eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = net(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_background_prior_tracks_illumination():
    x = torch.ones(1, 3, 64, 64)
    x[..., :32] *= 0.4
    bg = background_prior(x, kernel=17)
    assert float(bg[..., :24].mean()) < float(bg[..., 40:].mean())


def test_dropout_only_affects_training_mode():
    net = DocEnhanceNet(base=8, depth=3, dropout=0.5)
    x = torch.rand(1, 3, 64, 64)
    net.eval()
    with torch.no_grad():
        a, b = net(x), net(x)
    assert torch.allclose(a, b)


def test_regressor_shape_and_prior():
    net = CornerRegressor(base=8, depth=4, input_size=64).eval()
    with torch.no_grad():
        out = net(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 4, 2)
    assert float(out.min()) > 0.05 and float(out.max()) < 0.95


def test_heatmap_net_shape():
    net = CornerHeatmapNet(base=8, depth=3, out_stride=4).eval()
    with torch.no_grad():
        out = net(torch.rand(2, 3, 64, 64))
    assert out.shape[:2] == (2, 4)
    assert out.shape[-1] == 64 // 4


@pytest.mark.parametrize("pad", [0.0, 0.12])
def test_heatmap_round_trip_recovers_coordinates(pad):
    coords = torch.tensor([[[0.20, 0.30], [0.80, 0.25], [0.75, 0.85], [0.15, 0.90]]])
    hm = coords_to_heatmaps(coords, size=96, sigma=1.6, pad=pad)
    back = soft_argmax_2d(hm, pad=pad)
    assert torch.allclose(back, coords, atol=0.02), (back, coords)


def test_soft_argmax_handles_out_of_frame_corners():
    coords = torch.tensor([[[-0.06, 0.10], [1.05, 0.12], [1.02, 0.94], [-0.03, 0.9]]])
    hm = coords_to_heatmaps(coords, size=96, sigma=1.6, pad=0.12)
    back = soft_argmax_2d(hm, pad=0.12)
    assert torch.allclose(back, coords, atol=0.03)


def test_ssim_and_msssim_are_one_for_identical_images():
    x = torch.rand(1, 3, 96, 96)
    assert float(SSIM()(x, x)) == pytest.approx(1.0, abs=1e-4)
    assert float(MSSSIM()(x, x)) == pytest.approx(1.0, abs=1e-3)


def test_ssim_decreases_with_noise():
    x = torch.rand(1, 3, 96, 96)
    noisy = (x + torch.randn_like(x) * 0.2).clamp(0, 1)
    assert float(SSIM()(x, noisy)) < float(SSIM()(x, x))


def test_our_ssim_agrees_with_skimage():
    ski = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(0)
    a = rng.random((128, 128, 3)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.08, a.shape), 0, 1).astype(np.float32)
    ours = ssim(a, b)
    theirs = ski.structural_similarity(a, b, channel_axis=2, data_range=1.0,
                                       gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False)
    assert abs(ours - theirs) < 0.02, (ours, theirs)


def test_our_psnr_agrees_with_skimage():
    ski = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(1)
    a = rng.random((64, 64, 3)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1).astype(np.float32)
    assert abs(psnr(a, b) - ski.peak_signal_noise_ratio(a, b, data_range=1.0)) < 0.05


def test_sobel_loss_penalises_blur():
    x = torch.zeros(1, 1, 64, 64)
    x[..., 20:44, 20:44] = 1.0
    blurred = torch.nn.functional.avg_pool2d(x, 5, 1, 2)
    loss = SobelGradientLoss(channels=1)
    assert float(loss(blurred, x)) > float(loss(x, x))


def test_charbonnier_is_zero_at_the_optimum():
    x = torch.rand(2, 3, 16, 16)
    assert float(CharbonnierLoss()(x, x)) < 2e-3


@pytest.mark.parametrize("kind", ["mse", "l1", "l1_ssim", "combined"])
def test_loss_presets_build_and_run(kind):
    loss = build_enhancement_loss(kind)
    pred, target = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    total, comps = loss(pred, target)
    assert torch.isfinite(total) and float(total) > 0
    assert comps


def test_cyclic_permutation_invariance():
    c = torch.tensor([[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]])
    perms = cyclic_permutations(c)
    assert perms.shape == (1, 4, 4, 2)
    loss = CornerCoordLoss("wing", permutation_invariant=True)
    assert float(loss(c, c)) < 1e-6
    assert float(loss(c, c.roll(-2, dims=1))) < 1e-6
    assert float(loss(c, c * 0.5)) > 1e-3


def test_permutation_invariance_can_be_switched_off():
    c = torch.tensor([[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]])
    loss = CornerCoordLoss("l1", permutation_invariant=False)
    assert float(loss(c, c.roll(-1, dims=1))) > 1e-3


def test_heatmap_loss_runs_and_decreases_for_better_predictions():
    coords = torch.tensor([[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]])
    good = torch.logit(coords_to_heatmaps(coords, 32, pad=0.12).clamp(1e-4, 1 - 1e-4))
    bad = torch.randn(1, 4, 32, 32) * 0.1 - 4
    loss = HeatmapLoss()
    assert float(loss(good, coords)[0]) < float(loss(bad, coords)[0])


def test_corner_metrics_on_perfect_predictions():
    gt = np.tile(np.array([[10, 10], [90, 10], [90, 60], [10, 60]], np.float64), (5, 1, 1))
    sc = corner_metrics(gt.copy(), gt, (100, 70))
    assert sc.mce_px == pytest.approx(0.0)
    assert sc.success[4.0] == pytest.approx(1.0)
    assert sc.iou > 0.98


def test_corner_metrics_success_requires_all_four_corners():
    gt = np.array([[[10, 10], [90, 10], [90, 60], [10, 60]]], np.float64)
    pred = gt.copy()
    pred[0, 2] += 30
    sc = corner_metrics(pred, gt, (100, 70), thresholds=(8.0,))
    assert sc.success[8.0] == 0.0


def test_match_cyclic_absorbs_ordering_shift():
    gt = np.array([[[10, 10], [90, 10], [90, 60], [10, 60]]], np.float64)
    pred = np.roll(gt, -1, axis=1)
    assert corner_metrics(pred, gt, (100, 70), cyclic=True).mce_px < 1e-6
    assert corner_metrics(pred, gt, (100, 70), cyclic=False).mce_px > 10


def test_cer_and_wer_basics():
    assert cer("hello world", "hello world") == pytest.approx(0.0)
    assert wer("hello world", "hello world") == pytest.approx(0.0)
    assert cer("abcdefghij", "abcdefghix") == pytest.approx(0.1)
    assert wer("a b c d", "a b c") == pytest.approx(0.25)


def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("Hello,  World!") == "hello world"
