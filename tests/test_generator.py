from __future__ import annotations

import cv2
import numpy as np
import pytest

from docscanner.data.corpus import BackgroundCorpus, ScanCorpus
from docscanner.data.datasets import EnhancementDataset, resize_with_corners
from docscanner.data.degrade import DegradationConfig
from docscanner.data.generator import SyntheticSampleGenerator, rot90_with_points
from docscanner.utils.geometry import order_corners


def _make_page(w: int = 300, h: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    page = np.full((h, w, 3), 250, np.uint8)
    for i in range(14):
        y = 30 + i * 25
        x1 = 25 + int(rng.integers(0, 20))
        x2 = w - 25 - int(rng.integers(0, 40))
        cv2.line(page, (x1, y), (x2, y), (30, 30, 40), 3)
    cv2.rectangle(page, (40, h - 90), (w - 40, h - 40), (60, 90, 200), -1)
    return page


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpora")
    scans = root / "scans"
    bgs = root / "bg"
    scans.mkdir()
    bgs.mkdir()
    for i in range(3):
        cv2.imwrite(str(scans / f"p{i}.png"), _make_page(seed=i))
    rng = np.random.default_rng(0)
    for i in range(3):
        tex = (rng.random((160, 160, 3)) * 120 + 40).astype(np.uint8)
        cv2.imwrite(str(bgs / f"t{i}.png"), tex)
    return (ScanCorpus.from_dir(scans, name="scans"),
            BackgroundCorpus.from_dir(bgs, name="bg"))


@pytest.fixture(scope="module")
def generator(corpora):
    scans, bgs = corpora
    return SyntheticSampleGenerator(scans, bgs, photo_long_side=480, page_long_side=320)


@pytest.mark.parametrize("seed", range(6))
def test_homographies_compose_exactly(generator, seed):
    s = generator.generate(np.random.default_rng(seed))
    dw, dh = s.meta["doc_size"]
    pw, ph = s.page_size

    comp = s.h_rect @ s.h_doc2photo
    comp = comp / comp[2, 2]
    src = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]],
                   np.float32).reshape(-1, 1, 2)
    got = cv2.perspectiveTransform(src, comp).reshape(4, 2)
    want = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    assert np.abs(got - want).max() < 1e-2


@pytest.mark.parametrize("seed", range(6))
def test_corner_labels_land_on_the_page(generator, seed):
    s = generator.generate(np.random.default_rng(seed), want_rectified=False)
    dw, dh = s.meta["doc_size"]
    doc_corners = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]],
                           np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(doc_corners, s.h_doc2photo).reshape(4, 2)
    assert s.corners_content is not None
    assert np.allclose(order_corners(s.corners), order_corners(s.corners_content))
    if not any(n == "rot90" for n, _ in s.trace.ops):
        assert np.abs(np.sort(projected, axis=0)
                      - np.sort(s.corners_content, axis=0)).max() < 1.0


def test_identity_pipeline_reproduces_the_target(corpora):
    scans, bgs = corpora
    cfg = DegradationConfig(
        p_drop_shadow=0, p_paper_texture=0, p_downscale=0, p_illumination=0,
        p_soft_shadow=0, p_glare=0, p_vignette=0, p_chromatic=0, p_blur=0,
        p_motion_blur=0, p_defocus=0, p_noise=0, p_jpeg=0, p_auto_exposure=0,
        p_page_curl=0, p_rot90=0,
        brightness=(0, 0), contrast=(1, 1), gamma=(1, 1),
        color_cast_r=(1, 1), color_cast_b=(1, 1), saturation=(1, 1),
        gaussian_sigma=(0, 0),
    )
    gen = SyntheticSampleGenerator(scans, bgs, cfg=cfg, photo_long_side=700,
                                  page_long_side=320)
    s = gen.generate(np.random.default_rng(3))
    a = s.rectified[12:-12, 12:-12].astype(np.float32)
    b = s.target[12:-12, 12:-12].astype(np.float32)
    mse = float(((a - b) ** 2).mean())
    psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
    assert psnr > 22.0, f"identity pipeline only reached {psnr:.1f} dB"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rot90_moves_points_with_pixels(k):
    img = np.zeros((30, 50, 3), np.uint8)
    img[7, 41] = 255
    pts = np.array([[41.0, 7.0]], np.float32)
    out, moved = rot90_with_points(img, pts, k)
    ys, xs = np.where(out[..., 0] == 255)
    assert (abs(xs[0] - moved[0, 0]) < 1e-4) and (abs(ys[0] - moved[0, 1]) < 1e-4)


def test_quarter_turns_keep_pairs_aligned(corpora):
    scans, bgs = corpora
    cfg = DegradationConfig(p_rot90=1.0)
    gen = SyntheticSampleGenerator(scans, bgs, cfg=cfg, photo_long_side=600,
                                  page_long_side=320)
    shifts = []
    for seed in range(5):
        s = gen.generate(np.random.default_rng(seed))
        a = cv2.cvtColor(s.rectified, cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor(s.target, cv2.COLOR_RGB2GRAY).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(a - a.mean(), b - b.mean())
        shifts.append(abs(dx) + abs(dy))
    assert np.mean(shifts) < 4.0, f"mean misalignment {np.mean(shifts):.2f}px"


def test_resize_with_corners_scales_labels():
    img = np.zeros((100, 200, 3), np.uint8)
    corners = np.array([[0, 0], [199, 0], [199, 99], [0, 99]], np.float32)
    out, pts = resize_with_corners(img, corners, 50)
    assert out.shape[:2] == (50, 50)
    assert np.allclose(pts[1, 0], 199 * 50 / 200, atol=1e-3)
    assert np.allclose(pts[2, 1], 99 * 50 / 100, atol=1e-3)


def test_enhancement_dataset_returns_aligned_tensors(generator):
    ds = EnhancementDataset(generator, length=6, crop=96, crops_per_photo=3,
                            page_size_range=(160, 260))
    x, y = ds[0]
    assert x.shape == (3, 96, 96) and y.shape == (3, 96, 96)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    a = (x.numpy().transpose(1, 2, 0) * 255).astype(np.float32).mean(axis=2)
    b = (y.numpy().transpose(1, 2, 0) * 255).astype(np.float32).mean(axis=2)
    (dx, dy), _ = cv2.phaseCorrelate(a - a.mean(), b - b.mean())
    assert abs(dx) < 4 and abs(dy) < 4


def test_dataset_epochs_change_the_samples(generator):
    ds = EnhancementDataset(generator, length=4, crop=64, crops_per_photo=2)
    a, _ = ds[0]
    ds.set_epoch(1)
    b, _ = ds[0]
    assert not np.array_equal(a.numpy(), b.numpy())


@pytest.mark.parametrize("seed", range(6))
def test_distractor_pages_never_capture_the_label(corpora, seed):
    scans, bgs = corpora
    gen = SyntheticSampleGenerator(
        scans, bgs, cfg=DegradationConfig(p_distractor_page=1.0, p_rot90=0.0),
        photo_long_side=480, page_long_side=320)
    s = gen.generate(np.random.default_rng(seed), want_rectified=False)
    dw, dh = s.meta["doc_size"]
    doc_corners = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]],
                           np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(doc_corners, s.h_doc2photo).reshape(4, 2)
    assert np.abs(np.sort(projected, axis=0)
                  - np.sort(s.corners_content, axis=0)).max() < 1.0


def test_distractor_pages_actually_alter_the_scene(corpora):
    scans, bgs = corpora
    kw = {"photo_long_side": 480, "page_long_side": 320}
    off = SyntheticSampleGenerator(
        scans, bgs, cfg=DegradationConfig(p_distractor_page=0.0), **kw)
    on = SyntheticSampleGenerator(
        scans, bgs, cfg=DegradationConfig(p_distractor_page=1.0), **kw)

    def bright_fraction(g):
        vals = []
        for seed in range(8):
            s = g.generate(np.random.default_rng(seed), want_rectified=False)
            vals.append(float((s.photo.mean(axis=2) > 170).mean()))
        return float(np.mean(vals))

    assert bright_fraction(on) > bright_fraction(off) + 0.01


def test_distractor_labels_stay_valid(corpora):
    scans, bgs = corpora
    gen = SyntheticSampleGenerator(
        scans, bgs, cfg=DegradationConfig(p_distractor_page=1.0),
        photo_long_side=480, page_long_side=320)
    for seed in range(8):
        s = gen.generate(np.random.default_rng(seed), want_rectified=False)
        assert s.corners.shape == (4, 2) and np.isfinite(s.corners).all()
        assert np.allclose(s.corners, order_corners(s.corners))


def test_distractor_source_is_cached_and_downscaled(corpora):
    scans, bgs = corpora
    gen = SyntheticSampleGenerator(scans, bgs, photo_long_side=480, page_long_side=320)
    rng = np.random.default_rng(0)
    first = gen._distractor_source(rng)
    assert first is not None
    assert max(first.shape[:2]) <= 420
    assert gen._distractor_cache, "cache was not populated"
    n = len(gen._distractor_cache)
    for _ in range(5):
        gen._distractor_source(rng)
    assert len(gen._distractor_cache) == n, "cache should not grow per call"


def test_inner_frame_stays_clear_of_the_page_edge(corpora):
    scans, bgs = corpora
    cfg = DegradationConfig(p_inner_frame=1.0)
    gen = SyntheticSampleGenerator(scans, bgs, cfg=cfg,
                                   photo_long_side=480, page_long_side=320)
    doc = np.full((400, 300, 3), 240, np.uint8)

    for seed in range(12):
        framed = gen._draw_inner_frame(doc, np.random.default_rng(seed))
        assert framed.shape == doc.shape
        changed = np.argwhere(np.any(framed != doc, axis=2))
        if changed.size == 0:
            continue
        (y0, x0), (y1, x1) = changed.min(0), changed.max(0)
        h, w = doc.shape[:2]
        lo = min(cfg.inner_frame_margin) * 0.5
        assert y0 >= h * lo and x0 >= w * lo, f"seed {seed}: frame touches top/left"
        assert y1 <= h * (1 - lo) and x1 <= w * (1 - lo), \
            f"seed {seed}: frame touches bottom/right"
        assert framed[changed[:, 0], changed[:, 1]].mean() < doc.mean()


def test_inner_frame_keeps_input_and_target_aligned(corpora):
    scans, bgs = corpora
    gen = SyntheticSampleGenerator(
        scans, bgs, cfg=DegradationConfig(p_inner_frame=1.0, p_blur=0.0,
                                          p_noise=0.0, p_jpeg=0.0),
        photo_long_side=480, page_long_side=320)
    s = gen.generate(np.random.default_rng(11), want_rectified=True)
    assert s.rectified.shape == s.target.shape


def test_corner_policy_enables_the_frame_and_enhancement_does_not():
    assert DegradationConfig.for_corners().p_inner_frame > 0.0
    assert DegradationConfig.for_enhancement().p_inner_frame == 0.0


_CORNER_ONLY = ("p_page_tint", "p_page_band", "p_page_stack", "p_binding",
                "p_gutter", "p_round_corners", "p_dog_ear", "p_corner_occluder",
                "p_backlit", "p_pale_background", "p_ruled_paper")


def test_enhancement_policy_enables_no_corner_only_augmentation():
    cfg = DegradationConfig.for_enhancement()
    on = [k for k in _CORNER_ONLY if getattr(cfg, k, 0.0) > 0.0]
    assert not on, f"enhancement policy must leave these off: {on}"
    corners = DegradationConfig.for_corners()
    assert [k for k in _CORNER_ONLY if getattr(corners, k, 0.0) > 0.0], \
        "the corner policy should actually use them"


def test_corner_only_augmentation_does_not_touch_the_enhancement_stream(corpora):
    scans, bgs = corpora

    class WithoutTheKnobs:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            if name in _CORNER_ONLY:
                raise AttributeError(name)
            return getattr(object.__getattribute__(self, "_inner"), name)

    def sample(cfg):
        gen = SyntheticSampleGenerator(scans, bgs, cfg=cfg, photo_long_side=320,
                                       page_long_side=192)
        rng = np.random.default_rng(101)
        return [gen.generate(rng, want_rectified=True, page_size=(96, 96)).photo.copy()
                for _ in range(3)]

    base = DegradationConfig.for_enhancement()
    for a, b in zip(sample(base), sample(WithoutTheKnobs(DegradationConfig.for_enhancement()))):
        assert np.array_equal(a, b), "a corner-only knob shifted the random stream"


@pytest.mark.parametrize("knob", _CORNER_ONLY)
def test_every_corner_augmentation_keeps_a_usable_label(corpora, knob):
    from docscanner.utils.geometry import quad_is_convex

    scans, bgs = corpora
    cfg = DegradationConfig.for_corners()
    for k in _CORNER_ONLY:
        setattr(cfg, k, 0.0)
    setattr(cfg, knob, 1.0)
    gen = SyntheticSampleGenerator(scans, bgs, cfg=cfg, photo_long_side=320,
                                   page_long_side=192)
    rng = np.random.default_rng(5)
    for _ in range(3):
        s = gen.generate(rng, want_rectified=False)
        q = np.asarray(s.corners_content, np.float32)
        assert q.shape == (4, 2) and np.isfinite(q).all(), knob
        assert quad_is_convex(order_corners(q)), f"{knob} produced a degenerate quad"
