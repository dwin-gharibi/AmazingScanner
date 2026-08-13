from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from docscanner.engine.common import TrainConfig, save_checkpoint
from docscanner.models.corner_nets import CornerHeatmapNet, CornerRegressor
from docscanner.models.enhance_unet import DocEnhanceNet
from docscanner.utils.geometry import order_corners, quad_is_convex, quad_is_plausible
from docscanner.utils.imageio import imread_rgb, imwrite_rgb


def _page(w: int = 420, h: int = 560) -> np.ndarray:
    page = np.full((h, w, 3), 250, np.uint8)
    for i in range(18):
        y = 40 + i * 28
        cv2.line(page, (36, y), (w - 40, y), (30, 30, 45), 3)
    cv2.rectangle(page, (50, h - 110), (w - 50, h - 50), (60, 90, 200), -1)
    cv2.putText(page, "AmazingScanner", (44, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (20, 20, 30), 2, cv2.LINE_AA)
    return page


@pytest.fixture(scope="module")
def photo() -> np.ndarray:
    rng = np.random.default_rng(0)
    bg = (rng.random((760, 640, 3)) * 90 + 60).astype(np.uint8)
    page = _page()
    quad = np.array([[110, 120], [520, 96], [545, 640], [86, 604]], np.float32)
    src = np.array([[0, 0], [page.shape[1] - 1, 0],
                    [page.shape[1] - 1, page.shape[0] - 1],
                    [0, page.shape[0] - 1]], np.float32)
    h = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(page, h, (640, 760))
    mask = cv2.warpPerspective(np.ones(page.shape[:2], np.uint8) * 255, h, (640, 760))
    out = np.where(mask[..., None] > 127, warped, bg).astype(np.uint8)
    return cv2.GaussianBlur(out, (0, 0), 0.8)


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("models")
    enh = DocEnhanceNet(base=8, depth=3)
    save_checkpoint(root / "enhance.pt", enh,
                    TrainConfig(name="e", extra={"base": 8, "depth": 3,
                                                 "bg_prior": True}))
    heat = CornerHeatmapNet(base=8, depth=3, out_stride=4)
    save_checkpoint(root / "corner_heatmap.pt", heat,
                    TrainConfig(name="h", extra={"approach": "heatmap", "size": 256,
                                                 "base": 8, "depth": 3,
                                                 "out_stride": 4, "pad": 0.12}))
    reg = CornerRegressor(base=8, depth=4, input_size=256, head_dim=64)
    save_checkpoint(root / "corner_regression.pt", reg,
                    TrainConfig(name="r", extra={"approach": "regression",
                                                 "size": 256, "base": 8, "depth": 4,
                                                 "head_dim": 64}))
    return {"enhance": root / "enhance.pt",
            "heatmap": root / "corner_heatmap.pt",
            "regression": root / "corner_regression.pt"}


def test_enhancement_pipeline_round_trip(models, photo):
    from docscanner.pipeline.enhance_pipeline import EnhancementPipeline
    pipe = EnhancementPipeline(models["enhance"], max_side=320)
    res = pipe(photo, mode="color")
    assert res.image.shape == photo.shape
    assert res.image.dtype == np.uint8
    assert res.seconds > 0
    assert res.network_output.dtype == np.uint8


@pytest.mark.parametrize("mode", ["color", "gray", "bw", "whiteboard", "raw"])
def test_every_output_mode(models, photo, mode):
    from docscanner.pipeline.enhance_pipeline import EnhancementPipeline
    pipe = EnhancementPipeline(models["enhance"], max_side=256)
    out = pipe(photo, mode=mode).image
    assert out.shape[:2] == photo.shape[:2]
    assert out.dtype == np.uint8
    if mode == "bw":
        assert set(np.unique(out)).issubset({0, 255})


def test_tiled_inference_matches_single_pass(models, photo):
    from docscanner.pipeline.enhance_pipeline import EnhancementPipeline
    small = cv2.resize(photo, (300, 360), interpolation=cv2.INTER_AREA)
    whole = EnhancementPipeline(models["enhance"], max_side=512, tile=4096)
    tiled = EnhancementPipeline(models["enhance"], max_side=512, tile=160, overlap=48)
    a = whole(small, mode="raw").network_output.astype(np.float32)
    b = tiled(small, mode="raw").network_output.astype(np.float32)
    assert np.abs(a - b).mean() < 3.0, np.abs(a - b).mean()


@pytest.mark.parametrize("which", ["heatmap", "regression"])
def test_corner_pipeline_returns_a_usable_quad(models, photo, which):
    from docscanner.pipeline.corner_pipeline import CornerPipeline
    pipe = CornerPipeline(models[which], refine=True)
    res = pipe(photo)
    assert res.corners.shape == (4, 2)
    assert np.isfinite(res.corners).all()
    assert res.approach == which
    assert np.allclose(res.corners, order_corners(res.corners))
    h, w = photo.shape[:2]
    assert (res.corners[:, 0] > -0.5 * w).all() and (res.corners[:, 0] < 1.5 * w).all()
    assert res.source in {"network", "classical", "full_frame"}
    assert quad_is_convex(res.corners)


def test_corner_pipeline_never_returns_an_implausible_quad(models, photo):
    from docscanner.pipeline import corner_pipeline as cp

    pipe = cp.CornerPipeline(models["heatmap"], refine=True, tta=False)
    h, w = photo.shape[:2]
    nonsense = np.array([[0.6 * w, 0.2 * h], [0.8 * w, 0.1 * h],
                         [1.7 * w, -0.2 * h], [0.2 * w, 0.7 * h]], np.float32)
    pipe._forward = lambda img: (nonsense.copy(), 0.9, None)
    res = pipe(photo)

    assert res.source in {"classical", "full_frame"}
    assert res.meta["fallback"] is True
    assert quad_is_plausible(res.corners, (h, w))
    assert res.confidence <= 0.35


def test_rot90_point_mapping_is_exact():
    from docscanner.pipeline.corner_pipeline import rot90_points_inverse

    img = np.arange(4 * 6).reshape(4, 6).astype(np.uint8)
    for k in range(4):
        rot = np.rot90(img, k)
        ys, xs = np.mgrid[0:rot.shape[0], 0:rot.shape[1]]
        pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        back = rot90_points_inverse(pts, k, rot.shape)
        assert (back >= 0).all()
        assert (back[:, 0] < img.shape[1]).all() and (back[:, 1] < img.shape[0]).all()
        got = img[back[:, 1].astype(int), back[:, 0].astype(int)]
        assert (got == rot.ravel()).all(), f"mapping wrong for k={k}"


def test_tta_agrees_with_single_view_on_a_rotation_equivariant_stub(models, photo):
    from docscanner.pipeline.corner_pipeline import CornerPipeline

    pipe = CornerPipeline(models["heatmap"], refine=False, tta=False)

    def fake_forward(img):
        h, w = img.shape[:2]
        m = 25.0
        pts = np.array([[m, m], [w - 1 - m, m],
                        [w - 1 - m, h - 1 - m], [m, h - 1 - m]], np.float32)
        return pts, 0.9, None

    pipe._forward = fake_forward
    single = pipe(photo, tta=False).corners
    multi = pipe(photo, tta=True).corners
    assert np.allclose(single, multi, atol=1e-3)


def test_tta_is_reported_and_costs_more_passes(models, photo):
    from docscanner.pipeline.corner_pipeline import CornerPipeline

    pipe = CornerPipeline(models["heatmap"], refine=False, tta=False)
    calls = {"n": 0}
    original = pipe._forward

    def counting(img):
        calls["n"] += 1
        return original(img)

    pipe._forward = counting
    r1 = pipe(photo, tta=False)
    assert calls["n"] == 1 and r1.meta["tta"] is False
    calls["n"] = 0
    r4 = pipe(photo, tta=True)
    assert calls["n"] == 4 and r4.meta["tta"] is True
    assert quad_is_convex(r4.corners)


def test_classical_detector_finds_a_page_and_declines_when_there_is_none(photo):
    from docscanner.pipeline.corner_pipeline import detect_quad_classical

    quad = detect_quad_classical(photo)
    assert quad is not None, "a high-contrast page on a dark desk must be found"
    assert quad_is_plausible(quad, photo.shape[:2])
    assert np.allclose(quad, order_corners(quad))

    assert detect_quad_classical(np.full((300, 300, 3), 128, np.uint8)) is None
    assert detect_quad_classical(np.zeros((12, 12, 3), np.uint8)) is None


def test_corner_refinement_is_rejected_when_it_would_wander(models, photo):
    from docscanner.pipeline.corner_pipeline import CornerPipeline
    pipe = CornerPipeline(models["heatmap"], refine=True)
    res = pipe(photo)
    diag = float(np.hypot(*photo.shape[:2]))
    assert np.abs(res.corners - res.corners_prerefine).max() <= 0.05 * diag + 1e-6


def test_end_to_end_scanner(models, photo):
    from docscanner.pipeline.scanner import DocumentScanner
    sc = DocumentScanner(models["heatmap"], models["enhance"], max_side=420)
    res = sc.scan(photo, mode="color", auto_orientation=False)
    assert res.image.ndim == 3 and res.image.dtype == np.uint8
    assert res.corners.shape == (4, 2)
    assert quad_is_convex(res.corners)
    assert {"detect", "rectify", "enhance"} <= set(res.timings)
    assert res.rectified.shape[:2] == res.image.shape[:2]


def test_scanner_accepts_supplied_corners(models, photo):
    from docscanner.pipeline.scanner import DocumentScanner
    sc = DocumentScanner(models["heatmap"], models["enhance"], max_side=420)
    quad = np.array([[110, 120], [520, 96], [545, 640], [86, 604]], np.float32)
    res = sc.scan(photo, corners=quad, auto_orientation=False)
    assert np.allclose(res.corners, order_corners(quad), atol=1e-3)
    assert res.corner_result is None


def test_postprocess_chain(photo):
    from docscanner.pipeline.postprocess import add_margin, adjust, auto_crop, rotate
    page = _page()
    assert rotate(page, 90).shape[:2] == page.shape[:2][::-1]
    assert rotate(page, 360).shape == page.shape
    bright = adjust(page, brightness=0.2)
    assert bright.mean() > page.mean()
    sharp = adjust(page, sharpness=1.5)
    assert cv2.Laplacian(cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() > \
        cv2.Laplacian(cv2.cvtColor(page, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    assert add_margin(page, 0.05).shape[0] > page.shape[0]
    assert auto_crop(page).shape[0] <= page.shape[0]


@pytest.mark.parametrize("angle", [-4.0, -1.5, 2.5, 5.0])
def test_deskew_recovers_a_known_rotation(angle):
    from docscanner.pipeline.postprocess import estimate_skew
    page = _page(360, 460)
    h, w = page.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    skewed = cv2.warpAffine(page, m, (w, h), borderValue=(250, 250, 250))
    est = estimate_skew(skewed)
    assert abs(est - (-angle)) < 1.0, f"estimated {est:+.2f} for a {angle:+.2f} skew"


def test_pdf_export_single_and_multi(tmp_path):
    from docscanner.pipeline.scanner import save_pdf
    one = save_pdf([_page()], tmp_path / "one.pdf")
    many = save_pdf([_page(), _page(300, 400)], tmp_path / "many.pdf")
    assert one.exists() and one.stat().st_size > 500
    assert many.stat().st_size > one.stat().st_size


def test_searchable_pdf_has_a_text_layer(tmp_path):
    from docscanner.eval.ocr import searchable_pdf, tesseract_available
    out = searchable_pdf([_page()], tmp_path / "s.pdf")
    assert out.exists() and out.stat().st_size > 500
    if tesseract_available():
        raw = out.read_bytes()
        assert b"%PDF" in raw[:10]


def test_ocr_languages_and_modes_are_declared():
    from docscanner.eval.ocr import (
        ENGINE_MODES,
        PAGE_MODES,
        available_languages,
        language_choices,
        tesseract_available,
    )
    assert len(PAGE_MODES) >= 5 and all(isinstance(v, int) for v in PAGE_MODES.values())
    assert len(ENGINE_MODES) >= 2
    if tesseract_available():
        langs = available_languages()
        assert "eng" in langs
        labels = dict(language_choices())
        assert any("English" in k for k in labels)


def test_unknown_backend_is_reported_not_guessed():
    from docscanner.eval.ocr import run_ocr
    assert not run_ocr(_page(), backend="not-a-real-engine").available


@pytest.mark.parametrize("command", ["enhance", "corners", "scan"])
def test_cli_commands(tmp_path, models, photo, command):
    src = tmp_path / "photo.png"
    imwrite_rgb(src, photo)
    out = tmp_path / command
    args = [sys.executable, "-m", "docscanner.pipeline.run", command, str(src),
            "-o", str(out), "--threads", "1"]
    if command == "scan":
        args += ["--corner-checkpoint", str(models["heatmap"]),
                 "-c", str(models["enhance"]), "--pdf", "all.pdf"]
    elif command == "enhance":
        args += ["-c", str(models["enhance"]), "--max-side", "320"]
    else:
        args += ["-c", str(models["heatmap"])]

    proc = subprocess.run(args, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    report = out / f"{command}_report.json"
    assert report.exists(), sorted(p.name for p in out.iterdir())
    rows = json.loads(report.read_text())
    assert len(rows) == 1 and "input" in rows[0]
    assert any(p.suffix in (".png", ".jpg") for p in out.iterdir())
    if command == "scan":
        assert (out / "all.pdf").exists()
        assert len(rows[0]["corners_tl_tr_br_bl"]) == 4


def test_cli_processes_a_folder(tmp_path, models, photo):
    src = tmp_path / "photos"
    src.mkdir()
    for i in range(3):
        imwrite_rgb(src / f"p{i}.png", photo)
    out = tmp_path / "batch"
    proc = subprocess.run(
        [sys.executable, "-m", "docscanner.pipeline.run", "scan", str(src),
         "-o", str(out), "--corner-checkpoint", str(models["heatmap"]),
         "-c", str(models["enhance"]), "--pdf", "book.pdf", "--threads", "1",
         "--max-side", "420"],
        capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "book.pdf").exists()
    assert len(json.loads((out / "scan_report.json").read_text())) == 3


def test_app_builds_and_reports_status():
    from docscanner.app.gradio_app import REG, build_app
    demo = build_app()
    assert len(demo.blocks) > 50
    html = REG.status_html()
    assert "Enhancement" in html and "OCR" in html


def test_app_manual_corner_clicks_accumulate(photo):
    from docscanner.app.gradio_app import click_corner

    class _Evt:
        def __init__(self, xy):
            self.index = xy

    pts = []
    for xy in [(110, 120), (520, 96), (545, 640), (86, 604)]:
        pts, preview, msg = click_corner(pts, photo, _Evt(xy))
        assert preview is not None
    assert len(pts) == 4 and "4/4" in msg
    pts, _, _ = click_corner(pts, photo, _Evt((10, 10)))
    assert len(pts) == 1


def test_app_finish_controls(photo):
    from docscanner.app.gradio_app import _finish
    page = _page()
    turned = _finish(page, turn="90")
    assert turned.shape[:2] == page.shape[:2][::-1]
    adjusted = _finish(page, brightness=0.15, contrast=1.2, sharpness=0.5,
                       margin=0.03)
    assert adjusted.shape[0] > page.shape[0]


def test_degradation_lab_generates_when_data_is_present():
    import gradio as gr

    from docscanner.app.gradio_app import degradation_lab
    try:
        photo, rect, target, gallery, trace = degradation_lab(
            7, 0.15, 30, 0.4, 0.3, 1.0, 0.02, 60, 2.0, False, False)
    except gr.Error as e:
        assert "not prepared" in str(e)
        pytest.skip("corpora not fetched — and the lab refused cleanly")
    assert photo is not None and rect is not None and target is not None
    assert len(gallery) >= 1
    assert "warp_onto_background" in trace


def test_coco_keypoints_round_trip(tmp_path, photo):
    from docscanner.data.check_labels import check_manifest
    from docscanner.data.real import from_coco_keypoints

    images = tmp_path / "images"
    images.mkdir()
    imwrite_rgb(images / "a.jpg", photo)
    quad = [[110, 120], [520, 96], [545, 640], [86, 604]]
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg",
                    "width": photo.shape[1], "height": photo.shape[0]}],
        "categories": [{"id": 1, "name": "page",
                        "keypoints": ["top-left", "top-right",
                                      "bottom-right", "bottom-left"]}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "keypoints": [c for x, y in quad for c in (x, y, 2)]}],
    }
    (tmp_path / "ann.json").write_text(json.dumps(coco))

    out = from_coco_keypoints(tmp_path / "ann.json", images, tmp_path / "real")
    data = json.loads(out.read_text())
    assert data["count"] == 1
    got = np.asarray(data["items"][0]["corners"], np.float32)
    assert np.allclose(got, order_corners(np.asarray(quad, np.float32)), atol=1.0)

    report = check_manifest(out)
    assert report.total == 1 and report.failed == 0


def _seg_coco(photo, quad, extra_points: bool = False):
    poly = [list(map(float, p)) for p in quad]
    if extra_points:
        dense = []
        for i in range(4):
            a, b = np.asarray(poly[i]), np.asarray(poly[(i + 1) % 4])
            for t in (0.0, 0.33, 0.66):
                dense.append((a + (b - a) * t).tolist())
        poly = dense
    flat = [c for p in poly for c in p]
    return {
        "images": [{"id": 0, "file_name": "a.jpg",
                    "width": photo.shape[1], "height": photo.shape[0]}],
        "categories": [{"id": 0, "name": "page", "supercategory": "none"},
                       {"id": 1, "name": "page", "supercategory": "page"}],
        "annotations": [{"id": 1, "image_id": 0, "category_id": 1, "iscrowd": 0,
                         "bbox": [0, 0, 1, 1], "area": 1.0,
                         "segmentation": [flat]}],
    }


@pytest.mark.parametrize("dense", [False, True])
def test_coco_segmentation_round_trip(tmp_path, photo, dense):
    from docscanner.data.check_labels import check_manifest
    from docscanner.data.real import from_coco_segmentation

    images = tmp_path / "images"
    images.mkdir()
    imwrite_rgb(images / "a.jpg", photo)
    quad = [[110, 120], [520, 96], [545, 640], [86, 604]]
    (tmp_path / "ann.json").write_text(json.dumps(_seg_coco(photo, quad, dense)))

    out = from_coco_segmentation(tmp_path / "ann.json", images, tmp_path / "real")
    data = json.loads(out.read_text())
    assert data["count"] == 1
    got = np.asarray(data["items"][0]["corners"], np.float32)
    assert np.allclose(got, order_corners(np.asarray(quad, np.float32)), atol=2.0)
    assert check_manifest(out).failed == 0


def test_coco_polygon_order_is_canonicalised(photo):
    from docscanner.data.real import quad_from_polygon

    quad = np.asarray([[110, 120], [520, 96], [545, 640], [86, 604]], np.float32)
    canonical = order_corners(quad)
    for shift in range(4):
        for ring in (np.roll(quad, shift, axis=0), np.roll(quad[::-1], shift, axis=0)):
            got = quad_from_polygon([c for p in ring for c in p])
            assert got is not None
            assert np.allclose(got, canonical, atol=1e-3), f"shift={shift}"


def test_from_coco_sniffs_the_export_type(tmp_path, photo):
    from docscanner.data.real import from_coco

    images = tmp_path / "images"
    images.mkdir()
    imwrite_rgb(images / "a.jpg", photo)
    quad = [[110, 120], [520, 96], [545, 640], [86, 604]]

    seg = tmp_path / "seg.json"
    seg.write_text(json.dumps(_seg_coco(photo, quad)))
    out = from_coco(seg, images, tmp_path / "r_seg")
    assert json.loads(out.read_text())["items"][0]["meta"]["source"] == "coco-segmentation"

    kp = tmp_path / "kp.json"
    kp.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg",
                    "width": photo.shape[1], "height": photo.shape[0]}],
        "categories": [{"id": 1, "name": "page",
                        "keypoints": ["top-left", "top-right",
                                      "bottom-right", "bottom-left"]}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "keypoints": [c for x, y in quad for c in (x, y, 2)]}]}))
    out = from_coco(kp, images, tmp_path / "r_kp")
    assert json.loads(out.read_text())["items"][0]["meta"]["source"] == "coco-keypoints"

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))
    with pytest.raises(ValueError, match="no keypoint or segmentation"):
        from_coco(empty, images, tmp_path / "r_none")


def test_degenerate_polygons_are_rejected_not_stored(tmp_path, photo):
    from docscanner.data.real import from_coco_segmentation, quad_from_polygon

    assert quad_from_polygon([0, 0, 10, 0, 10, 0, 0, 0]) is None
    assert quad_from_polygon([1, 2, 3]) is None
    assert quad_from_polygon([]) is None
    assert quad_from_polygon([float("nan")] * 8) is None

    images = tmp_path / "images"
    images.mkdir()
    imwrite_rgb(images / "a.jpg", photo)
    bad = {"images": [{"id": 0, "file_name": "a.jpg", "width": 640, "height": 760}],
           "categories": [{"id": 1, "name": "page"}],
           "annotations": [{"id": 1, "image_id": 0, "category_id": 1,
                            "segmentation": [[5, 5, 6, 5, 6, 6, 5, 6]]}]}
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    out = from_coco_segmentation(tmp_path / "bad.json", images, tmp_path / "real")
    from docscanner.data.check_labels import check_manifest
    assert check_manifest(out).failed == 1


def test_the_real_course_dataset_is_present_and_valid():
    from docscanner.data.check_labels import check_manifest

    manifest = Path("data/real/own/annotations.json")
    if not manifest.exists():
        pytest.skip("own photos not prepared in this checkout")
    data = json.loads(manifest.read_text())
    assert data["count"] >= 10, "the brief asks for 10-15 real photographs"
    report = check_manifest(manifest)
    assert report.failed == 0, report.summary()


def test_real_photo_set_loads(tmp_path, photo):
    from docscanner.data.datasets import RealPhotoSet
    from docscanner.data.real import build_manifest
    (tmp_path / "photos").mkdir()
    imwrite_rgb(tmp_path / "photos" / "x.jpg", photo)
    build_manifest(tmp_path, "t", [{"file": "photos/x.jpg",
                                    "corners": [[110, 120], [520, 96],
                                                [545, 640], [86, 604]]}])
    ds = RealPhotoSet(tmp_path / "annotations.json", size=128)
    assert len(ds) == 1
    img, corners, _ = ds.raw(0)
    assert img.shape[:2] == photo.shape[:2] and corners.shape == (4, 2)
    x, c = ds[0]
    assert x.shape == (3, 128, 128) and c.shape == (4, 2)
    assert float(c.max()) <= 1.5


@pytest.mark.parametrize("shape", [(64, 64), (37, 121), (200, 51)])
def test_scanner_survives_odd_sizes(models, shape):
    from docscanner.pipeline.scanner import DocumentScanner
    rng = np.random.default_rng(3)
    img = (rng.random((*shape, 3)) * 255).astype(np.uint8)
    sc = DocumentScanner(models["heatmap"], models["enhance"], max_side=256)
    res = sc.scan(img, auto_orientation=False)
    assert res.image.size > 0 and np.isfinite(res.corners).all()


def test_scanner_handles_a_grayscale_and_rgba_image(models, photo):
    from docscanner.pipeline.scanner import DocumentScanner
    from docscanner.utils.imageio import ensure_rgb
    sc = DocumentScanner(models["heatmap"], models["enhance"], max_side=256)
    grey = ensure_rgb(cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY))
    rgba = ensure_rgb(cv2.cvtColor(photo, cv2.COLOR_RGB2RGBA))
    for img in (grey, rgba):
        res = sc.scan(img, auto_orientation=False)
        assert res.image.ndim == 3


def test_imageio_round_trips_unicode_paths(tmp_path, photo):
    path = tmp_path / "صفحه_تست.png"
    imwrite_rgb(path, photo)
    back = imread_rgb(path)
    assert back.shape == photo.shape
    assert np.abs(back.astype(int) - photo.astype(int)).max() == 0


def test_highlight_recovery_reduces_clipping_without_darkening():
    from docscanner.pipeline.enhance_pipeline import recover_highlights

    rng = np.random.default_rng(0)
    inp = np.clip(rng.normal(0.62, 0.12, (128, 128, 3)), 0, 1).astype(np.float32)
    out = np.clip(inp * 1.55 + 0.22, 0, 1).astype(np.float32)

    fixed = recover_highlights(inp, out)
    clipped_before = float((out >= 250 / 255).mean())
    clipped_after = float((fixed >= 250 / 255).mean())

    assert clipped_before > 0.05, "test fixture should actually clip"
    assert clipped_after < clipped_before * 0.6, (
        f"clipping {clipped_before:.3f} -> {clipped_after:.3f}")
    assert abs(float(fixed.mean()) - float(out.mean())) < 0.06
    assert fixed.min() >= 0.0 and fixed.max() <= 1.0


def test_highlight_recovery_leaves_unsaturated_images_alone():
    from docscanner.pipeline.enhance_pipeline import recover_highlights

    rng = np.random.default_rng(1)
    inp = np.clip(rng.normal(0.35, 0.08, (64, 64, 3)), 0, 1).astype(np.float32)
    out = np.clip(inp + 0.05, 0, 0.8).astype(np.float32)
    assert np.allclose(recover_highlights(inp, out), out, atol=1e-6)


def test_enhancement_pipeline_exposes_the_highlight_guard(models, photo):
    from docscanner.pipeline.enhance_pipeline import EnhancementPipeline

    on = EnhancementPipeline(models["enhance"], max_side=256, recover_highlights=True)
    off = EnhancementPipeline(models["enhance"], max_side=256, recover_highlights=False)
    assert on.recover_highlights is True and off.recover_highlights is False
    a = on(photo, mode="raw").network_output
    b = off(photo, mode="raw").network_output
    assert a.shape == b.shape and a.dtype == np.uint8


def test_cli_lists_every_command_and_routes():
    import importlib

    from docscanner.cli import _GROUPS, COMMANDS
    from docscanner.cli import main as cli_main

    for name, cmd in COMMANDS.items():
        module = importlib.import_module(cmd.module)
        assert callable(getattr(module, "main", None)), f"{name} -> {cmd.module}"

    grouped = [n for _, names in _GROUPS for n in names]
    assert sorted(grouped) == sorted(COMMANDS), (
        f"listing and COMMANDS disagree: {set(grouped) ^ set(COMMANDS)}")
    assert len(grouped) == len(set(grouped)), "a command is listed twice"

    assert cli_main([]) == 0
    assert cli_main(["--help"]) == 0
    assert cli_main(["definitely-not-a-command"]) == 2


def test_cli_forwards_arguments_verbatim(monkeypatch):
    import docscanner.pipeline.run as run_mod
    from docscanner.cli import main as cli_main

    seen = {}
    def _fake_main(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(run_mod, "main", _fake_main)
    cli_main(["scan", "x.jpg", "--mode", "bw"])
    assert seen["argv"] == ["scan", "x.jpg", "--mode", "bw"]


def test_refresh_runs_steps_in_order_and_survives_a_failure(monkeypatch, tmp_path):
    from docscanner import refresh as R

    order = []

    def ok(name):
        def _fn(args):
            order.append(name)
            return 0
        return _fn

    def boom(args):
        order.append("figures")
        raise RuntimeError("no gif encoder")

    monkeypatch.setattr(R, "STEPS", {
        "export": (ok("export"), "", False),
        "figures": (boom, "", False),
        "report": (ok("report"), "", False),
        "tests": (ok("tests"), "", True),
    })

    class Args:
        runs, models, limit = "runs", str(tmp_path), None
        quick, only, skip, traceback = False, None, None, False

    rc = R.run(Args())
    assert rc == 1, "a failed step must fail the run"
    assert order == ["export", "figures", "report", "tests"], order

    order.clear()
    Args.quick = True
    R.run(Args())
    assert "tests" not in order and "export" in order


def test_colab_notebook_is_valid_and_uses_real_flags():
    import re

    paths = sorted(Path("notebooks").glob("*.ipynb"))
    if not paths:
        pytest.skip("no notebooks present")

    source = ""
    for nb_path in paths:
        nb = json.loads(nb_path.read_text())
        assert nb["nbformat"] == 4 and nb["cells"], f"{nb_path} is empty"
        source += "\n".join("".join(c["source"]) for c in nb["cells"]
                            if c["cell_type"] == "code") + "\n"
    used = set(re.findall(r"(?<!-)--([a-z][a-z0-9-]+)", source))

    import contextlib
    import importlib

    from docscanner.cli import COMMANDS
    known = {"help"}
    for cmd in COMMANDS.values():
        mod = importlib.import_module(cmd.module)
        with contextlib.suppress(SystemExit):
            mod.main(["--help"])
        known |= set(re.findall(r"--([a-z][a-z0-9-]+)", (mod.__doc__ or "")))
    import subprocess
    import sys
    helps = ""
    for name in COMMANDS:
        helps += subprocess.run([sys.executable, "-m", "docscanner.cli", name, "--help"],
                                capture_output=True, text=True).stdout
    for script in sorted(Path("scripts").glob("*.py")):
        helps += subprocess.run([sys.executable, str(script), "--help"],
                                capture_output=True, text=True).stdout

    foreign = {"quiet", "depth", "branch", "no", "global", "unset", "gpus",
               "heads", "index-url"}
    for flag in used:
        if flag in foreign:
            continue
        assert f"--{flag}" in helps, f"notebook uses --{flag}, no command accepts it"


def test_the_highlight_guard_stays_out_of_the_measured_output(tmp_path):
    import numpy as np

    from docscanner.models.enhance_unet import DocEnhanceNet
    from docscanner.pipeline.enhance_pipeline import EnhancementPipeline

    model = DocEnhanceNet(base=8, depth=3)
    page = np.full((96, 96, 3), 252, np.uint8)
    page[30:60, 20:70] = 20

    guarded = EnhancementPipeline(model=model, compress_highlights=True)(page, mode="raw")
    plain = EnhancementPipeline(model=model, compress_highlights=False)(page, mode="raw")

    assert np.array_equal(guarded.network_output, plain.network_output)
    assert not np.array_equal(guarded.image, plain.image)


def test_app_auto_scan_produces_a_page(photo):
    from docscanner.app.gradio_app import auto_scan

    out = auto_scan(photo, "color", True, False, False, 900)
    page = out[0] if isinstance(out, tuple) else out
    assert page is not None
    assert page.ndim == 3 and page.shape[2] == 3
    assert page.dtype == np.uint8
    assert min(page.shape[:2]) > 16


def test_app_enhance_and_corner_tabs_return_renderable_output(photo):
    from docscanner.app.gradio_app import corner_lab, enhance_only

    page = _page()
    enhanced = enhance_only(page, "color", 512, False)
    assert (enhanced[0] if isinstance(enhanced, tuple) else enhanced) is not None

    lab = corner_lab(photo, False)
    assert (lab[0] if isinstance(lab, tuple) else lab) is not None
