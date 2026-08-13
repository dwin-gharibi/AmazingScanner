from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from docscanner.data.labeling import (
    CORNER_NAMES,
    LabelSession,
    add_or_move_point,
    quad_feedback,
)
from docscanner.utils.geometry import order_corners
from docscanner.utils.imageio import imwrite_rgb

QUAD = [[110.0, 120.0], [520.0, 96.0], [545.0, 640.0], [86.0, 604.0]]


@pytest.fixture
def photos(tmp_path):
    rng = np.random.default_rng(7)
    out = tmp_path / "src"
    out.mkdir()
    paths = []
    for i in range(3):
        desk = np.full((760, 640, 3), 58, np.uint8)
        desk = (desk + rng.normal(0, 7, desk.shape)).clip(0, 255).astype(np.uint8)
        page = np.full((500, 380, 3), 236, np.uint8)
        for k in range(12):
            y = 40 + k * 34
            cv2.line(page, (34, y), (346, y), (68, 68, 76), 2)
        src = np.float32([[0, 0], [379, 0], [379, 499], [0, 499]])
        h = cv2.getPerspectiveTransform(src, np.float32(QUAD))
        warp = cv2.warpPerspective(page, h, (640, 760))
        mask = cv2.warpPerspective(np.full((500, 380), 255, np.uint8), h, (640, 760))
        photo = np.where(mask[..., None] > 0, warp, desk).astype(np.uint8)
        p = out / f"photo_{i}.jpg"
        imwrite_rgb(p, photo, quality=95)
        paths.append(p)
    return paths


def test_clicks_append_until_four():
    pts = []
    for i, (x, y) in enumerate(QUAD):
        pts, idx, moved = add_or_move_point(pts, x, y)
        assert idx == i and not moved
    assert len(pts) == 4


def test_a_second_click_on_top_of_the_first_corrects_it():
    pts, _, _ = add_or_move_point([], 300.0, 300.0)
    pts, idx, moved = add_or_move_point(pts, 310.0, 305.0, snap_px=24.0)
    assert moved and idx == 0 and len(pts) == 1


def test_click_near_an_existing_corner_moves_it_instead_of_adding():
    pts = [[100.0, 100.0], [300.0, 100.0]]
    pts, idx, moved = add_or_move_point(pts, 105.0, 103.0, snap_px=24.0)
    assert moved and idx == 0
    assert len(pts) == 2 and pts[0] == [105.0, 103.0]


def test_a_fifth_click_moves_the_nearest_corner_never_resets():
    pts = [list(p) for p in QUAD]
    pts, idx, moved = add_or_move_point(pts, 530.0, 100.0)
    assert moved and idx == 1 and len(pts) == 4
    assert pts[1] == [530.0, 100.0]
    pts, idx, moved = add_or_move_point(pts, 5.0, 5.0)
    assert moved and len(pts) == 4


def test_feedback_guides_then_validates():
    ok, msg = quad_feedback([], (760, 640))
    assert not ok and CORNER_NAMES[0] in msg
    ok, msg = quad_feedback(QUAD[:2], (760, 640))
    assert not ok and CORNER_NAMES[2] in msg
    ok, msg = quad_feedback(QUAD, (760, 640))
    assert ok, msg


def test_feedback_rejects_a_bowtie_and_a_speck():
    bowtie = [[100, 100], [400, 100], [100, 500], [400, 500]]
    ok, msg = quad_feedback(bowtie, (760, 640))
    assert not ok and "convex" in msg
    speck = [[10, 10], [30, 10], [30, 30], [10, 30]]
    ok, msg = quad_feedback(speck, (760, 640))
    assert not ok and "%" in msg


def test_session_add_navigate_annotate_and_persist(tmp_path, photos):
    sess = LabelSession(out_dir=tmp_path / "ds")
    added = sess.add_images(photos)
    assert len(added) == 3 and len(sess) == 3
    assert sess.stats() == {"total": 3, "labelled": 0, "remaining": 3}

    sess.set(QUAD)
    assert sess.stats()["labelled"] == 1
    stored = np.asarray(sess.get(), np.float32)
    assert np.allclose(stored, order_corners(np.asarray(QUAD, np.float32)), atol=1e-3)

    sess.step(1)
    assert sess.get() is None
    sess.set(QUAD)
    sess.goto(2)
    sess.set(QUAD)
    assert sess.stats()["remaining"] == 0

    data = json.loads(sess.manifest_path.read_text())
    assert data["count"] == 3 and len(data["items"]) == 3
    assert all("corners" in it for it in data["items"])


def test_partial_session_writes_a_valid_manifest_and_resumes(tmp_path, photos):
    root = tmp_path / "ds"
    sess = LabelSession(out_dir=root)
    sess.add_images(photos)
    sess.set(QUAD)
    sess.save()

    data = json.loads((root / "annotations.json").read_text())
    assert data["count"] == 1
    assert len(data["items"]) == 1
    assert len(data["pending"]) == 2

    again = LabelSession.load(root)
    assert len(again) == 3
    assert again.stats()["labelled"] == 1
    assert np.allclose(np.asarray(again.corners[again.files[0]], np.float32),
                       order_corners(np.asarray(QUAD, np.float32)), atol=1e-2)


def test_next_unlabelled_skips_finished_photos(tmp_path, photos):
    sess = LabelSession(out_dir=tmp_path / "ds")
    sess.add_images(photos)
    sess.goto(0)
    sess.set(QUAD)
    sess.goto(1)
    sess.set(QUAD)
    sess.goto(0)
    assert sess.next_unlabelled() == sess.files[2]


def test_duplicate_filenames_do_not_overwrite_each_other(tmp_path, photos):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    import shutil
    shutil.copyfile(photos[0], a / "IMG_0001.jpg")
    shutil.copyfile(photos[1], b / "IMG_0001.jpg")
    sess = LabelSession(out_dir=tmp_path / "ds")
    sess.add_images([a / "IMG_0001.jpg", b / "IMG_0001.jpg"])
    assert len(sess) == 2
    assert len(set(sess.files)) == 2
    for rel in sess.files:
        assert (sess.out_dir / rel).exists()


def test_remove_photo(tmp_path, photos):
    sess = LabelSession(out_dir=tmp_path / "ds")
    sess.add_images(photos)
    sess.remove_current(delete_file=True)
    assert len(sess) == 2


def test_manifest_round_trips_through_the_project_loader(tmp_path, photos):
    from docscanner.data.datasets import RealPhotoSet

    sess = LabelSession(out_dir=tmp_path / "ds")
    sess.add_images(photos)
    for _ in range(3):
        sess.set(QUAD)
        sess.step(1)

    ds = RealPhotoSet(sess.manifest_path)
    assert len(ds) == 3
    img, corners = ds.raw(0)[:2]
    assert img.ndim == 3 and corners.shape == (4, 2)
    assert np.allclose(corners, order_corners(np.asarray(QUAD, np.float32)), atol=1e-2)


def test_check_labels_passes_on_what_the_tool_writes(tmp_path, photos):
    from docscanner.data.check_labels import check_manifest

    sess = LabelSession(out_dir=tmp_path / "ds")
    sess.add_images(photos)
    for _ in range(3):
        sess.set(QUAD)
        sess.step(1)

    report = check_manifest(sess.manifest_path,
                            contact_sheet=tmp_path / "sheet.jpg")
    assert report.total == 3 and report.ok == 3
    assert not report.problems, report.summary()
    assert (tmp_path / "sheet.jpg").exists()


def test_coco_export_is_reimportable(tmp_path, photos):
    from docscanner.data.real import from_coco_keypoints

    sess = LabelSession(out_dir=tmp_path / "ds")
    sess.add_images(photos)
    for _ in range(3):
        sess.set(QUAD)
        sess.step(1)
    coco_path = sess.export_coco(tmp_path / "coco.json")

    coco = json.loads(coco_path.read_text())
    assert len(coco["images"]) == 3 and len(coco["annotations"]) == 3
    assert coco["categories"][0]["keypoints"] == list(CORNER_NAMES)
    assert coco["annotations"][0]["num_keypoints"] == 4

    back = from_coco_keypoints(coco_path, sess.out_dir / "photos",
                               tmp_path / "reimport", name="round-trip")
    data = json.loads(back.read_text())
    assert data["count"] == 3
    got = np.asarray(data["items"][0]["corners"], np.float32)
    assert np.allclose(got, order_corners(np.asarray(QUAD, np.float32)), atol=1e-2)


def test_suggest_and_snap_produce_a_sane_quad(tmp_path, photos):
    from docscanner.app.labeler import LabelerState, snap_to_edges
    from docscanner.utils.geometry import quad_is_plausible
    from docscanner.utils.imageio import imread_rgb

    state = LabelerState(tmp_path / "ds")
    state.session.add_images(photos)
    img = state.session.image()

    quad, how = state.suggest(img)
    assert len(quad) == 4, how
    assert quad_is_plausible(np.asarray(quad, np.float32), img.shape[:2])
    err = np.linalg.norm(order_corners(np.asarray(quad, np.float32))
                         - order_corners(np.asarray(QUAD, np.float32)), axis=1).mean()
    assert err < 25, f"suggestion is {err:.1f} px from the true page"

    rough = (np.asarray(QUAD, np.float32) + np.float32([[4, -3], [-5, 4], [3, 3], [-4, -5]]))
    snapped, msg = snap_to_edges(imread_rgb(state.session.out_dir / state.session.current),
                                 rough.tolist())
    assert len(snapped) == 4
    err_snap = np.linalg.norm(order_corners(np.asarray(snapped, np.float32))
                              - order_corners(np.asarray(QUAD, np.float32)), axis=1).mean()
    assert err_snap <= 6.0, f"{msg} -> {err_snap:.1f} px"


def test_capture_rectify_scan_and_score_close_the_loop(tmp_path, photos):
    from docscanner.app.labeler import LabelerState
    from docscanner.utils.imageio import imread_rgb

    state = LabelerState(tmp_path / "ds")

    frame = imread_rgb(photos[0])
    cap = tmp_path / "capture.jpg"
    imwrite_rgb(cap, frame, quality=95)
    assert state.session.add_images([cap]) and len(state.session) == 1

    img = state.session.image()
    quad, _ = state.suggest(img)
    state.session.set(quad)

    page = state.rectify(img, quad)
    assert page is not None and page.ndim == 3
    ph, pw = page.shape[:2]
    assert 1.1 < ph / pw < 1.6, f"rectified aspect {ph / pw:.2f} is not page-like"

    scanned, msg = state.scan_with(img, quad, mode="color")
    assert scanned is not None and scanned.dtype == np.uint8, msg

    report = state.evaluate_detector()
    assert "mean corner error" in report or "No trained corner checkpoint" in report


def test_rectify_and_scan_decline_before_four_corners(tmp_path, photos):
    from docscanner.app.labeler import LabelerState

    state = LabelerState(tmp_path / "ds")
    state.session.add_images(photos[:1])
    img = state.session.image()
    assert state.rectify(img, [[1, 2], [3, 4]]) is None
    out, msg = state.scan_with(img, [], mode="color")
    assert out is None and "four corners" in msg


def test_score_reports_when_nothing_is_labelled(tmp_path, photos):
    from docscanner.app.labeler import LabelerState

    state = LabelerState(tmp_path / "ds")
    state.session.add_images(photos)
    assert "Nothing labelled" in state.evaluate_detector()


def test_snap_refuses_before_four_corners(photos):
    from docscanner.app.labeler import snap_to_edges
    from docscanner.utils.imageio import imread_rgb

    pts, msg = snap_to_edges(imread_rgb(photos[0]), [[1, 2], [3, 4]])
    assert len(pts) == 2 and "four" in msg


def test_drawing_helpers_handle_every_stage(photos):
    from docscanner.app.labeler import draw_annotation, magnifier
    from docscanner.utils.imageio import imread_rgb

    img = imread_rgb(photos[0])
    for n in range(5):
        out = draw_annotation(img, QUAD[:n], active=max(0, n - 1) if n else None)
        assert out.shape == img.shape and out.dtype == np.uint8
    assert magnifier(img, None) is None
    for pt in ([0, 0], [639, 759], QUAD[0]):
        z = magnifier(img, list(map(float, pt)))
        assert z is not None and z.ndim == 3


def test_labeler_app_builds(tmp_path):
    from docscanner.app.labeler import build_labeler

    demo = build_labeler(tmp_path / "ds")
    assert demo is not None


def test_every_callback_is_wired_to_the_inputs_it_declares(tmp_path):
    import inspect

    from docscanner.app.labeler import build_labeler

    demo = build_labeler(tmp_path / "ds")
    checked = 0
    for fn in demo.fns.values():
        call = getattr(fn, "fn", None)
        if call is None or not callable(call):
            continue
        name = getattr(call, "__name__", "")
        if not name.startswith(("on_", "_go")):
            continue
        sig = inspect.signature(call)
        required = []
        for p in sig.parameters.values():
            if p.default is not inspect.Parameter.empty:
                continue
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                continue
            if "SelectData" in str(p.annotation):
                continue
            required.append(p)
        assert len(fn.inputs) >= len(required), (
            f"{name} needs {len(required)} input(s) but is bound to {len(fn.inputs)}")
        checked += 1
    assert checked >= 6, "expected to inspect the labeler's callbacks"


def test_photo_key_is_stable_whether_given_a_filename_or_a_stem():
    from docscanner.data.real import photo_key

    roboflow = "img4_main_jpg.rf.43906e9cf57ab8beca4aff58bf00cfec"
    assert photo_key(roboflow) == "img4"
    assert photo_key(roboflow + ".jpg") == "img4"
    assert photo_key("img4_scanned.jpg") == "img4"
    assert photo_key("img4_scanned") == "img4"
    assert photo_key(roboflow + ".jpg") == photo_key("img4_scanned.jpg")
    assert photo_key("photos/img13_main.jpg") == photo_key("reference/img13_scanned.jpg")


def test_ocr_default_language_covers_the_documents_we_evaluate():
    from docscanner.eval.ocr import DEFAULT_OCR_LANG

    langs = set(DEFAULT_OCR_LANG.split("+"))
    assert "eng" in langs
    assert "fas" in langs, (
        "the 24 real photographs are Persian pages — reading them with the "
        "English model alone measures the script, not the enhancement")
