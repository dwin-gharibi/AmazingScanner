from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from ..data.check_labels import check_manifest
from ..data.labeling import CORNER_NAMES, LabelSession, add_or_move_point, quad_feedback
from ..utils.geometry import order_corners
from ..utils.viz import PALETTE
from .theme import CSS, build_theme, hero, note, stat_card

DEFAULT_OUT = Path("data/real/own")

__all__ = ["build_labeler", "main", "LabelerState"]


def draw_annotation(img: np.ndarray, pts: list[list[float]],
                    active: int | None = None) -> np.ndarray:
    out = img.copy()
    n = len(pts)
    scale = max(out.shape[:2]) / 900.0
    r = max(5, int(9 * scale))
    thick = max(2, int(3 * scale))

    if n >= 4:
        poly = np.asarray(pts[:4], np.int32).reshape(-1, 1, 2)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], PALETTE["pred"])
        out = cv2.addWeighted(overlay, 0.16, out, 0.84, 0)
        cv2.polylines(out, [poly], True, PALETTE["pred"], thick, cv2.LINE_AA)
    elif n >= 2:
        for a, b in itertools.pairwise(pts):
            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     PALETTE["pred"], thick, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts[:4]):
        x, y = int(round(x)), int(round(y))
        colour = PALETTE["accent"] if i == active else PALETTE["gt"]
        cv2.circle(out, (x, y), r + thick, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), r, colour, -1, cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (x + r + 4, y - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * scale, (255, 255, 255),
                    max(3, int(4 * scale)), cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (x + r + 4, y - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * scale, colour,
                    max(1, int(2 * scale)), cv2.LINE_AA)
    return out


def magnifier(img: np.ndarray, point: list[float] | None, half: int = 46,
              zoom: int = 6) -> np.ndarray | None:
    if point is None:
        return None
    h, w = img.shape[:2]
    cx, cy = int(round(point[0])), int(round(point[1]))
    x0, y0 = cx - half, cy - half
    patch = cv2.copyMakeBorder(img, half, half, half, half,
                               cv2.BORDER_CONSTANT, value=(24, 24, 28))
    patch = patch[y0 + half:y0 + half + 2 * half, x0 + half:x0 + half + 2 * half]
    if patch.size == 0:
        return None
    big = cv2.resize(patch, (2 * half * zoom, 2 * half * zoom),
                     interpolation=cv2.INTER_NEAREST)
    c = half * zoom
    cv2.line(big, (c, 0), (c, big.shape[0]), PALETTE["accent"], 1, cv2.LINE_AA)
    cv2.line(big, (0, c), (big.shape[1], c), PALETTE["accent"], 1, cv2.LINE_AA)
    cv2.circle(big, (c, c), zoom * 2, PALETTE["accent"], 1, cv2.LINE_AA)
    return big


class LabelerState:
    def __init__(self, out_dir: str | Path = DEFAULT_OUT, name: str = "own-photos"):
        self.session = LabelSession.load(out_dir, name=name)
        self._pipe = None
        self._pipe_tried = False

    def detector(self):
        if not self._pipe_tried:
            self._pipe_tried = True
            try:
                from .gradio_app import REG
                if REG.heatmap_ckpt or REG.regression_ckpt:
                    self._pipe = REG.corners()
            except Exception:
                self._pipe = None
        return self._pipe

    def suggest(self, img: np.ndarray) -> tuple[list[list[float]], str]:
        from ..pipeline.corner_pipeline import detect_quad_classical

        pipe = self.detector()
        if pipe is not None:
            try:
                res = pipe(img)
                if res.source == "network":
                    return res.corners.tolist(), "detector (CNN + edge refinement)"
                return res.corners.tolist(), f"detector fell back to {res.source}"
            except Exception as exc:
                return [], f"detector failed: {exc}"
        quad = detect_quad_classical(img)
        if quad is None:
            return [], "no page found - click the corners yourself"
        return quad.tolist(), "classical detector (no trained model on disk)"

    def rectify(self, img: np.ndarray, pts) -> np.ndarray | None:
        from ..pipeline.scanner import rectify_with_corners

        pts = list(pts or [])
        if len(pts) < 4:
            return None
        quad = order_corners(np.asarray(pts[:4], np.float32))
        page, _ = rectify_with_corners(img, quad, max_side=1400)
        return page

    def scan_with(self, img: np.ndarray, pts, mode: str = "color"):
        from ..pipeline.enhance_pipeline import apply_output_mode

        page = self.rectify(img, pts)
        if page is None:
            return None, "Place all four corners first."
        try:
            from .gradio_app import REG
            if REG.enhance_ckpt is None:
                return apply_output_mode(page, mode), (
                    "No enhancement checkpoint on disk yet - showing the "
                    "rectified page only.")
            out = REG.enhancer(page, mode="raw").network_output
            return apply_output_mode(out, mode), "Rectified and enhanced."
        except Exception as exc:
            return apply_output_mode(page, mode), f"Enhancement unavailable: {exc}"

    def evaluate_detector(self) -> str:
        from ..utils.geometry import quad_iou

        sess = self.session
        labelled = [f for f in sess.files if f in sess.corners]
        if not labelled:
            return "Nothing labelled yet."
        pipe = self.detector()
        if pipe is None:
            return ("No trained corner checkpoint on disk yet - train one "
                    "(`make train`) and this will score it against your labels.")

        errs, ious, sources = [], [], []
        for rel in labelled:
            img = sess.image(rel)
            if img is None:
                continue
            gt = order_corners(np.asarray(sess.corners[rel], np.float32))
            res = pipe(img)
            errs.append(float(np.linalg.norm(res.corners - gt, axis=1).mean()))
            ious.append(quad_iou(res.corners, gt))
            sources.append(res.source)
        if not errs:
            return "Could not read any labelled photo."
        e = np.asarray(errs)
        diag = 1.0
        lines = [
            f"Detector vs your labels, on {len(e)} photograph(s):",
            "",
            f"  mean corner error   {e.mean():8.2f} px",
            f"  median              {np.median(e):8.2f} px",
            f"  worst               {e.max():8.2f} px",
            f"  quad IoU            {np.mean(ious):8.4f}",
            f"  within 8 px         {100 * (e < 8).mean():7.1f} %",
            f"  within 16 px        {100 * (e < 16).mean():7.1f} %",
            "",
            f"  detection path      {({s: sources.count(s) for s in set(sources)})}",
        ]
        del diag
        return "\n".join(lines)


def snap_to_edges(img: np.ndarray, pts: list[list[float]]) -> tuple[list[list[float]], str]:
    from ..pipeline.corner_pipeline import refine_quad_with_edges

    if len(pts) < 4:
        return pts, "Place all four corners first."
    quad = order_corners(np.asarray(pts[:4], np.float32))
    refined, ok = refine_quad_with_edges(img, quad)
    if not ok:
        return quad.tolist(), ("No straight border found near those corners - "
                               "they were left as clicked.")
    shift = float(np.abs(refined - quad).max())
    return refined.tolist(), f"Snapped to the page border (moved up to {shift:.1f} px)."


def _status_html(state: LabelerState, msg: str, valid: bool | None = None) -> str:
    s = state.session.stats()
    tone = "good" if s["remaining"] == 0 and s["total"] else (
        "warn" if s["labelled"] else "")
    cards = [
        stat_card("Photos", str(s["total"]), "in this set"),
        stat_card("Labelled", str(s["labelled"]), "saved to manifest", tone),
        stat_card("Remaining", str(s["remaining"]), "still to do",
                  "" if s["remaining"] == 0 else "warn"),
    ]
    if valid is not None:
        cards.append(stat_card("This photo", "valid" if valid else "check it",
                               msg[:48], "good" if valid else "bad"))
    return f'<div class="ds-stats">{"".join(cards)}</div>'


def build_labeler(out_dir: str | Path = DEFAULT_OUT,
                  name: str = "own-photos") -> gr.Blocks:
    state = LabelerState(out_dir, name)

    def _render(msg: str = "", active: int | None = None):
        sess = state.session
        img = sess.image()
        if img is None:
            return (None, None, _status_html(state, "no photos yet"),
                    "Upload photographs to begin.", [], "-")
        pts = sess.get() or []
        valid, fb = quad_feedback(pts, img.shape[:2])
        label = (f"{sess.index + 1} / {len(sess)} - {Path(sess.current or '').name}")
        return (draw_annotation(img, pts, active),
                magnifier(img, pts[active] if active is not None and pts else None),
                _status_html(state, fb, valid if pts else None),
                msg or fb, pts, label)

    def on_upload(files):
        if not files:
            return _render("Nothing uploaded.")
        paths = [f if isinstance(f, str) else getattr(f, "name", None) for f in files]
        added = state.session.add_images([p for p in paths if p])
        state.session.save()
        if added:
            state.session.goto(state.session.files.index(added[0]))
        return _render(f"Added {len(added)} photo(s). "
                       f"Click the {CORNER_NAMES[0]} corner, or press Suggest.")

    def on_capture(frame):
        if frame is None:
            return _render("Take a photo with the camera first.")
        import tempfile

        from ..utils.imageio import imwrite_rgb
        tmp = Path(tempfile.mkdtemp()) / "capture.jpg"
        imwrite_rgb(tmp, np.asarray(frame), quality=95)
        added = state.session.add_images([tmp])
        state.session.save()
        if added:
            state.session.goto(state.session.files.index(added[0]))
        return _render("Captured. Press Suggest, or click the corners.")

    def on_click(pts, evt: gr.SelectData):
        img = state.session.image()
        if img is None:
            return _render("Upload photographs first.")
        idx = evt.index
        if not isinstance(idx, (list, tuple)) or len(idx) < 2:
            return _render("Could not read the click position.")
        h, w = img.shape[:2]
        x = float(np.clip(idx[0], 0, w - 1))
        y = float(np.clip(idx[1], 0, h - 1))
        new_pts, touched, moved = add_or_move_point(list(pts or []), x, y)
        state.session.set(new_pts, save=len(new_pts) >= 4)
        if len(new_pts) < 4:
            state.session.corners.pop(state.session.current or "", None)
        verb = "Moved corner" if moved else "Placed corner"
        out = _render(f"{verb} {touched + 1} ({CORNER_NAMES[touched]}).", touched)
        return (out[0], out[1], out[2], out[3], new_pts, out[5])

    def on_suggest(pts):
        img = state.session.image()
        if img is None:
            return _render("Upload photographs first.")
        quad, how = state.suggest(img)
        if not quad:
            return _render(how)
        state.session.set(quad)
        out = _render(f"Suggested by {how} - drag any corner that is off.", 0)
        return (out[0], out[1], out[2], out[3], quad, out[5])

    def on_snap(pts):
        img = state.session.image()
        if img is None:
            return _render("Upload photographs first.")
        new_pts, msg = snap_to_edges(img, list(pts or []))
        if len(new_pts) >= 4:
            state.session.set(new_pts)
        out = _render(msg, 0 if new_pts else None)
        return (out[0], out[1], out[2], out[3], new_pts, out[5])

    def on_clear():
        state.session.clear()
        out = _render("Cleared. Click the top-left corner.")
        return (out[0], out[1], out[2], out[3], [], out[5])

    def on_nav(delta: int, only_unlabelled: bool = False):
        def _go(_pts):
            if not len(state.session):
                return _render("Upload photographs first.")
            if only_unlabelled:
                state.session.next_unlabelled()
            else:
                state.session.step(delta)
            out = _render()
            return (out[0], out[1], out[2], out[3], out[4], out[5])
        return _go

    def on_delete():
        state.session.remove_current()
        out = _render("Photo removed from the set.")
        return (out[0], out[1], out[2], out[3], out[4] or [], out[5])

    def on_preview(pts):
        img = state.session.image()
        if img is None:
            return None, "Upload or capture a photograph first."
        page = state.rectify(img, pts)
        if page is None:
            return None, "Place all four corners first."
        h, w = page.shape[:2]
        return page, (f"Flattened with your corners: {w}x{h} px, "
                      f"{max(w, h) / max(min(w, h), 1):.2f}:1. "
                      "If this looks skewed, a corner is off.")

    def on_scan(pts, mode):
        img = state.session.image()
        if img is None:
            return None, "Upload or capture a photograph first."
        page, msg = state.scan_with(img, pts, mode)
        return page, msg

    def on_validate():
        path = state.session.save()
        if not state.session.stats()["labelled"]:
            return "Nothing labelled yet.", None
        sheet = Path(state.session.out_dir) / "label_review.jpg"
        report = check_manifest(path, contact_sheet=sheet)
        return report.summary(), (str(sheet) if sheet.exists() else None)

    def on_score():
        state.session.save()
        return state.evaluate_detector()

    def on_export():
        manifest = state.session.save()
        coco = state.session.export_coco(
            Path(state.session.out_dir) / "coco_keypoints.json")
        return (f"Manifest: {manifest}\nCOCO keypoints: {coco}\n\n"
                f"{state.session.stats()}"), str(manifest), str(coco)

    with gr.Blocks(title="AmazingScanner - Label Studio",
                   analytics_enabled=False) as demo:
        gr.HTML(hero(
            "🏷️ AmazingScanner Label Studio",
            "Capture or upload your photographs, mark the four page corners, "
            "check the flattened result, and export the dataset - the whole "
            "Section 1.2 workflow in one place.",
            ["camera capture", "auto-suggest", "sub-pixel edge snap",
             "live validation", "autosave + resume", "COCO keypoints export",
             "scores the detector on your own pages"]))

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab("Upload"):
                    up = gr.File(label="Upload photographs", file_count="multiple",
                                 file_types=["image"])
                with gr.Tab("Camera"):
                    cam = gr.Image(label="Take a photo", sources=["webcam", "clipboard"],
                                   type="numpy", height=240)
                    b_capture = gr.Button("📷 Add this capture to the set",
                                          variant="primary")
                gr.HTML(note(
                    f"Order matters: <b>1 {CORNER_NAMES[0]} → 2 {CORNER_NAMES[1]} → "
                    f"3 {CORNER_NAMES[2]} → 4 {CORNER_NAMES[3]}</b>, going "
                    "clockwise around the page <i>as it reads</i>. Click near a "
                    "corner to move it; you never have to start over."))
                with gr.Row():
                    b_suggest = gr.Button("✨ Suggest corners", variant="primary")
                    b_snap = gr.Button("🧲 Snap to edges")
                with gr.Row():
                    b_prev = gr.Button("← Prev")
                    b_next = gr.Button("Next →")
                    b_todo = gr.Button("Next unlabelled")
                with gr.Row():
                    b_clear = gr.Button("Clear corners")
                    b_del = gr.Button("Remove photo")
                zoom = gr.Image(label="Magnifier", height=250)
                with gr.Accordion("Finish up", open=True):
                    b_check = gr.Button("✅ Validate all labels")
                    b_score = gr.Button("📈 Score the detector on my photos")
                    b_export = gr.Button("⬇ Export manifest + COCO")
                    out_report = gr.Textbox(label="Report", lines=10)
                    out_sheet = gr.Image(label="Contact sheet", height=240)
                    f_manifest = gr.File(label="annotations.json")
                    f_coco = gr.File(label="coco_keypoints.json")
            with gr.Column(scale=5):
                caption = gr.Markdown("-")
                stats = gr.HTML()
                with gr.Tab("Annotate"):
                    canvas = gr.Image(label="Click the corners", height=620,
                                      interactive=True)
                with gr.Tab("Check the label"):
                    gr.HTML(note("The real test of an annotation: flatten the "
                                 "page with the corners as labelled. A quad that "
                                 "is one corner off looks fine as an outline and "
                                 "obviously wrong here."))
                    b_preview = gr.Button("Flatten with my corners",
                                          variant="primary")
                    prev_img = gr.Image(label="Rectified page", height=520)
                    prev_msg = gr.Markdown("-")
                with gr.Tab("Scan it"):
                    gr.HTML(note("The end-to-end chain on this photograph, using "
                                 "<b>your</b> corners instead of the detector's: "
                                 "rectify, then enhance."))
                    with gr.Row():
                        scan_mode = gr.Dropdown(
                            ["color", "gray", "bw", "whiteboard", "raw"],
                            value="color", label="Output style", scale=2)
                        b_scanit = gr.Button("Scan this photo", variant="primary",
                                             scale=1)
                    scan_img = gr.Image(label="Finished page", height=520)
                    scan_msg = gr.Markdown("-")
                msg = gr.Markdown("Upload or capture photographs to begin.")

        pts_state = gr.State([])
        outs = [canvas, zoom, stats, msg, pts_state, caption]

        up.upload(on_upload, [up], outs)
        b_capture.click(on_capture, [cam], outs)
        canvas.select(on_click, [pts_state], outs)
        b_suggest.click(on_suggest, [pts_state], outs)
        b_snap.click(on_snap, [pts_state], outs)
        b_clear.click(on_clear, None, outs)
        b_prev.click(on_nav(-1), [pts_state], outs)
        b_next.click(on_nav(+1), [pts_state], outs)
        b_todo.click(on_nav(0, only_unlabelled=True), [pts_state], outs)
        b_del.click(on_delete, None, outs)
        b_preview.click(on_preview, [pts_state], [prev_img, prev_msg])
        b_scanit.click(on_scan, [pts_state, scan_mode], [scan_img, scan_msg])
        b_check.click(on_validate, None, [out_report, out_sheet])
        b_score.click(on_score, None, [out_report])
        b_export.click(on_export, None, [out_report, f_manifest, f_coco])
        demo.load(lambda: _render(), None, outs)

    return demo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Label page corners for your own photos")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="dataset directory (default: data/real/own)")
    ap.add_argument("--name", default="own-photos")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args(argv)

    demo = build_labeler(args.out, args.name)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                theme=build_theme(), css=CSS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
