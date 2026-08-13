from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

__all__ = ["tesseract_available", "easyocr_available", "ocr_backends",
           "resolve_backend", "ocr_available", "OCRResult", "run_ocr", "cer", "wer",
           "normalize_text", "readability_report", "detect_orientation",
           "searchable_pdf", "extract_text", "available_languages", "DEFAULT_OCR_LANG",
           "language_choices", "PAGE_MODES", "ENGINE_MODES", "LANGUAGE_NAMES"]


PAGE_MODES: dict[str, int] = {
    "Automatic (page with columns)": 3,
    "Single uniform block of text": 6,
    "Single text line": 7,
    "Single word": 8,
    "Sparse text, any order": 11,
    "Sparse text with orientation detection": 12,
    "Raw line (no layout analysis)": 13,
}

ENGINE_MODES: dict[str, int] = {
    "Neural (LSTM)": 3,
    "Legacy + neural": 2,
    "Legacy only": 0,
}

LANGUAGE_NAMES: dict[str, str] = {
    "eng": "English", "fas": "Persian / Farsi", "ara": "Arabic",
    "deu": "German", "fra": "French", "spa": "Spanish", "rus": "Russian",
    "chi_sim": "Chinese (simplified)", "chi_tra": "Chinese (traditional)",
    "jpn": "Japanese", "kor": "Korean", "hin": "Hindi", "ita": "Italian",
    "por": "Portuguese", "nld": "Dutch", "tur": "Turkish", "pol": "Polish",
    "ukr": "Ukrainian", "heb": "Hebrew", "urd": "Urdu", "osd": "orientation only",
}


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def easyocr_available() -> bool:
    try:
        return True
    except Exception:
        return False


def ocr_backends() -> tuple[str, ...]:
    out = []
    if easyocr_available():
        out.append("easyocr")
    if tesseract_available():
        out.append("tesseract")
    return tuple(out)


def resolve_backend(backend: str = "auto") -> str | None:
    available = ocr_backends()
    if backend and backend != "auto":
        return backend if backend in available else None
    return available[0] if available else None


def ocr_available(backend: str = "auto") -> bool:
    return resolve_backend(backend) is not None


_EASYOCR_LANG = {
    "eng": "en", "fas": "fa", "ara": "ar", "deu": "de", "fra": "fr",
    "spa": "es", "rus": "ru", "ita": "it", "por": "pt", "nld": "nl",
    "tur": "tr", "pol": "pl", "ukr": "uk", "hin": "hi", "urd": "ur",
    "chi_sim": "ch_sim", "chi_tra": "ch_tra", "jpn": "ja", "kor": "ko",
}


@lru_cache(maxsize=4)
def _easyocr_reader(langs: tuple[str, ...]):
    import easyocr
    return easyocr.Reader(list(langs), gpu=_easyocr_gpu(), verbose=False)


def _easyocr_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _run_ocr_easyocr(image: np.ndarray, lang: str, min_word_conf: float,
                     with_boxes: bool) -> OCRResult:
    codes = tuple(dict.fromkeys(
        _EASYOCR_LANG[c] for c in str(lang).split("+") if c in _EASYOCR_LANG)) or ("en",)
    try:
        reader = _easyocr_reader(codes)
        img = _prepare_for_easyocr(image)
        raw = reader.readtext(img, detail=1, paragraph=False)
    except Exception as exc:
        return OCRResult(available=False, meta={"error": str(exc),
                                                "backend": "easyocr"})

    words, confs, boxes = [], [], []
    for item in raw:
        if len(item) < 3:
            continue
        box, txt, conf = item[0], item[1], float(item[2]) * 100.0
        t = (txt or "").strip()
        if not t:
            continue
        words.append(t)
        confs.append(conf)
        if with_boxes:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            boxes.append({"text": t, "conf": conf,
                          "box": [int(min(xs)), int(min(ys)),
                                  int(max(xs) - min(xs)), int(max(ys) - min(ys))]})

    accepted = [(t, c) for t, c in zip(words, confs) if c >= min_word_conf]
    text = " ".join(words)
    meta: dict[str, Any] = {
        "all_words": len(words),
        "mean_conf_all": float(np.mean(confs)) if confs else 0.0,
        "lang": "+".join(codes), "backend": "easyocr",
    }
    if with_boxes:
        meta["boxes"] = boxes
        meta["scale"] = _prepare_for_easyocr(image).shape[1] / max(
            np.asarray(image).shape[1], 1)
    return OCRResult(
        text=text,
        mean_confidence=float(np.mean([c for _, c in accepted])) if accepted else 0.0,
        words=len(accepted),
        chars=len(text.replace(" ", "")),
        meta=meta,
    )


@lru_cache(maxsize=1)
def available_languages() -> tuple[str, ...]:
    if not tesseract_available():
        return ()
    try:
        import pytesseract
        langs = [str(l) for l in pytesseract.get_languages(config="")]
    except Exception:
        return ("eng",)
    return tuple(sorted(l for l in langs if l and l != "osd"))


def language_choices() -> list[tuple[str, str]]:
    return [(f"{LANGUAGE_NAMES.get(c, c)} ({c})", c) for c in available_languages()]


@dataclass
class OCRResult:
    text: str = ""
    mean_confidence: float = 0.0
    words: int = 0
    chars: int = 0
    available: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img * 255 if img.max() <= 1.5 else img, 0, 255).astype(np.uint8)
    return img


def _upscale(img: np.ndarray, min_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) >= min_side:
        return img
    s = min_side / max(min(h, w), 1)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)


def _prepare_for_ocr(image: np.ndarray, min_side: int = 1000) -> np.ndarray:
    img = _to_uint8(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return _upscale(img, min_side)


def _prepare_for_easyocr(image: np.ndarray, min_side: int = 800) -> np.ndarray:
    img = _to_uint8(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return _upscale(img[..., :3], min_side)


DEFAULT_OCR_LANG = "eng+fas"


def run_ocr(image: np.ndarray, psm: int = 3, lang: str = DEFAULT_OCR_LANG,
            min_word_conf: float = 40.0, oem: int = 3,
            with_boxes: bool = False, backend: str = "auto") -> OCRResult:
    chosen = resolve_backend(backend)
    if chosen is None:
        return OCRResult(available=False)
    if chosen == "easyocr":
        return _run_ocr_easyocr(image, lang, min_word_conf, with_boxes)
    if not tesseract_available():
        return OCRResult(available=False)
    import pytesseract

    installed = available_languages()
    if installed:
        wanted = [x for x in str(lang).split("+") if x]
        keep = [x for x in wanted if x in installed]
        lang = "+".join(keep) if keep else (installed[0] if installed else "eng")

    img = _prepare_for_ocr(image)
    cfg = f"--oem {int(oem)} --psm {int(psm)}"
    try:
        data = pytesseract.image_to_data(img, lang=lang, config=cfg,
                                         output_type=pytesseract.Output.DICT)
    except Exception as exc:
        return OCRResult(available=False, meta={"error": str(exc)})

    words, confs = [], []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        t = (txt or "").strip()
        if not t or c < 0:
            continue
        words.append(t)
        confs.append(c)

    accepted = [(t, c) for t, c in zip(words, confs) if c >= min_word_conf]
    text = " ".join(words)
    meta: dict[str, Any] = {
        "all_words": len(words),
        "mean_conf_all": float(np.mean(confs)) if confs else 0.0,
        "lang": lang, "psm": int(psm), "oem": int(oem),
    }
    if with_boxes:
        boxes = []
        n = len(data.get("text", []))
        for i in range(n):
            t = (data["text"][i] or "").strip()
            try:
                c = float(data["conf"][i])
            except (TypeError, ValueError):
                continue
            if t and c >= 0:
                boxes.append({"text": t, "conf": c,
                              "box": [int(data["left"][i]), int(data["top"][i]),
                                      int(data["width"][i]), int(data["height"][i])]})
        meta["boxes"] = boxes
        meta["scale"] = img.shape[1] / max(np.asarray(image).shape[1], 1)
    return OCRResult(
        text=text,
        mean_confidence=float(np.mean([c for _, c in accepted])) if accepted else 0.0,
        words=len(accepted),
        chars=len(text.replace(" ", "")),
        meta=meta,
    )


def detect_orientation(image: np.ndarray) -> int:
    if not tesseract_available():
        return 0
    import pytesseract
    try:
        osd = pytesseract.image_to_osd(_prepare_for_ocr(image, 800))
        m = re.search(r"Rotate: (\d+)", osd)
        return int(m.group(1)) % 360 if m else 0
    except Exception:
        return 0


def normalize_text(text: str, keep_case: bool = False) -> str:
    t = text if keep_case else text.lower()
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _levenshtein(a: list | str, b: list | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    r = normalize_text(reference) if normalize else reference
    h = normalize_text(hypothesis) if normalize else hypothesis
    r = r.replace(" ", "")
    h = h.replace(" ", "")
    if not r:
        return float("nan")
    return _levenshtein(r, h) / len(r)


def wer(reference: str, hypothesis: str) -> float:
    r = normalize_text(reference).split()
    h = normalize_text(hypothesis).split()
    if not r:
        return float("nan")
    return _levenshtein(r, h) / len(r)


def readability_report(
    images: dict[str, np.ndarray],
    reference_text: str | None = None,
    reference_key: str | None = None,
    lang: str = DEFAULT_OCR_LANG,
    backend: str = "auto",
) -> dict[str, dict[str, float]]:
    results = {k: run_ocr(v, lang=lang, backend=backend) for k, v in images.items()}
    ref = reference_text
    if ref is None and reference_key and reference_key in results:
        ref = results[reference_key].text

    out: dict[str, dict[str, float]] = {}
    for k, r in results.items():
        row = {
            "confidence": r.mean_confidence,
            "words": float(r.words),
            "chars": float(r.chars),
        }
        if ref:
            row["cer"] = cer(ref, r.text)
            row["wer"] = wer(ref, r.text)
        out[k] = row
    return out


def extract_text(image: np.ndarray, lang: str = "eng", psm: int = 3,
                 oem: int = 3) -> str:
    return run_ocr(image, psm=psm, lang=lang, oem=oem).text


def searchable_pdf(images, path, lang: str = "eng") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages = list(images)
    if not pages:
        raise ValueError("searchable_pdf: no pages")

    if not tesseract_available():
        from ..pipeline.scanner import save_pdf
        return save_pdf(pages, out)

    import pytesseract
    from PIL import Image

    chunks = []
    for page in pages:
        arr = np.asarray(page)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        try:
            chunks.append(pytesseract.image_to_pdf_or_hocr(
                Image.fromarray(arr[..., :3]), extension="pdf", lang=lang))
        except Exception:
            chunks = []
            break

    if not chunks:
        from ..pipeline.scanner import save_pdf
        return save_pdf(pages, out)

    if len(chunks) == 1:
        out.write_bytes(chunks[0])
        return out
    try:
        import io

        from pypdf import PdfReader, PdfWriter
        writer = PdfWriter()
        for c in chunks:
            for pg in PdfReader(io.BytesIO(c)).pages:
                writer.add_page(pg)
        with out.open("wb") as fh:
            writer.write(fh)
        return out
    except Exception:
        out.write_bytes(chunks[0])
        return out
