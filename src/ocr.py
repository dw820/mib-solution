"""Stage 1 - PDF -> an OCR record the extractor understands.

Renders each page with pypdfium2 (~200 DPI) and reads it with RapidOCR (PP-OCR
models via ONNX; CPU, offline, bundled in the wheel - no downloads at runtime).

Orientation is GATED: OCR the page as-rendered once; only if it does not look
upright (few lines / tall boxes / low confidence) do we try the other three
90-degree rotations and keep whichever reads best. That is ~3x cheaper than
always trying four and is the behaviour the pipeline was validated against.

The record shape matches what the pipeline's `common.doc_lines` consumes:
    {case_id, scale, pages: [
        {page, k, w, h,
         lines: [{t, s, x0, y0, x1, y1}],   # TRUSTED visible OCR
         hidden_text}                        # UNTRUSTED PDF text layer (injection
    ]}                                        # surface); kept separate, never a value source
"""
from __future__ import annotations

import os

import numpy as np
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR

RENDER_SCALE = 2                 # pypdfium scale=2 ~ 200 DPI (matches the trained pipeline)

# One engine per process. When the run is parallelized across vCPUs (solution.py),
# each worker caps intra-op threads to 1 (via MIB_OCR_THREADS) so N workers don't
# oversubscribe the 4 vCPUs; 0 leaves onnxruntime's default (single-process runs).
_THREADS = int(os.environ.get("MIB_OCR_THREADS", "0"))
_OCR = RapidOCR(**({"intra_op_num_threads": _THREADS} if _THREADS > 0 else {}))


def _run(img: np.ndarray):
    res, _ = _OCR(img)           # rendered pages are RGB; RapidOCR handles colour internally
    return res or []             # list of [box(4 pts), text, score]


def _total_conf(res) -> float:
    return float(sum(s for _, _, s in res))


def _looks_upright(res) -> bool:
    """Upright text lines are many, wide-and-short, and read confidently."""
    if len(res) < 5:
        return False
    aspects, confs = [], []
    for box, _t, s in res:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        w = max(xs) - min(xs); h = max(ys) - min(ys)
        aspects.append(w / max(h, 1e-6)); confs.append(s)
    return float(np.median(aspects)) >= 1.5 and float(np.mean(confs)) >= 0.75


def _orient(rgb: np.ndarray):
    """Return (k_turns_ccw, res) for the best orientation; pays for rotations only when needed."""
    res = _run(rgb)
    if _looks_upright(res):
        return 0, res
    best_k, best_res, best_s = 0, res, _total_conf(res)
    for k in (1, 2, 3):
        rot = np.ascontiguousarray(np.rot90(rgb, k))
        r = _run(rot)
        s = _total_conf(r)
        if s > best_s:
            best_k, best_res, best_s = k, r, s
    return best_k, best_res


def _pack_lines(res):
    out = []
    for box, text, score in res:
        xs = [float(p[0]) for p in box]; ys = [float(p[1]) for p in box]
        out.append({"t": text, "s": round(float(score), 4),
                    "x0": round(min(xs), 1), "y0": round(min(ys), 1),
                    "x1": round(max(xs), 1), "y1": round(max(ys), 1)})
    return out


def _hidden_text(doc, i) -> str:
    """PDF text layer for page i. UNTRUSTED - kept only so downstream code can see
    where it DISAGREES with the visible pixels; never used as a value source."""
    try:
        return doc[i].get_textpage().get_text_range() or ""
    except Exception:
        return ""


def ocr_document(pdf_path) -> dict:
    """OCR one PDF into a pipeline record. Raises on unreadable PDFs (the caller skips them)."""
    doc = pdfium.PdfDocument(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        rgb = np.array(doc[i].render(scale=RENDER_SCALE).to_pil())
        k, res = _orient(rgb)
        pages.append({"page": i, "k": k,
                      "w": int(rgb.shape[1]), "h": int(rgb.shape[0]),
                      "lines": _pack_lines(res),
                      "hidden_text": _hidden_text(doc, i)})
    return {"case_id": str(getattr(pdf_path, "stem", pdf_path)), "scale": RENDER_SCALE, "pages": pages}
