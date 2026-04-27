from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import re
import time
import fitz
import pandas as pd
import pdfplumber

from .config import CONFIG


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    doc_path: str
    page: int
    kind: str
    text: str
    table: List[List[str]] | None
    bbox: Tuple[float, float, float, float] | None
    page_w: float | None = None
    page_h: float | None = None
    paragraph_bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)


ProgressCallback = Optional[Callable[..., None]]


def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _poly_to_bbox(poly) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _union_bbox(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float] | None:
    if not bboxes:
        return None
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (float(x0), float(y0), float(x1), float(y1))


def _dedupe_boxes(
    boxes: List[Tuple[float, float, float, float]],
    eps: float = 0.5,
) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    for b in boxes:
        try:
            x0, y0, x1, y1 = [float(v) for v in b]
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        found = False
        for e in out:
            if (
                abs(x0 - e[0]) <= eps
                and abs(y0 - e[1]) <= eps
                and abs(x1 - e[2]) <= eps
                and abs(y1 - e[3]) <= eps
            ):
                found = True
                break
        if not found:
            out.append((x0, y0, x1, y1))
    return out


def _limit_boxes(
    boxes: List[Tuple[float, float, float, float]],
    max_boxes: int = 12,
) -> List[Tuple[float, float, float, float]]:
    boxes = _dedupe_boxes(boxes)
    boxes = sorted(boxes, key=lambda r: (r[1], r[0], r[3], r[2]))
    return boxes[:max_boxes]


def _sort_lines_reading_order(lines: List[Dict], y_tol: float = 6.0) -> List[Dict]:
    lines = sorted(lines, key=lambda r: (r["y0"], r["x0"]))
    out: List[Dict] = []
    bucket: List[Dict] = []
    cur_y = None

    def flush() -> None:
        nonlocal bucket
        if not bucket:
            return
        bucket = sorted(bucket, key=lambda r: r["x0"])
        out.extend(bucket)
        bucket = []

    for r in lines:
        if cur_y is None:
            cur_y = r["y0"]
            bucket.append(r)
            continue
        if abs(r["y0"] - cur_y) <= y_tol:
            bucket.append(r)
        else:
            flush()
            cur_y = r["y0"]
            bucket.append(r)
    flush()
    return out


_PADDLE_OCR = None
_PADDLE_OCR_CFG = None


def _get_paddle_ocr(lang: str):
    global _PADDLE_OCR, _PADDLE_OCR_CFG
    cfg = (lang or "en", False, True)
    if _PADDLE_OCR is not None and _PADDLE_OCR_CFG == cfg:
        return _PADDLE_OCR

    from paddleocr import PaddleOCR

    kwargs = dict(
        lang=lang,
        use_angle_cls=True,
        use_gpu=False,
        det_limit_side_len=1600,
    )

    try:
        kwargs["show_log"] = False
        ocr = PaddleOCR(**kwargs)
    except TypeError:
        kwargs.pop("show_log", None)
        ocr = PaddleOCR(**kwargs)

    _PADDLE_OCR = ocr
    _PADDLE_OCR_CFG = cfg
    return ocr


def _render_page_to_pil(doc: fitz.Document, page_index0: int, dpi: int):
    page = doc.load_page(page_index0)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    from PIL import Image

    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _downscale_pil(img, max_side: int = 1600):
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h))


def _render_worker(pdf_path_str: str, page_index0: int, dpi: int, max_side: int) -> tuple[int, bytes]:
    from io import BytesIO
    from PIL import Image  # noqa: F401

    doc = fitz.open(pdf_path_str)
    try:
        img = _render_page_to_pil(doc, page_index0, dpi=dpi)
        img = _downscale_pil(img, max_side=max_side)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return (page_index0 + 1, buf.getvalue())
    finally:
        doc.close()


def _paddle_ocr_page_to_lines(page_image_pil, lang: str, zoom: float) -> List[Dict]:
    import numpy as np

    ocr = _get_paddle_ocr(lang)
    rgb = np.asarray(page_image_pil)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return []
    bgr = rgb[:, :, ::-1].copy()

    res = ocr.ocr(bgr, cls=True)
    if not res:
        return []

    items = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res
    lines: List[Dict] = []

    for item in items:
        try:
            poly = item[0]
            txt = (item[1][0] or "").strip()
            if not txt:
                continue
            x0, y0, x1, y1 = _poly_to_bbox(poly)
            lines.append(
                {
                    "text": txt,
                    "x0": float(x0) / zoom,
                    "y0": float(y0) / zoom,
                    "x1": float(x1) / zoom,
                    "y1": float(y1) / zoom,
                }
            )
        except Exception:
            continue

    return _sort_lines_reading_order(lines)


def _adaptive_gap_threshold(items: List[Dict], default: float = 10.0) -> float:
    if not items:
        return default
    heights = []
    for it in items:
        try:
            h = float(it["y1"]) - float(it["y0"])
            if h > 0:
                heights.append(h)
        except Exception:
            pass
    if not heights:
        return default
    heights.sort()
    mid = heights[len(heights) // 2]
    return max(default, float(mid) * 1.6)


def _lines_to_chunk_parts(
    lines: List[Dict],
    max_chunk_chars: int,
    min_chars: int,
    gap_threshold: float | None = None,
) -> List[Dict]:
    if not lines:
        return []

    if gap_threshold is None:
        gap_threshold = _adaptive_gap_threshold(lines, default=10.0)

    paragraphs: List[Dict] = []
    cur_text: List[str] = []
    cur_boxes: List[Tuple[float, float, float, float]] = []
    prev_y1: float | None = None

    for ln in lines:
        if prev_y1 is not None and (ln["y0"] - prev_y1) > float(gap_threshold):
            txt = _clean_text(" ".join(cur_text))
            if txt:
                pbbox = _union_bbox(cur_boxes)
                paragraphs.append({"text": txt, "bbox": pbbox})
            cur_text = []
            cur_boxes = []

        cur_text.append(ln["text"])
        cur_boxes.append((ln["x0"], ln["y0"], ln["x1"], ln["y1"]))
        prev_y1 = ln["y1"]

    txt = _clean_text(" ".join(cur_text))
    if txt:
        pbbox = _union_bbox(cur_boxes)
        paragraphs.append({"text": txt, "bbox": pbbox})

    merged: List[Dict] = []
    buf_txt = ""
    buf_boxes: List[Tuple[float, float, float, float]] = []
    buf_para_boxes: List[Tuple[float, float, float, float]] = []

    for p in paragraphs:
        t = _clean_text(p.get("text") or "")
        if not t:
            continue

        pbbox = p.get("bbox")

        if not buf_txt:
            buf_txt = t
            if pbbox:
                buf_boxes.append(pbbox)
                buf_para_boxes.append(pbbox)
            continue

        if len(buf_txt) + 2 + len(t) <= max_chunk_chars:
            buf_txt = buf_txt + "\n\n" + t
            if pbbox:
                buf_boxes.append(pbbox)
                buf_para_boxes.append(pbbox)
        else:
            out_txt = _clean_text(buf_txt)
            if len(out_txt) >= min_chars:
                merged.append(
                    {
                        "text": out_txt,
                        "bbox": _union_bbox(buf_boxes),
                        "paragraph_bboxes": _limit_boxes(buf_para_boxes),
                    }
                )
            buf_txt = t
            buf_boxes = [pbbox] if pbbox else []
            buf_para_boxes = [pbbox] if pbbox else []

    out_txt = _clean_text(buf_txt)
    if out_txt and len(out_txt) >= min_chars:
        merged.append(
            {
                "text": out_txt,
                "bbox": _union_bbox(buf_boxes),
                "paragraph_bboxes": _limit_boxes(buf_para_boxes),
            }
        )

    return merged


def _native_page_to_parts(
    page: pdfplumber.page.Page,
    max_chunk_chars: int,
    min_chars: int,
) -> List[Dict]:
    try:
        words = page.extract_words(
            keep_blank_chars=False,
            use_text_flow=True,
        ) or []
    except Exception:
        words = []

    items: List[Dict] = []
    for w in words:
        txt = (w.get("text") or "").strip()
        if not txt:
            continue
        x0 = float(w.get("x0", 0.0))
        x1 = float(w.get("x1", 0.0))
        y0 = float(w.get("top", 0.0))
        y1 = float(w.get("bottom", 0.0))
        items.append({"text": txt, "x0": x0, "y0": y0, "x1": x1, "y1": y1})

    if not items:
        return []

    items = _sort_lines_reading_order(items, y_tol=3.5)
    return _lines_to_chunk_parts(
        lines=items,
        max_chunk_chars=max_chunk_chars,
        min_chars=min_chars,
        gap_threshold=None,
    )


def _should_ocr_page(page_text: str) -> bool:
    txt = _clean_text(page_text or "")
    min_chars = int(getattr(CONFIG, "OCR_MIN_CHARS", 120))
    if len(txt) < min_chars:
        return True

    alnum = sum(ch.isalnum() for ch in txt)
    ratio = (alnum / max(1, len(txt)))
    min_ratio = float(getattr(CONFIG, "OCR_MIN_ALNUM_RATIO", 0.45))
    if ratio < min_ratio:
        return True

    return False


def _emit_pdf_progress(
    *,
    progress_cb: ProgressCallback,
    doc_name: str,
    current_page: int,
    total_pages: int,
    mode: str,
    message: str,
    progress_base: int,
    progress_span: int,
) -> None:
    if not progress_cb:
        return

    ratio = (current_page / total_pages) if total_pages > 0 else 1.0
    progress = int(progress_base + (progress_span * ratio))
    progress = max(0, min(100, progress))

    progress_cb(
        stage="chunking",
        status="running",
        progress=progress,
        message=message,
        current_document=doc_name,
        current_page=current_page,
        total_pages=total_pages,
        extraction_mode=mode,
    )


def extract_chunks_from_pdf(
    pdf_path: Path,
    doc_name: str,
    min_paragraph_chars: int,
    max_chunk_chars: int,
    ocr_always: bool = False,
    ocr_dpi: int = 96,
    ocr_lang: str = "id",
    progress_cb: ProgressCallback = None,
    progress_base: int = 8,
    progress_span: int = 24,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    doc_t0 = time.time()

    render_workers = max(1, int(getattr(CONFIG, "OCR_RENDER_WORKERS", 4)))
    prefetch_pages = max(1, int(getattr(CONFIG, "OCR_PREFETCH_PAGES", 8)))
    max_side = int(getattr(CONFIG, "OCR_MAX_SIDE", 1600))
    zoom = ocr_dpi / 72.0

    mode = "OCR_ALL" if ocr_always else "HYBRID"
    print(
        f"[PaddleOCR] START DOC {doc_name} | MODE={mode} | render_workers={render_workers} prefetch={prefetch_pages} dpi={ocr_dpi} max_side={max_side} lang={ocr_lang}",
        flush=True,
    )

    from io import BytesIO
    from PIL import Image

    def process_tables(page, page_no: int, page_w: float, page_h: float) -> None:
        try:
            tables = page.find_tables() or []
        except Exception:
            tables = []

        for k, t in enumerate(tables, start=1):
            try:
                tbl = t.extract() or []
            except Exception:
                continue

            norm = [[(c if c is not None else "") for c in row] for row in tbl]
            df = pd.DataFrame(norm)
            table_text = _clean_text(df.to_csv(index=False))

            if table_text:
                tbbox = tuple(t.bbox) if getattr(t, "bbox", None) else None
                para_boxes = [tbbox] if tbbox else []
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_name}::p{page_no}::tb{k}",
                        doc_name=doc_name,
                        doc_path=str(pdf_path),
                        page=page_no,
                        kind="table",
                        text=table_text,
                        table=norm,
                        bbox=tbbox,
                        page_w=page_w,
                        page_h=page_h,
                        paragraph_bboxes=para_boxes,
                    )
                )

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)

        ex: ProcessPoolExecutor | None = None
        in_flight = set()
        fut_to_page: Dict[object, int] = {}
        rendered_png: Dict[int, bytes] = {}

        def ensure_executor() -> ProcessPoolExecutor:
            nonlocal ex
            if ex is None:
                ex = ProcessPoolExecutor(max_workers=render_workers)
            return ex

        def submit_one(pno: int) -> None:
            ex2 = ensure_executor()
            fut = ex2.submit(_render_worker, str(pdf_path), pno - 1, ocr_dpi, max_side)
            in_flight.add(fut)
            fut_to_page[fut] = pno

        def prefetch_from(pno: int) -> None:
            if prefetch_pages <= 0:
                return
            for j in range(pno, min(total_pages, pno + prefetch_pages - 1) + 1):
                if j in rendered_png:
                    continue
                if any(v == j for v in fut_to_page.values()):
                    continue
                submit_one(j)

        def ensure_rendered(pno: int) -> bytes:
            if pno in rendered_png:
                return rendered_png.pop(pno)

            submit_one(pno)
            while True:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    in_flight.remove(fut)
                    pno_done = fut_to_page.pop(fut, None)
                    if pno_done is None:
                        continue
                    pno2, png_bytes = fut.result()
                    if pno2 == pno:
                        return png_bytes
                    rendered_png[pno2] = png_bytes

        try:
            for i, page in enumerate(pdf.pages, start=1):
                page_w = float(page.width)
                page_h = float(page.height)

                _emit_pdf_progress(
                    progress_cb=progress_cb,
                    doc_name=doc_name,
                    current_page=i,
                    total_pages=total_pages,
                    mode="preparing_page",
                    message=f"Preparing page {i}/{total_pages}",
                    progress_base=progress_base,
                    progress_span=progress_span,
                )

                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""

                page_text = _clean_text(page_text)
                do_ocr = bool(ocr_always) or _should_ocr_page(page_text)

                if not do_ocr:
                    _emit_pdf_progress(
                        progress_cb=progress_cb,
                        doc_name=doc_name,
                        current_page=i,
                        total_pages=total_pages,
                        mode="native",
                        message=f"Reading native text on page {i}/{total_pages}",
                        progress_base=progress_base,
                        progress_span=progress_span,
                    )

                    parts = _native_page_to_parts(
                        page=page,
                        max_chunk_chars=max_chunk_chars,
                        min_chars=min_paragraph_chars,
                    )

                    t_idx = 0
                    for p in parts:
                        text = _clean_text(p.get("text") or "")
                        if not text or len(text) < min_paragraph_chars:
                            continue
                        t_idx += 1
                        bbox = p.get("bbox") or (0.0, 0.0, page_w, page_h)
                        para_boxes = p.get("paragraph_bboxes") or ([bbox] if bbox else [])
                        chunks.append(
                            Chunk(
                                chunk_id=f"{doc_name}::p{i}::t{t_idx}",
                                doc_name=doc_name,
                                doc_path=str(pdf_path),
                                page=i,
                                kind="text",
                                text=text,
                                table=None,
                                bbox=bbox,
                                page_w=page_w,
                                page_h=page_h,
                                paragraph_bboxes=para_boxes,
                            )
                        )
                else:
                    _emit_pdf_progress(
                        progress_cb=progress_cb,
                        doc_name=doc_name,
                        current_page=i,
                        total_pages=total_pages,
                        mode="ocr",
                        message=f"Running OCR on page {i}/{total_pages}",
                        progress_base=progress_base,
                        progress_span=progress_span,
                    )

                    ocr_t0 = time.time()
                    prefetch_from(i)

                    png_bytes = ensure_rendered(i)
                    page_img = Image.open(BytesIO(png_bytes)).convert("RGB")

                    lines = _paddle_ocr_page_to_lines(page_img, lang=ocr_lang, zoom=zoom)
                    parts = _lines_to_chunk_parts(
                        lines=lines,
                        max_chunk_chars=max_chunk_chars,
                        min_chars=20,
                        gap_threshold=None,
                    )

                    total_chars = sum(len(p.get("text") or "") for p in parts)
                    print(
                        f"[PaddleOCR] OCR p{i}/{total_pages} done | chunks={len(parts)} chars={total_chars} | {time.time() - ocr_t0:.2f}s",
                        flush=True,
                    )

                    t_idx = 0
                    for p in parts:
                        text = _clean_text(p.get("text") or "")
                        if not text or len(text) < 20:
                            continue
                        t_idx += 1
                        bbox = p.get("bbox") or (0.0, 0.0, page_w, page_h)
                        para_boxes = p.get("paragraph_bboxes") or ([bbox] if bbox else [])
                        chunks.append(
                            Chunk(
                                chunk_id=f"{doc_name}::p{i}::t{t_idx}",
                                doc_name=doc_name,
                                doc_path=str(pdf_path),
                                page=i,
                                kind="text",
                                text=text,
                                table=None,
                                bbox=bbox,
                                page_w=page_w,
                                page_h=page_h,
                                paragraph_bboxes=para_boxes,
                            )
                        )

                process_tables(page, i, page_w, page_h)

                _emit_pdf_progress(
                    progress_cb=progress_cb,
                    doc_name=doc_name,
                    current_page=i,
                    total_pages=total_pages,
                    mode="page_done",
                    message=f"Finished page {i}/{total_pages}",
                    progress_base=progress_base,
                    progress_span=progress_span,
                )

        finally:
            if ex is not None:
                ex.shutdown(wait=True, cancel_futures=False)

    print(
        f"[PaddleOCR] FINISH DOC {doc_name} | pages={total_pages} | total_time={time.time() - doc_t0:.2f}s",
        flush=True,
    )
    return chunks

