from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, Callable

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


ProgressCallback = Optional[Callable[..., None]]


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    doc_path: str
    page: int
    kind: str
    text: str
    table: Optional[Any] = None
    bbox: Optional[Any] = None
    page_w: Optional[float] = None
    page_h: Optional[float] = None
    anchor: Optional[str] = None
    para_start: Optional[int] = None
    para_end: Optional[int] = None
    table_index: Optional[int] = None


def _emit_docx_progress(
    *,
    progress_cb: ProgressCallback,
    doc_name: str,
    current: int,
    total: int,
    message: str,
    progress_base: int,
    progress_span: int,
    stage: str = "chunking",
):
    if not progress_cb:
        return

    ratio = current / total if total > 0 else 1.0
    progress = int(progress_base + progress_span * ratio)
    progress = max(0, min(100, progress))

    progress_cb(
        stage=stage,
        status="running",
        progress=progress,
        message=message,
        current_document=doc_name,
        current_index=current,
        total_items=total,
        extraction_mode="docx",
    )


def _make_chunk_id(doc_name: str, part: str, idx: int, extra: str = "") -> str:
    raw = f"{doc_name}|{part}|{idx}|{extra}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:16]


def _iter_block_items(doc: Document) -> List[Union[Paragraph, Table]]:
    parent = doc.element.body
    out: List[Union[Paragraph, Table]] = []
    for child in parent.iterchildren():
        if child.tag.endswith("}p"):
            out.append(Paragraph(child, doc))
        elif child.tag.endswith("}tbl"):
            out.append(Table(child, doc))
    return out


def _is_heading(p: Paragraph) -> bool:
    style = (p.style.name or "") if p.style else ""
    return style.lower().startswith("heading")


def _heading_level(p: Paragraph) -> int:
    style = (p.style.name or "") if p.style else ""
    s = style.lower().strip()
    if not s.startswith("heading"):
        return 0
    tail = s.replace("heading", "").strip()
    try:
        return int(tail)
    except Exception:
        return 1


def _stringify_table(rows: List[List[str]]) -> str:
    lines: List[str] = []
    for r in rows:
        vals = [(x.strip() if x else "-") for x in r]
        lines.append(" | ".join(vals))
    return "\n".join(lines).strip()


def _table_to_rows(tbl: Table) -> List[List[str]]:
    rows: List[List[str]] = []
    for r in tbl.rows:
        row_cells: List[str] = []
        for c in r.cells:
            row_cells.append((c.text or "").strip())
        rows.append(row_cells)
    return rows


def _merge_paragraphs_to_chunks(
    paras: List[Tuple[int, str]],
    *,
    max_chunk_chars: int,
    min_paragraph_chars: int,
) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    buf: List[str] = []
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    cur_len = 0

    def flush() -> None:
        nonlocal buf, start_idx, end_idx, cur_len
        if not buf or start_idx is None or end_idx is None:
            buf, start_idx, end_idx, cur_len = [], None, None, 0
            return
        text = "\n\n".join(buf).strip()
        if text:
            out.append((text, start_idx, end_idx))
        buf, start_idx, end_idx, cur_len = [], None, None, 0

    for pi, ptxt in paras:
        t = (ptxt or "").strip()
        if not t:
            continue

        add_len = len(t) + (2 if buf else 0)
        if buf and (cur_len + add_len > max_chunk_chars):
            flush()

        if start_idx is None:
            start_idx = pi
        end_idx = pi

        buf.append(t)
        cur_len += add_len

    flush()

    return [
        (text, ps, pe)
        for (text, ps, pe) in out
        if len(text) >= min_paragraph_chars or len(text.split()) >= 6
    ]


def extract_chunks_from_docx(
    doc_path: Path,
    doc_name: str,
    *,
    min_paragraph_chars: int,
    max_chunk_chars: int,
    progress_cb: ProgressCallback = None,
    progress_base: int = 8,
    progress_span: int = 24,
) -> List[Chunk]:
    d = Document(str(doc_path))
    blocks = _iter_block_items(d)

    total_blocks = len(blocks)

    para_counter = 0
    table_counter = 0
    current_headings: List[Tuple[int, str]] = []
    collected_paras: List[Tuple[int, str]] = []
    chunks: List[Chunk] = []

    def heading_prefix() -> str:
        if not current_headings:
            return ""
        current_headings.sort(key=lambda x: x[0])
        return " > ".join([h for _, h in current_headings if h]) + "\n\n"

    def flush_paras() -> None:
        nonlocal collected_paras, chunks
        if not collected_paras:
            return

        merged = _merge_paragraphs_to_chunks(
            collected_paras,
            max_chunk_chars=max_chunk_chars,
            min_paragraph_chars=min_paragraph_chars,
        )

        for (t, ps, pe) in merged:
            text = (heading_prefix() + t).strip()
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(
                        doc_name,
                        "docx_text",
                        len(chunks) + 1,
                        extra=f"{ps}-{pe}",
                    ),
                    doc_name=doc_name,
                    doc_path=str(doc_path),
                    page=1,
                    kind="docx_text",
                    text=text,
                    anchor=f"p-{ps}",
                    para_start=ps,
                    para_end=pe,
                )
            )

        collected_paras = []

    for i, b in enumerate(blocks, start=1):

        _emit_docx_progress(
            progress_cb=progress_cb,
            doc_name=doc_name,
            current=i,
            total=total_blocks,
            message=f"Processing DOCX block {i}/{total_blocks}",
            progress_base=progress_base,
            progress_span=progress_span,
        )

        if isinstance(b, Paragraph):
            txt = (b.text or "").strip()
            if not txt:
                continue

            para_counter += 1

            if _is_heading(b):
                flush_paras()
                lvl = _heading_level(b)
                current_headings = [(l, h) for (l, h) in current_headings if l < lvl]
                current_headings.append((lvl, txt))
                continue

            collected_paras.append((para_counter, txt))
            continue

        if isinstance(b, Table):
            flush_paras()
            table_counter += 1

            _emit_docx_progress(
                progress_cb=progress_cb,
                doc_name=doc_name,
                current=i,
                total=total_blocks,
                message=f"Processing table #{table_counter}",
                progress_base=progress_base,
                progress_span=progress_span,
            )

            rows = _table_to_rows(b)
            table_text = _stringify_table(rows)
            if not table_text:
                continue

            text = table_text
            if current_headings:
                text = (heading_prefix() + table_text).strip()

            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc_name, "docx_table", table_counter),
                    doc_name=doc_name,
                    doc_path=str(doc_path),
                    page=1,
                    kind="docx_table",
                    text=text,
                    table=rows,
                    anchor=f"t-{table_counter}",
                    table_index=table_counter,
                )
            )

    flush_paras()

    return chunks

