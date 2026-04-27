from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import HTTPException

from src.config import CONFIG
from src.utils import ensure_dirs


_ALLOWED_EXTS = {".pdf", ".docx"}


def ensure_app_dirs() -> None:
    ensure_dirs(CONFIG.PDF_DIR, CONFIG.STORAGE_DIR, CONFIG.UPLOAD_DIR)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = Path(name).suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"File type tidak didukung: {ext}. Hanya: {sorted(_ALLOWED_EXTS)}",
        )

    return name


def _resolve_doc_path(doc_name: str) -> Path:
    ensure_app_dirs()

    safe_name = Path(doc_name).name
    p = CONFIG.PDF_DIR / safe_name

    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if p.suffix.lower() not in _ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    return p


def save_uploaded_doc(filename: str, content: bytes) -> Path:
    ensure_app_dirs()
    safe = _safe_filename(filename)
    out_path = CONFIG.PDF_DIR / safe
    out_path.write_bytes(content)
    return out_path


def list_docs() -> List[Path]:
    ensure_app_dirs()

    out: List[Path] = []
    for p in CONFIG.PDF_DIR.glob("**/*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in _ALLOWED_EXTS:
            out.append(p)

    return sorted(out)


def get_doc_path(doc_name: str) -> Path:
    return _resolve_doc_path(doc_name)


def save_uploaded_pdf(filename: str, content: bytes) -> Path:
    return save_uploaded_doc(filename, content)


def list_pdfs() -> List[Path]:
    return list_docs()


def get_pdf_path(doc_name: str) -> Path:
    return get_doc_path(doc_name)

