from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence, Callable, Optional

import numpy as np
from tqdm import tqdm

from app.services.state import get_embedder
from .config import CONFIG
from .docx_utils import extract_chunks_from_docx
from .pdf_utils import extract_chunks_from_pdf
from .utils import append_jsonl, ensure_dirs, read_json, write_json
from .vector_store import VectorStore


ProgressCallback = Optional[Callable[..., None]]


def _doc_id(doc_path: Path) -> str:
    st = doc_path.stat()
    raw = f"{doc_path.resolve()}|{st.st_mtime}|{st.st_size}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _normalize(embs: np.ndarray) -> np.ndarray:
    if embs.size == 0:
        return embs.astype(np.float32)
    n = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return (embs / n).astype(np.float32)


def list_doc_files(doc_dir: Path) -> List[Path]:
    exts = {".pdf", ".docx"}
    return sorted(p for p in doc_dir.glob("**/*") if p.is_file() and p.suffix.lower() in exts)


def _emit_progress(
    progress_cb: ProgressCallback,
    **payload: Any,
) -> None:
    if progress_cb:
        progress_cb(**payload)


def _embed_texts(
    model: Any,
    texts: Sequence[str],
    progress_cb: ProgressCallback = None,
    progress_base: int = 35,
    progress_span: int = 40,
    current_document: Optional[str] = None,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    batch_size = int(CONFIG.EMBED_BATCH_SIZE)
    embeddings: List[np.ndarray] = []

    total_batches = max(1, (len(texts) + batch_size - 1) // batch_size)
    processed_texts = 0

    _emit_progress(
        progress_cb,
        stage="embedding",
        status="running",
        progress=progress_base,
        message=f"Starting embedding for {len(texts)} chunks",
        current_document=current_document,
        processed_batches=0,
        total_batches=total_batches,
        processed_chunks=0,
        total_chunks=len(texts),
    )

    for batch_idx, i in enumerate(tqdm(range(0, len(texts), batch_size), desc="Embedding"), start=1):
        batch = list(texts[i : i + batch_size])
        embs = model.encode(batch, batch_size=batch_size, show_progress_bar=False)
        embeddings.append(np.asarray(embs, dtype=np.float32))

        processed_texts += len(batch)
        ratio = batch_idx / total_batches
        progress = int(progress_base + (progress_span * ratio))
        progress = max(0, min(100, progress))

        _emit_progress(
            progress_cb,
            stage="embedding",
            status="running",
            progress=progress,
            message=f"Embedding batch {batch_idx}/{total_batches}",
            current_document=current_document,
            processed_batches=batch_idx,
            total_batches=total_batches,
            processed_chunks=processed_texts,
            total_chunks=len(texts),
        )

    return _normalize(np.vstack(embeddings))


def _extract_chunks_any(
    path: Path,
    doc_name: str,
    progress_cb: ProgressCallback = None,
    progress_base: int = 8,
    progress_span: int = 24,
) -> List[Any]:
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_chunks_from_pdf(
            pdf_path=path,
            doc_name=doc_name,
            min_paragraph_chars=CONFIG.MIN_PARAGRAPH_CHARS,
            max_chunk_chars=CONFIG.MAX_CHUNK_CHARS,
            ocr_always=False,
            ocr_dpi=CONFIG.OCR_DPI,
            ocr_lang=CONFIG.PADDLE_OCR_LANG,
            progress_cb=progress_cb,
            progress_base=progress_base,
            progress_span=progress_span,
        )

    if ext == ".docx":
        return extract_chunks_from_docx(
            doc_path=path,
            doc_name=doc_name,
            min_paragraph_chars=CONFIG.MIN_PARAGRAPH_CHARS,
            max_chunk_chars=CONFIG.MAX_CHUNK_CHARS,
            progress_cb=progress_cb,
            progress_base=progress_base,
            progress_span=progress_span,
        )

    return []


def _chunk_to_record(c: Any) -> Dict[str, Any]:
    return {
        "chunk_id": getattr(c, "chunk_id", None),
        "doc_name": getattr(c, "doc_name", None),
        "doc_path": getattr(c, "doc_path", None),
        "page": getattr(c, "page", None),
        "kind": getattr(c, "kind", None),
        "text": getattr(c, "text", None),
        "table": getattr(c, "table", None),
        "bbox": getattr(c, "bbox", None),
        "paragraph_bboxes": getattr(c, "paragraph_bboxes", None),
        "page_w": getattr(c, "page_w", None),
        "page_h": getattr(c, "page_h", None),
        "anchor": getattr(c, "anchor", None),
        "para_start": getattr(c, "para_start", None),
        "para_end": getattr(c, "para_end", None),
        "table_index": getattr(c, "table_index", None),
    }


def _rewrite_chunks_jsonl(chunks_path: Path, chunks: List[Dict[str, Any]]) -> None:
    if chunks_path.exists():
        chunks_path.unlink()
    for rec in chunks:
        append_jsonl(chunks_path, rec)


def ingest_docs(
    doc_paths: List[Path],
    progress_cb: ProgressCallback = None,
) -> Dict[str, Any]:
    ensure_dirs(CONFIG.STORAGE_DIR, CONFIG.PDF_DIR, CONFIG.UPLOAD_DIR)

    registry = read_json(CONFIG.DOC_REGISTRY_PATH, default={"docs": {}})
    docs: Dict[str, Any] = registry.get("docs", {}) or {}

    _emit_progress(
        progress_cb,
        stage="preparing",
        status="running",
        progress=2,
        message="Preparing index resources",
        processed_docs=0,
        total_docs=len(doc_paths),
    )

    embedder = get_embedder()
    embedder.max_seq_length = 512

    _emit_progress(
        progress_cb,
        stage="preparing",
        status="running",
        progress=5,
        message="Loading vector store",
        processed_docs=0,
        total_docs=len(doc_paths),
    )

    store = VectorStore(CONFIG.FAISS_INDEX_PATH, CONFIG.CHUNKS_PATH).load()

    new_records: List[Dict[str, Any]] = []
    new_texts: List[str] = []
    added_docs = 0
    processed_docs = 0

    for doc_idx, doc_path in enumerate(doc_paths, start=1):
        doc_name = doc_path.name
        did = _doc_id(doc_path)

        if did in docs:
            processed_docs += 1
            _emit_progress(
                progress_cb,
                stage="skipping",
                status="running",
                progress=8,
                message=f"Skipping already indexed document: {doc_name}",
                current_document=doc_name,
                processed_docs=processed_docs,
                total_docs=len(doc_paths),
            )
            continue

        _emit_progress(
            progress_cb,
            stage="chunking",
            status="running",
            progress=8,
            message=f"Extracting content from {doc_name}",
            current_document=doc_name,
            processed_docs=processed_docs,
            total_docs=len(doc_paths),
        )

        chunks = _extract_chunks_any(
            doc_path,
            doc_name,
            progress_cb=progress_cb,
            progress_base=8,
            progress_span=24,
        )

        for c in chunks:
            rec = _chunk_to_record(c)
            new_records.append(rec)
            new_texts.append("passage: " + (rec.get("text") or ""))

        docs[did] = {
            "doc_name": doc_name,
            "doc_path": str(doc_path),
            "num_chunks": len(chunks),
        }
        added_docs += 1
        processed_docs += 1

        _emit_progress(
            progress_cb,
            stage="chunking",
            status="running",
            progress=32,
            message=f"Collected {len(chunks)} chunks from {doc_name}",
            current_document=doc_name,
            processed_docs=processed_docs,
            total_docs=len(doc_paths),
            processed_chunks=len(new_records),
            total_chunks=len(new_records),
        )

    if added_docs == 0:
        return {"added_docs": 0, "added_chunks": 0, "status": "no_new_docs"}

    _emit_progress(
        progress_cb,
        stage="embedding",
        status="running",
        progress=35,
        message=f"Generating embeddings for {len(new_texts)} chunks",
        processed_docs=processed_docs,
        total_docs=len(doc_paths),
        processed_chunks=0,
        total_chunks=len(new_texts),
    )

    embs = _embed_texts(
        embedder,
        new_texts,
        progress_cb=progress_cb,
        progress_base=35,
        progress_span=40,
        current_document=None,
    )

    _emit_progress(
        progress_cb,
        stage="writing_chunks",
        status="running",
        progress=76,
        message="Writing extracted chunks to storage",
        processed_docs=processed_docs,
        total_docs=len(doc_paths),
        processed_chunks=len(new_records),
        total_chunks=len(new_records),
    )

    for rec in new_records:
        append_jsonl(CONFIG.CHUNKS_PATH, rec)

    store.add(
        embs,
        new_records,
        progress_cb=progress_cb,
        progress_base=80,
        progress_span=15,
        current_document=None,
    )

    _emit_progress(
        progress_cb,
        stage="finalizing",
        status="running",
        progress=97,
        message="Updating document registry",
        processed_docs=processed_docs,
        total_docs=len(doc_paths),
        processed_chunks=len(new_records),
        total_chunks=len(new_records),
    )

    registry["docs"] = docs
    write_json(CONFIG.DOC_REGISTRY_PATH, registry)

    _emit_progress(
        progress_cb,
        stage="done",
        status="done",
        progress=100,
        message="Indexing completed successfully",
        processed_docs=processed_docs,
        total_docs=len(doc_paths),
        processed_chunks=len(new_records),
        total_chunks=len(new_records),
        added_docs=added_docs,
        added_chunks=len(new_records),
    )

    return {
        "added_docs": added_docs,
        "added_chunks": len(new_records),
        "status": "ok",
    }


def reindex_single_doc_keep_others(
    doc_path: Path,
    progress_cb: ProgressCallback = None,
) -> Dict[str, Any]:
    ensure_dirs(CONFIG.STORAGE_DIR, CONFIG.PDF_DIR, CONFIG.UPLOAD_DIR)

    doc_name = doc_path.name
    registry = read_json(CONFIG.DOC_REGISTRY_PATH, default={"docs": {}})
    docs: Dict[str, Any] = registry.get("docs", {}) or {}

    _emit_progress(
        progress_cb,
        stage="preparing",
        status="running",
        progress=2,
        message=f"Preparing re-index for {doc_name}",
        current_document=doc_name,
        processed_docs=0,
        total_docs=1,
    )

    embedder = get_embedder()
    embedder.max_seq_length = 512

    store = VectorStore(CONFIG.FAISS_INDEX_PATH, CONFIG.CHUNKS_PATH).load()

    _emit_progress(
        progress_cb,
        stage="removing_old_index",
        status="running",
        progress=6,
        message=f"Removing previous index for {doc_name}",
        current_document=doc_name,
        processed_docs=0,
        total_docs=1,
    )

    if hasattr(store, "delete_by_doc_name"):
        store.delete_by_doc_name(doc_name)

    old_chunks = store.chunks or []
    keep_chunks = [ch for ch in old_chunks if (ch or {}).get("doc_name") != doc_name]
    _rewrite_chunks_jsonl(Path(CONFIG.CHUNKS_PATH), keep_chunks)

    _emit_progress(
        progress_cb,
        stage="chunking",
        status="running",
        progress=10,
        message=f"Extracting content from {doc_name}",
        current_document=doc_name,
        processed_docs=0,
        total_docs=1,
    )

    chunks = _extract_chunks_any(
        doc_path,
        doc_name,
        progress_cb=progress_cb,
        progress_base=10,
        progress_span=22,
    )

    new_records: List[Dict[str, Any]] = []
    new_texts: List[str] = []

    for c in chunks:
        rec = _chunk_to_record(c)
        new_records.append(rec)
        new_texts.append("passage: " + (rec.get("text") or ""))

    if not new_texts:
        return {
            "status": "empty",
            "added_docs": 0,
            "added_chunks": 0,
            "message": "Tidak ada chunk yang diekstrak.",
        }

    _emit_progress(
        progress_cb,
        stage="embedding",
        status="running",
        progress=35,
        message=f"Generating embeddings for {len(new_texts)} chunks",
        current_document=doc_name,
        processed_docs=0,
        total_docs=1,
        processed_chunks=0,
        total_chunks=len(new_texts),
    )

    new_embs = _embed_texts(
        embedder,
        new_texts,
        progress_cb=progress_cb,
        progress_base=35,
        progress_span=40,
        current_document=doc_name,
    )

    _emit_progress(
        progress_cb,
        stage="writing_chunks",
        status="running",
        progress=76,
        message="Writing refreshed chunks to storage",
        current_document=doc_name,
        processed_docs=0,
        total_docs=1,
        processed_chunks=len(new_records),
        total_chunks=len(new_records),
    )

    for rec in new_records:
        append_jsonl(CONFIG.CHUNKS_PATH, rec)

    store.add(
        new_embs,
        new_records,
        progress_cb=progress_cb,
        progress_base=80,
        progress_span=15,
        current_document=doc_name,
    )

    old_keys = [k for k, v in docs.items() if (v or {}).get("doc_name") == doc_name]
    for k in old_keys:
        docs.pop(k, None)

    did = _doc_id(doc_path)
    docs[did] = {
        "doc_name": doc_name,
        "doc_path": str(doc_path),
        "num_chunks": len(chunks),
    }

    _emit_progress(
        progress_cb,
        stage="finalizing",
        status="running",
        progress=97,
        message=f"Updating registry for {doc_name}",
        current_document=doc_name,
        processed_docs=1,
        total_docs=1,
        processed_chunks=len(new_records),
        total_chunks=len(new_records),
    )

    registry["docs"] = docs
    write_json(CONFIG.DOC_REGISTRY_PATH, registry)

    _emit_progress(
        progress_cb,
        stage="done",
        status="done",
        progress=100,
        message=f"Re-index completed for {doc_name}",
        current_document=doc_name,
        processed_docs=1,
        total_docs=1,
        processed_chunks=len(new_records),
        total_chunks=len(new_records),
        added_docs=1,
        added_chunks=len(new_records),
    )

    return {"status": "ok", "added_docs": 1, "added_chunks": len(new_records)}


if __name__ == "__main__":
    ensure_dirs(CONFIG.PDF_DIR, CONFIG.STORAGE_DIR)

    docs = list_doc_files(CONFIG.PDF_DIR)
    print(f"[INFO] Ditemukan {len(docs)} dokumen di: {CONFIG.PDF_DIR}")

    if not docs:
        print("[WARN] Tidak ada dokumen untuk di-index. Taruh dokumen di data/pdf terlebih dahulu.")
    else:
        result = ingest_docs(docs)
        print("[DONE]", result)

