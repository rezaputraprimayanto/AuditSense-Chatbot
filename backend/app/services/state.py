from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict

from sentence_transformers import SentenceTransformer

from src.config import CONFIG
from src.llm import LLMConfig, LocalLlama
from src.vector_store import VectorStore


@dataclass
class AppState:
    store: Optional[VectorStore] = None
    store_chunks_mtime: float = 0.0
    store_registry_mtime: float = 0.0
    store_chroma_mtime: float = 0.0

    embedder: Optional[SentenceTransformer] = None
    llm: Optional[LocalLlama] = None
    llm_lock: threading.Lock = field(default_factory=threading.Lock)

    reranker: Optional[Any] = None
    reranker_name: str = "not_loaded"
    reranker_device: str = "none"

    index_jobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    index_jobs_lock: threading.Lock = field(default_factory=threading.Lock)


STATE = AppState()


def _file_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0


def _chroma_marker_mtime(persist_dir: Path) -> float:
    """
    Chroma PersistentClient writes multiple files inside persist_dir.
    We use a stable marker: pick the newest mtime among common files/folders.
    If directory does not exist, return 0.
    """
    try:
        if not persist_dir.exists():
            return 0.0

        newest = 0.0
        for child in persist_dir.iterdir():
            try:
                newest = max(newest, child.stat().st_mtime)
            except Exception:
                pass
        return newest
    except Exception:
        return 0.0


def get_store() -> VectorStore:
    chunks_path = Path(CONFIG.CHUNKS_PATH)
    registry_path = Path(CONFIG.DOC_REGISTRY_PATH)

    chunks_mtime = _file_mtime(chunks_path)
    registry_mtime = _file_mtime(registry_path)

    if STATE.store is None:
        store = VectorStore(CONFIG.FAISS_INDEX_PATH, CONFIG.CHUNKS_PATH).load()
        chroma_mtime = _chroma_marker_mtime(store.persist_dir)

        STATE.store = store
        STATE.store_chunks_mtime = chunks_mtime
        STATE.store_registry_mtime = registry_mtime
        STATE.store_chroma_mtime = chroma_mtime
        return STATE.store

    chroma_mtime_now = _chroma_marker_mtime(STATE.store.persist_dir)

    changed = (
        chunks_mtime != STATE.store_chunks_mtime
        or registry_mtime != STATE.store_registry_mtime
        or chroma_mtime_now != STATE.store_chroma_mtime
    )

    if changed:
        try:
            STATE.store.load()
        except Exception:
            STATE.store = VectorStore(CONFIG.FAISS_INDEX_PATH, CONFIG.CHUNKS_PATH).load()

        STATE.store_chunks_mtime = chunks_mtime
        STATE.store_registry_mtime = registry_mtime
        STATE.store_chroma_mtime = _chroma_marker_mtime(STATE.store.persist_dir)

    return STATE.store


def get_embedder() -> SentenceTransformer:
    if STATE.embedder is None:
        model_path = str(CONFIG.EMBED_MODEL_NAME)

        STATE.embedder = SentenceTransformer(
            model_path,
            device=str(getattr(CONFIG, "EMBED_DEVICE", "cpu")),
            local_files_only=True,
        )
        STATE.embedder.max_seq_length = 512

    return STATE.embedder


def get_llm() -> LocalLlama:
    if STATE.llm is None:
        STATE.llm = LocalLlama(
            LLMConfig(
                model_path=str(CONFIG.LLAMA_GGUF_PATH),
                n_ctx=CONFIG.N_CTX,
                n_threads=CONFIG.N_THREADS,
                n_gpu_layers=CONFIG.N_GPU_LAYERS,
                temperature=CONFIG.TEMPERATURE,
                max_tokens=CONFIG.MAX_TOKENS,
            )
        ).load()
    return STATE.llm


def reset_store_cache() -> None:
    STATE.store = None
    STATE.store_chunks_mtime = 0.0
    STATE.store_registry_mtime = 0.0
    STATE.store_chroma_mtime = 0.0


def reset_embedder_cache() -> None:
    STATE.embedder = None


def reset_reranker_cache() -> None:
    STATE.reranker = None
    STATE.reranker_name = "not_loaded"
    STATE.reranker_device = "none"


def _now_ts() -> float:
    return time.time()


def create_index_job(
    *,
    scope: str,
    force: bool,
    total_docs: int = 0,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    job_id = f"idx_{uuid.uuid4().hex[:12]}"
    now = _now_ts()

    job = {
        "job_id": job_id,
        "type": "reindex" if force else "index",
        "scope": scope,
        "force": force,
        "title": title or ("Re-indexing documents" if force else "Indexing documents"),
        "status": "queued",
        "stage": "queued",
        "message": "Waiting to start indexing process",
        "progress": 0,
        "current_document": None,
        "current_page": None,
        "total_pages": None,
        "processed_docs": 0,
        "total_docs": max(0, int(total_docs)),
        "processed_chunks": 0,
        "total_chunks": 0,
        "processed_batches": 0,
        "total_batches": 0,
        "added_docs": 0,
        "added_chunks": 0,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "finished_at": None,
    }

    with STATE.index_jobs_lock:
        STATE.index_jobs[job_id] = job

    return job.copy()


def get_index_job(job_id: str) -> Optional[Dict[str, Any]]:
    with STATE.index_jobs_lock:
        job = STATE.index_jobs.get(job_id)
        return job.copy() if job else None


def update_index_job(job_id: str, **patch: Any) -> Optional[Dict[str, Any]]:
    with STATE.index_jobs_lock:
        job = STATE.index_jobs.get(job_id)
        if not job:
            return None

        for key, value in patch.items():
            if value is not None:
                job[key] = value

        if "progress" in patch and patch.get("progress") is not None:
            try:
                p = int(patch["progress"])
            except Exception:
                p = 0
            job["progress"] = max(0, min(100, p))

        job["updated_at"] = _now_ts()
        return job.copy()


def start_index_job(
    job_id: str,
    *,
    title: Optional[str] = None,
    message: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    return update_index_job(
        job_id,
        status="running",
        stage="preparing",
        progress=1,
        title=title,
        message=message or "Preparing indexing pipeline",
        error=None,
    )


def finish_index_job(
    job_id: str,
    *,
    message: str = "Indexing completed successfully",
    added_docs: int = 0,
    added_chunks: int = 0,
) -> Optional[Dict[str, Any]]:
    with STATE.index_jobs_lock:
        job = STATE.index_jobs.get(job_id)
        if not job:
            return None

        job["status"] = "done"
        job["stage"] = "done"
        job["progress"] = 100
        job["message"] = message
        job["added_docs"] = int(added_docs)
        job["added_chunks"] = int(added_chunks)
        job["error"] = None
        job["updated_at"] = _now_ts()
        job["finished_at"] = job["updated_at"]
        return job.copy()


def fail_index_job(job_id: str, message: str) -> Optional[Dict[str, Any]]:
    with STATE.index_jobs_lock:
        job = STATE.index_jobs.get(job_id)
        if not job:
            return None

        job["status"] = "error"
        job["stage"] = "error"
        job["message"] = message or "Indexing failed"
        job["error"] = message or "Indexing failed"
        job["updated_at"] = _now_ts()
        job["finished_at"] = job["updated_at"]
        return job.copy()


def remove_index_job(job_id: str) -> None:
    with STATE.index_jobs_lock:
        STATE.index_jobs.pop(job_id, None)


def cleanup_finished_index_jobs(max_age_seconds: int = 900) -> None:
    now = _now_ts()
    with STATE.index_jobs_lock:
        to_delete = []
        for job_id, job in STATE.index_jobs.items():
            status = str(job.get("status") or "")
            finished_at = job.get("finished_at")
            if status in {"done", "error"} and isinstance(finished_at, (int, float)):
                if now - float(finished_at) > max_age_seconds:
                    to_delete.append(job_id)

        for job_id in to_delete:
            STATE.index_jobs.pop(job_id, None)

