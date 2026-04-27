from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from .config import CONFIG
from .utils import read_jsonl


@dataclass
class SearchHit:
    score: float
    chunk: Dict[str, Any]


ProgressCallback = Optional[Callable[..., None]]


class VectorStore:
    def __init__(self, index_path: Path, chunks_path: Path):
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)

        self.persist_dir = self._resolve_persist_dir(self.index_path)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
        )

        self.chunks: List[Dict[str, Any]] = []
        self._chunk_by_id: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _resolve_persist_dir(index_path: Path) -> Path:
        if index_path.exists() and index_path.is_dir():
            return index_path
        if index_path.suffix == "":
            return index_path
        return index_path.parent / "chroma"

    def _refresh_chunks_cache(self) -> None:
        self.chunks = read_jsonl(self.chunks_path)
        self._chunk_by_id = {
            str(c.get("chunk_id")): c
            for c in self.chunks
            if c.get("chunk_id") is not None
        }

    def load(self) -> "VectorStore":
        self._refresh_chunks_cache()

        try:
            if self.collection.count() == 0 and self._chunk_by_id:
                self._rebuild_from_chunks_jsonl()
        except Exception:
            pass

        return self

    def _rebuild_from_chunks_jsonl(self) -> None:
        try:
            self.client.delete_collection("chunks")
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection("chunks")
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
        )

        if self.chunks_path.exists():
            try:
                self.chunks_path.unlink()
            except Exception:
                pass

        self.chunks = []
        self._chunk_by_id = {}

    def build_new(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
        progress_cb: ProgressCallback = None,
        progress_base: int = 80,
        progress_span: int = 15,
        current_document: Optional[str] = None,
    ) -> None:
        self.reset()
        self._upsert(
            embeddings,
            chunks,
            progress_cb=progress_cb,
            progress_base=progress_base,
            progress_span=progress_span,
            current_document=current_document,
        )

    def add(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
        progress_cb: ProgressCallback = None,
        progress_base: int = 80,
        progress_span: int = 15,
        current_document: Optional[str] = None,
    ) -> None:
        self._upsert(
            embeddings,
            chunks,
            progress_cb=progress_cb,
            progress_base=progress_base,
            progress_span=progress_span,
            current_document=current_document,
        )

    def _chunk_to_metadata(self, cid: str, ch: Dict[str, Any]) -> Dict[str, Any]:
        def clean(v: Any):
            if v is None:
                return ""
            if isinstance(v, (str, int, float, bool)):
                return v
            return str(v)

        def clean_int(v: Any) -> int:
            try:
                return int(v)
            except Exception:
                return 0

        return {
            "chunk_id": str(cid),
            "doc_name": clean(ch.get("doc_name")),
            "doc_path": clean(ch.get("doc_path")),
            "page": clean_int(ch.get("page")),
            "kind": clean(ch.get("kind")),
            "anchor": clean(ch.get("anchor")),
            "para_start": clean_int(ch.get("para_start")),
            "para_end": clean_int(ch.get("para_end")),
            "table_index": clean_int(ch.get("table_index")),
        }

    def _emit_store_progress(
        self,
        *,
        progress_cb: ProgressCallback,
        progress_base: int,
        progress_span: int,
        processed_batches: int,
        total_batches: int,
        processed_chunks: int,
        total_chunks: int,
        current_document: Optional[str],
    ) -> None:
        if not progress_cb:
            return

        ratio = (processed_batches / total_batches) if total_batches > 0 else 1.0
        progress = int(progress_base + (progress_span * ratio))
        progress = max(0, min(100, progress))

        progress_cb(
            stage="storing",
            status="running",
            progress=progress,
            message=f"Writing vector batch {processed_batches}/{total_batches}",
            current_document=current_document,
            processed_batches=processed_batches,
            total_batches=total_batches,
            processed_chunks=processed_chunks,
            total_chunks=total_chunks,
        )

    def _upsert(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
        progress_cb: ProgressCallback = None,
        progress_base: int = 80,
        progress_span: int = 15,
        current_document: Optional[str] = None,
    ) -> None:
        if embeddings is None or not chunks:
            return

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        embeddings = embeddings.astype(np.float32, copy=False)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        base = len(self._chunk_by_id)

        for j, ch in enumerate(chunks):
            cid = ch.get("chunk_id") or ch.get("id")
            cid = str(cid) if cid is not None else f"chunk_{base + j}"
            ids.append(cid)

            text = ch.get("text") or ch.get("content") or ch.get("chunk") or ""
            documents.append(str(text))

            metadatas.append(self._chunk_to_metadata(cid, ch))

        n = len(ids)
        if embeddings.shape[0] != n:
            raise ValueError(f"Embeddings rows ({embeddings.shape[0]}) != chunks ({n})")

        batch_size = int(getattr(CONFIG, "CHROMA_UPSERT_BATCH", 2000))
        total_batches = max(1, (n + batch_size - 1) // batch_size)

        if progress_cb:
            self._emit_store_progress(
                progress_cb=progress_cb,
                progress_base=progress_base,
                progress_span=progress_span,
                processed_batches=0,
                total_batches=total_batches,
                processed_chunks=0,
                total_chunks=n,
                current_document=current_document,
            )

        processed_chunks = 0
        processed_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

            processed_batches += 1
            processed_chunks = end

            self._emit_store_progress(
                progress_cb=progress_cb,
                progress_base=progress_base,
                progress_span=progress_span,
                processed_batches=processed_batches,
                total_batches=total_batches,
                processed_chunks=processed_chunks,
                total_chunks=n,
                current_document=current_document,
            )

        for ch in chunks:
            cid2 = ch.get("chunk_id") or ch.get("id")
            if cid2 is not None:
                self._chunk_by_id[str(cid2)] = ch

        self.chunks = list(self._chunk_by_id.values())

    def search(self, query_emb: np.ndarray, top_k: int) -> List[SearchHit]:
        if query_emb is None:
            return []

        if not isinstance(query_emb, np.ndarray):
            query_emb = np.array(query_emb, dtype=np.float32)

        query_emb = query_emb.astype(np.float32, copy=False).reshape(-1)

        try:
            count = int(self.collection.count())
        except Exception:
            count = 0

        n_results = max(1, min(int(top_k), count)) if count > 0 else 0
        if n_results == 0:
            return []

        res = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=n_results,
            include=["metadatas", "distances", "documents"],
        )

        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        documents = res.get("documents", [[]])[0]

        out: List[SearchHit] = []
        refreshed = False

        for md, dist, doc_text in zip(metadatas, distances, documents):
            cid = (md or {}).get("chunk_id")
            if not cid:
                continue

            if str(cid) not in self._chunk_by_id and not refreshed:
                self._refresh_chunks_cache()
                refreshed = True

            full = self._chunk_by_id.get(str(cid))
            if not full:
                full = {
                    "chunk_id": cid,
                    "doc_name": (md or {}).get("doc_name"),
                    "doc_path": (md or {}).get("doc_path"),
                    "page": (md or {}).get("page"),
                    "kind": (md or {}).get("kind"),
                    "anchor": (md or {}).get("anchor"),
                    "para_start": (md or {}).get("para_start"),
                    "para_end": (md or {}).get("para_end"),
                    "table_index": (md or {}).get("table_index"),
                    "text": doc_text or "",
                }

            score = 1.0 - float(dist)
            out.append(SearchHit(score=score, chunk=full))

        return out

    def delete_by_doc_name(self, doc_name: str) -> None:
        self.collection.delete(where={"doc_name": doc_name})

        self.chunks = [c for c in self.chunks if c.get("doc_name") != doc_name]
        self._chunk_by_id = {
            str(c.get("chunk_id")): c
            for c in self.chunks
            if c.get("chunk_id") is not None
        }

    def is_ready(self) -> bool:
        try:
            return self.collection.count() > 0
        except Exception:
            return False

