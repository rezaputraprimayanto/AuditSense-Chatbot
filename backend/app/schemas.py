from typing import Any, List, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_name: str
    doc_path: str


class IndexResponse(BaseModel):
    status: str
    added_docs: int = 0
    added_chunks: int = 0
    message: Optional[str] = None


class Citation(BaseModel):
    doc_name: str
    doc_path: str
    page: Optional[int] = None
    kind: str = "text"
    chunk_id: Optional[str] = None
    snippet: Optional[str] = None
    table: Optional[Any] = None
    bbox: Optional[Any] = None
    page_w: Optional[float] = None
    page_h: Optional[float] = None
    score: Optional[float] = None

    anchor: Optional[str] = None
    para_start: Optional[int] = None
    para_end: Optional[int] = None
    table_index: Optional[int] = None
    loc: Optional[str] = None
    rerank: Optional[float] = None


class ChatDone(BaseModel):
    answer: str
    citations: List[Citation]

