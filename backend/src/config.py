from dataclasses import dataclass
from pathlib import Path
import os


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no")


@dataclass(frozen=True)
class AppConfig:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

    DATA_DIR: Path = PROJECT_ROOT / "data"
    PDF_DIR: Path = DATA_DIR / "pdf"
    UPLOAD_DIR: Path = PDF_DIR

    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    CHUNKS_PATH: Path = STORAGE_DIR / "chunks.jsonl"
    FAISS_INDEX_PATH: Path = STORAGE_DIR / "vector_store"
    DOC_REGISTRY_PATH: Path = STORAGE_DIR / "doc_registry.json"

    # Eval / observability
    LLM_EVAL_LOG_PATH: Path = Path(os.getenv("LLM_EVAL_LOG_PATH", str(STORAGE_DIR / "llm_eval.jsonl")))
    ENABLE_LLM_EVAL_LOG: bool = _env_bool("ENABLE_LLM_EVAL_LOG", True)

    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LLAMA_GGUF_PATH: Path = MODELS_DIR / "gemma4" / "gemma-4-26B-A4B-it-Q4_K_M.gguf"

    EMBED_MODEL_NAME: str = os.getenv(
        "EMBED_MODEL_NAME",
        r"models\embeddings\multilingual-e5-small\models--intfloat--multilingual-e5-small\snapshots\c007d7ef6fd86656326059b28395a7a03a7c5846",
    )
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))
    EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cpu")

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "6"))
    RETRIEVAL_DEBUG: bool = _env_bool("RETRIEVAL_DEBUG", True)
    RETRIEVAL_CANDIDATE_K: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "24"))
    USE_MULTI_QUERY: bool = _env_bool("USE_MULTI_QUERY", True)
    MULTI_QUERY_MAX: int = int(os.getenv("MULTI_QUERY_MAX", "4"))

    MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "7000"))
    MAX_CHUNK_CHARS_IN_PROMPT: int = int(os.getenv("MAX_CHUNK_CHARS_IN_PROMPT", "700"))
    PER_DOC_QUOTA: int = int(os.getenv("PER_DOC_QUOTA", "3"))
    MIN_DOCS_QUOTA: int = int(os.getenv("MIN_DOCS_QUOTA", "1"))
    MAX_RETURN_SOURCES: int = int(os.getenv("MAX_RETURN_SOURCES", "6"))

    # Rerank
    USE_RERANK: bool = _env_bool("USE_RERANK", True)
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "16"))
    RERANK_MODEL_NAME: str = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    CHROMA_UPSERT_BATCH: int = int(os.getenv("CHROMA_UPSERT_BATCH", "2000"))

    # Chunking
    MIN_PARAGRAPH_CHARS: int = int(os.getenv("MIN_PARAGRAPH_CHARS", "120"))
    MAX_CHUNK_CHARS: int = int(os.getenv("MAX_CHUNK_CHARS", "1200"))

    # OCR
    USE_OCR: bool = _env_bool("USE_OCR", True)
    OCR_DPI: int = int(os.getenv("OCR_DPI", "110"))
    OCR_RENDER_WORKERS: int = int(os.getenv("OCR_RENDER_WORKERS", "2"))
    OCR_PREFETCH_PAGES: int = int(os.getenv("OCR_PREFETCH_PAGES", "4"))
    OCR_MAX_SIDE: int = int(os.getenv("OCR_MAX_SIDE", "1400"))

    OCR_MIN_CHARS: int = int(os.getenv("OCR_MIN_CHARS", "140"))
    OCR_MIN_ALNUM_RATIO: float = float(os.getenv("OCR_MIN_ALNUM_RATIO", "0.45"))

    PADDLE_OCR_LANG: str = os.getenv("PADDLE_OCR_LANG", "id")
    PADDLE_OCR_USE_GPU: bool = _env_bool("PADDLE_OCR_USE_GPU", False)
    PADDLE_OCR_USE_ANGLE_CLS: bool = _env_bool("PADDLE_OCR_USE_ANGLE_CLS", True)

    # LLM
    N_CTX: int = int(os.getenv("N_CTX", "262144"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "768"))
    LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "768"))
    PROMPT_OVERHEAD_TOKENS: int = int(os.getenv("PROMPT_OVERHEAD_TOKENS", "800"))

    N_THREADS: int = int(os.getenv("N_THREADS", "8"))
    N_GPU_LAYERS: int = int(os.getenv("N_GPU_LAYERS", "0"))


CONFIG = AppConfig()

