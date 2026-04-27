from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from app.services.state import STATE, get_embedder, get_llm, get_store
from src.config import CONFIG
from src.prompts import SYSTEM_PROMPT_ID, build_user_prompt


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def _store_ready(store: Any) -> bool:
    try:
        if hasattr(store, "is_ready"):
            return bool(store.is_ready())
    except Exception:
        pass

    try:
        return len(getattr(store, "chunks", []) or []) > 0
    except Exception:
        return False


def _is_all_docs(selected_doc: str) -> bool:
    return (selected_doc or "").strip().lower() == "semua dokumen"


def _norm_doc_name(x: Any) -> str:
    s = (str(x or "")).strip().lower().replace("\\", "/")
    s = s.split("/")[-1]
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _doc_match(doc_name: Any, selected_doc: str) -> bool:
    return _norm_doc_name(doc_name) == _norm_doc_name(selected_doc)


def _chunk_text(c: Dict[str, Any]) -> str:
    if not c:
        return ""

    v = c.get("text")
    if isinstance(v, str) and v.strip():
        return v.strip()

    for k in ("content", "chunk_text", "page_text", "body", "raw_text", "value"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for k in ("paragraphs", "paras", "lines"):
        v = c.get(k)
        if isinstance(v, list) and v:
            joined = "\n".join([str(x) for x in v if str(x).strip()])
            if joined.strip():
                return joined.strip()

    return ""


def _informativeness_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    chars = len(t)
    digits = sum(ch.isdigit() for ch in t)
    tokens = re.findall(r"[A-Za-z0-9]+", t.lower())
    uniq = len(set(tokens))
    return float(chars) * 0.001 + float(digits) * 0.15 + float(uniq) * 0.01


def _clean_query_for_retrieval(q: str) -> str:
    s = (q or "").strip()
    s = re.sub(r"\[SCOPE_DOKUMEN_AKTIF:[^\]]*\]\s*", "", s, flags=re.IGNORECASE)

    m = re.search(r"\(Pertanyaan sekarang\)\s*(.*)$", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = (m.group(1) or "").strip()

    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_llm_sources_section(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return a

    a = re.sub(
        r"(\n|\r|\r\n)\s*(Sumber|Sources)\s*:?\s*(.|\n|\r)*$",
        "",
        a,
        flags=re.IGNORECASE,
    ).strip()

    a = re.sub(
        r"(\n|\r|\r\n)\s*(Source|References)\s*:?\s*(.|\n|\r)*$",
        "",
        a,
        flags=re.IGNORECASE,
    ).strip()

    return a


def _is_summary_intent(q: str) -> bool:
    s = (q or "").lower()
    return bool(re.search(r"\b(rangkum|ringkas|ringkasan|resume|summarize|summary)\b", s))


def _extract_years(q: str) -> List[str]:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", q or "")
    years = [y for y in years if 1900 <= int(y) <= 2100]
    out: List[str] = []
    seen = set()
    for y in years:
        if y not in seen:
            out.append(y)
            seen.add(y)
    return out


def _expand_queries(message: str) -> List[str]:
    msg0 = _clean_query_for_retrieval(message)
    if not msg0:
        return []

    msg = msg0
    msg_l = msg.lower()
    seeds: List[str] = [msg]

    years = _extract_years(msg)
    compare_hint = bool(re.search(r"\b(bandingkan|perbanding|compare|comparison|tren|trend)\b", msg_l))

    synonyms: List[Tuple[re.Pattern, List[str]]] = [
        (
            re.compile(r"\b(aset|asset|aktiva)\s+lancar\b", flags=re.IGNORECASE),
            ["aset lancar", "aktiva lancar", "current assets", "jumlah aset lancar", "total aset lancar"],
        ),
        (
            re.compile(r"\b(total\s+aset|total\s+asset)\b", flags=re.IGNORECASE),
            ["total aset", "jumlah aset", "total assets"],
        ),
        (
            re.compile(r"\b(liabilitas|liability|kewajiban)\b", flags=re.IGNORECASE),
            ["liabilitas", "kewajiban", "liabilities"],
        ),
        (
            re.compile(r"\b(ekuitas|equity)\b", flags=re.IGNORECASE),
            ["ekuitas", "equity", "total ekuitas"],
        ),
        (
            re.compile(r"\b(arus\s+kas|cash\s+flow)\b", flags=re.IGNORECASE),
            ["arus kas", "cash flow", "arus kas bersih"],
        ),
        (
            re.compile(r"\b(laba\s+bersih|net\s+profit)\b", flags=re.IGNORECASE),
            ["laba bersih", "net profit", "profit setelah pajak"],
        ),
        (
            re.compile(r"\b(pendapatan|revenue)\b", flags=re.IGNORECASE),
            ["pendapatan", "revenue", "penjualan"],
        ),
    ]

    for pat, alts in synonyms:
        if pat.search(msg):
            for a in alts:
                if a.lower() != msg_l:
                    seeds.append(a)

    if re.search(r"\b(berapa|nominal|jumlah|total)\b", msg_l):
        if "berapa" in msg_l:
            seeds.append(msg.replace("berapa", "jumlah").strip())
        if "nominal" in msg_l:
            seeds.append(msg.replace("nominal", "jumlah").strip())

    if years:
        topic = re.sub(r"\b(19\d{2}|20\d{2})\b", "", msg).strip()
        topic = re.sub(r"\s+", " ", topic).strip()
        if topic:
            for y in years[:3]:
                seeds.append(f"{topic} {y}")
                seeds.append(f"{y} {topic}")
        if compare_hint and len(years) >= 2 and topic:
            seeds.append(f"perbandingan {topic} {years[0]} {years[-1]}")

    out: List[str] = []
    seen = set()
    for s in seeds:
        s2 = re.sub(r"\s+", " ", (s or "").strip())
        if not s2:
            continue
        k = s2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s2)

    max_q = int(getattr(CONFIG, "MULTI_QUERY_MAX", 4))
    return out[:max_q]


class _Reranker:
    def __init__(self):
        self._model = None
        self._device = None
        self._name = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        if not bool(getattr(CONFIG, "USE_RERANK", True)):
            self._model = False
            return

        model_name = str(getattr(CONFIG, "RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

        try:
            import torch
            from sentence_transformers import CrossEncoder
        except Exception:
            self._model = False
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._ensure_model()

        if self._model is False or self._model is None:
            for c in candidates:
                c["_rerank"] = None
            return candidates

        top_n = int(getattr(CONFIG, "RERANK_TOP_N", 16))
        cand = candidates[:top_n]

        pairs = [(query, _chunk_text(c)) for c in cand]
        try:
            scores = self._model.predict(pairs)
        except Exception:
            for c in candidates:
                c["_rerank"] = None
            return candidates

        for c, s in zip(cand, scores):
            c["_rerank"] = float(s)

        for c in candidates[top_n:]:
            c["_rerank"] = None

        def keyfn(x: Dict[str, Any]):
            rr = x.get("_rerank")
            if rr is None:
                rr = -1e9
            base = float(x.get("_base_score") or 0.0)
            info = float(x.get("_info") or 0.0)
            return (rr, base, info)

        return sorted(candidates, key=keyfn, reverse=True)

    @property
    def info(self) -> str:
        if self._model is False:
            return "rerank=disabled"
        if self._model is None:
            return "rerank=not_loaded"
        return f"rerank={self._name} device={self._device}"


_RERANKER = _Reranker()


@dataclass
class _Hit:
    score: float
    chunk: Dict[str, Any]


_ID_STOPWORDS = {
    "yang", "dan", "atau", "di", "ke", "dari", "pada", "untuk", "dengan", "sebagai",
    "adalah", "itu", "ini", "dalam", "antara", "oleh", "agar", "maka", "jika", "bila",
    "karena", "sehingga", "tidak", "bukan", "apa", "siapa", "kapan", "dimana", "bagaimana",
    "mengapa", "kenapa", "berapa", "jelaskan", "tolong", "mohon", "ringkas", "rangkum",
    "ringkasan", "resume", "the", "a", "an", "and", "or", "in", "on", "of", "to", "for",
    "with", "as", "is", "are", "was", "were", "what", "who", "when", "where", "why", "how",
    "please", "summarize", "summary",
}


def _tokenize_lex(q: str) -> List[str]:
    s = (q or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t and t not in _ID_STOPWORDS]
    return toks[:24]


def _lexical_score(query: str, text: str) -> float:
    q0 = (query or "").strip().lower()
    t0 = (text or "").lower()

    if not q0 or not t0:
        return 0.0

    toks = _tokenize_lex(q0)
    if not toks:
        return 0.0

    phrase_hit = 1.0 if q0 in t0 else 0.0

    hit = 0
    for tok in toks:
        if tok in t0:
            hit += 1

    overlap = hit / max(1, len(toks))
    all_hit = 1.0 if hit == len(toks) else 0.0

    label_bonus = 0.0
    if len(toks) <= 4:
        label_pat = r"\b" + r"\s+".join(map(re.escape, toks)) + r"\s*:"
        if re.search(label_pat, t0, flags=re.IGNORECASE):
            label_bonus = 0.75

    return phrase_hit * 2.5 + overlap * 1.5 + all_hit * 0.75 + label_bonus


def _need_lexical_fallback(query: str, merged_hits_len: int) -> bool:
    q = (query or "").strip()
    if not q:
        return False

    toks = _tokenize_lex(q)
    if ":" in q:
        return True
    if len(toks) <= 4:
        return True
    if merged_hits_len < 6:
        return True
    return False


def _lexical_search_chunks(
    store: Any,
    query: str,
    *,
    selected_doc: str,
    limit: int,
) -> List[_Hit]:
    chunks = getattr(store, "chunks", None) or []
    if not chunks:
        return []

    out: List[Tuple[float, Dict[str, Any]]] = []
    for c in chunks:
        if not isinstance(c, dict):
            continue

        if (not _is_all_docs(selected_doc)) and (not _doc_match(c.get("doc_name"), selected_doc)):
            continue

        txt = _chunk_text(c)
        if not txt:
            continue

        sc = _lexical_score(query, txt)
        if sc <= 0.0:
            continue

        out.append((sc, c))

    out.sort(key=lambda x: x[0], reverse=True)

    hits: List[_Hit] = []
    for sc, c in out[: max(1, int(limit))]:
        hits.append(_Hit(score=float(sc), chunk=c))
    return hits


def _source_location_str(c: Dict[str, Any]) -> str:
    kind = str(c.get("kind") or "")
    doc_name = str(c.get("doc_name") or "")

    if kind.startswith("pdf") or (c.get("page") not in (None, "", 0) and doc_name.lower().endswith(".pdf")):
        return f"Halaman {c.get('page')}"

    if kind.startswith("docx"):
        if c.get("table_index") is not None:
            return f"Tabel {c.get('table_index')}"
        ps = c.get("para_start")
        pe = c.get("para_end")
        if ps is not None and pe is not None:
            return f"Paragraf {ps}-{pe}" if ps != pe else f"Paragraf {ps}"
        if ps is not None:
            return f"Paragraf {ps}"
        return "DOCX"

    pg = c.get("page")
    return f"Lokasi {pg}" if pg is not None else "Lokasi"


def _normalize_boxes(raw_boxes: Any) -> List[List[float]]:
    out: List[List[float]] = []
    if not isinstance(raw_boxes, list):
        return out

    for r in raw_boxes:
        if not isinstance(r, (list, tuple)) or len(r) != 4:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in r]
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        out.append([x0, y0, x1, y1])
    return out


def _build_context_and_sources(
    hits: List[Any],
    selected_doc: str,
    *,
    query_text: str,
    max_context_chars: int,
    per_doc_quota: int,
    min_docs_quota: int,
    enable_debug: bool,
    use_rerank: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    filtered: List[Any] = []
    for h in hits:
        if isinstance(h, _Hit):
            c = h.chunk or {}
        else:
            c = getattr(h, "chunk", None) or {}

        if _is_all_docs(selected_doc) or _doc_match(c.get("doc_name"), selected_doc):
            filtered.append(h)

    seen = set()
    cands: List[Dict[str, Any]] = []
    for h in filtered:
        if isinstance(h, _Hit):
            c = h.chunk or {}
            base_score = float(h.score)
        else:
            c = getattr(h, "chunk", None) or {}
            base_score = getattr(h, "score", None)
            try:
                base_score = float(base_score) if base_score is not None else None
            except Exception:
                base_score = None

        content = _chunk_text(c)
        if not content:
            continue

        src_key = (c.get("doc_name"), c.get("chunk_id"))
        if src_key in seen:
            continue
        seen.add(src_key)

        cands.append({**c, "_base_score": base_score, "_info": _informativeness_score(content)})

    if not cands:
        return "", []

    cands.sort(
        key=lambda x: (
            x.get("_base_score") is not None,
            x.get("_base_score") or 0.0,
            x.get("_info") or 0.0,
        ),
        reverse=True,
    )

    if use_rerank:
        cands = _RERANKER.rerank(query_text, cands)
    else:
        for c in cands:
            c["_rerank"] = None

    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cands:
        by_doc[str(c.get("doc_name") or "")].append(c)

    doc_names = [d for d in by_doc.keys() if d]
    want_docs = min(max(min_docs_quota, 1), len(doc_names)) if _is_all_docs(selected_doc) else 1

    def doc_strength(d: str) -> float:
        top = by_doc[d][0] if by_doc[d] else None
        if not top:
            return -1e9
        rr = top.get("_rerank")
        if rr is None:
            rr = -1e9
        bs = float(top.get("_base_score") or 0.0)
        info = float(top.get("_info") or 0.0)
        if use_rerank:
            return float(rr) * 10.0 + bs + info * 0.1
        return bs + info * 0.1

    doc_order = sorted(doc_names, key=doc_strength, reverse=True)
    doc_order = doc_order[:want_docs] if _is_all_docs(selected_doc) else doc_order[:1]

    picked: List[Dict[str, Any]] = []
    picked_set = set()

    for d in doc_order:
        quota = int(per_doc_quota)
        for c in by_doc[d][: max(1, quota)]:
            key = (c.get("doc_name"), c.get("chunk_id"))
            if key in picked_set:
                continue
            picked.append(c)
            picked_set.add(key)

    for c in cands:
        key = (c.get("doc_name"), c.get("chunk_id"))
        if key in picked_set:
            continue
        picked.append(c)
        picked_set.add(key)

    blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
    total_chars = 0
    max_chunk_in_prompt = int(getattr(CONFIG, "MAX_CHUNK_CHARS_IN_PROMPT", 700))

    for c in picked:
        content = _chunk_text(c)
        if not content:
            continue

        loc = _source_location_str(c)
        header = f"[{c.get('doc_name')} | {loc} | {c.get('kind')}]"

        short = content[:max_chunk_in_prompt]
        block = f"{header}\n{short}"

        if total_chars + len(block) > max_context_chars:
            break

        blocks.append(block)
        total_chars += len(block)

        sources.append(
            {
                "doc_name": c.get("doc_name"),
                "doc_path": c.get("doc_path"),
                "kind": c.get("kind"),
                "chunk_id": c.get("chunk_id"),
                "snippet": content[:max_chunk_in_prompt],
                "table": c.get("table"),
                "bbox": c.get("bbox"),
                "paragraph_bboxes": _normalize_boxes(c.get("paragraph_bboxes")),
                "page_w": c.get("page_w"),
                "page_h": c.get("page_h"),
                "score": c.get("_base_score"),
                "rerank": c.get("_rerank"),
                "page": c.get("page"),
                "anchor": c.get("anchor"),
                "para_start": c.get("para_start"),
                "para_end": c.get("para_end"),
                "table_index": c.get("table_index"),
                "loc": loc,
            }
        )

    context = "\n\n---\n\n".join(blocks) if blocks else ""

    if enable_debug:
        dist = defaultdict(int)
        for s in sources:
            dist[str(s.get("doc_name") or "")] += 1
        dist_str = ", ".join([f"{k}:{v}" for k, v in sorted(dist.items(), key=lambda x: x[0]) if k])

        print("=== RETRIEVAL DEBUG ===", flush=True)
        print(f"selected_doc: {selected_doc!r}", flush=True)
        print(f"use_rerank={use_rerank} {_RERANKER.info}", flush=True)
        print(f"picked_sources: {len(sources)}", flush=True)
        print(f"CTX_CHARS: {len(context)} doc_dist: {dist_str}", flush=True)

        for i, s in enumerate(sources[:10], start=1):
            snippet = " ".join((s.get("snippet") or "").split())[:140]
            print(
                f"#{i} doc={s.get('doc_name')} loc={s.get('loc')} "
                f"base={s.get('score')} rerank={s.get('rerank')} text={snippet}",
                flush=True,
            )

    return context, sources


def _estimate_tokens(text: str) -> int:
    s = (text or "").strip()
    if not s:
        return 0
    return max(1, len(s) // 4)


def _eval_log_path() -> Path:
    p = getattr(CONFIG, "LLM_EVAL_LOG_PATH", None)
    if p:
        return Path(p)
    return Path(CONFIG.STORAGE_DIR) / "llm_eval.jsonl"


def _append_eval_log(payload: Dict[str, Any]) -> None:
    if not bool(getattr(CONFIG, "ENABLE_LLM_EVAL_LOG", True)):
        return

    try:
        path = _eval_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def generate_session_title(first_message: str, max_len: int = 44) -> str:
    text = (first_message or "").strip()
    text = re.sub(r"\s+", " ", text).strip().rstrip("?!.,")
    words = text.split()[:8]
    title = " ".join(words).strip()

    if not title:
        return "Obrolan Baru"

    title = title[:1].upper() + title[1:]
    if len(title) > max_len:
        title = title[:max_len].rstrip() + "…"
    return title


def _format_history_for_prompt(history: List[Dict[str, Any]], max_turns: int = 12) -> str:
    if not history:
        return ""

    recent = history[-max_turns:]
    lines: List[str] = []
    for m in recent:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    return "\n".join(lines).strip()


def _retrieve_hits(
    retrieval_query: str,
    *,
    selected_doc: str = "Semua dokumen",
    use_multi_query: bool = True,
) -> List[Any]:
    store = get_store()
    embedder = get_embedder()

    base_top_k = int(getattr(CONFIG, "TOP_K", 6))
    candidate_k = int(getattr(CONFIG, "RETRIEVAL_CANDIDATE_K", max(24, base_top_k * 4)))

    clean_q = _clean_query_for_retrieval(retrieval_query)

    if _is_summary_intent(clean_q) and not _is_all_docs(selected_doc):
        queries = [
            f"ringkasan {selected_doc}",
            "pendahuluan",
            "tujuan",
            "ruang lingkup",
            "metodologi",
            "temuan",
            "rekomendasi",
            "kesimpulan",
        ][: int(getattr(CONFIG, "MULTI_QUERY_MAX", 4))]
    else:
        queries = _expand_queries(clean_q) if use_multi_query else [clean_q]

    merged: Dict[str, Any] = {}

    for q in queries:
        q_emb = embedder.encode(["query: " + q], show_progress_bar=False)[0]
        q_emb = _normalize(q_emb)

        hits = store.search(q_emb, top_k=candidate_k)
        for h in hits:
            c = getattr(h, "chunk", None) or {}
            if (not _is_all_docs(selected_doc)) and (not _doc_match(c.get("doc_name"), selected_doc)):
                continue

            cid = str(c.get("chunk_id") or "")
            if not cid:
                continue

            if cid not in merged:
                merged[cid] = h
                continue

            try:
                if float(getattr(h, "score", 0.0)) > float(getattr(merged[cid], "score", 0.0)):
                    merged[cid] = h
            except Exception:
                pass

    merged_hits = list(merged.values())

    use_lex = bool(getattr(CONFIG, "USE_LEXICAL_FALLBACK", True))
    if use_lex and _need_lexical_fallback(clean_q, len(merged_hits)):
        lex_limit = int(getattr(CONFIG, "LEXICAL_FALLBACK_K", 20))
        lex_hits = _lexical_search_chunks(store, clean_q, selected_doc=selected_doc, limit=lex_limit)

        for lh in lex_hits:
            c = lh.chunk or {}
            cid = str(c.get("chunk_id") or "")
            if not cid:
                continue
            if cid not in merged:
                merged[cid] = lh

        merged_hits = list(merged.values())

    if bool(getattr(CONFIG, "RETRIEVAL_DEBUG", True)):
        print("=== RETRIEVAL QUERY DEBUG ===", flush=True)
        print(
            f"queries={len(queries)} candidate_k={candidate_k} selected_doc={selected_doc!r} "
            f"use_multi_query={use_multi_query} use_lex={use_lex} merged_hits={len(merged_hits)}",
            flush=True,
        )
        for i, q in enumerate(queries[:10], start=1):
            print(f"q{i}: {q}", flush=True)

    return merged_hits


def _llm_chat_safe(llm: Any, system_prompt: str, user_prompt: str, *, max_out: int) -> str:
    try:
        return llm.chat(system_prompt, user_prompt, max_tokens=max_out)
    except TypeError:
        try:
            return llm.chat(system_prompt, user_prompt, max_new_tokens=max_out)
        except TypeError:
            return llm.chat(system_prompt, user_prompt)


def generate_example_prompts(
    selected_doc: str = "Semua dokumen",
    *,
    max_examples: int = 4,
) -> List[str]:
    store = get_store()
    if not _store_ready(store):
        return []

    seed_query = (
        f"ringkasan {selected_doc}"
        if not _is_all_docs(selected_doc)
        else "ringkasan isi dokumen laporan keuangan audit temuan risiko kebijakan"
    )

    hits = _retrieve_hits(
        seed_query,
        selected_doc=selected_doc,
        use_multi_query=False,
    )

    _, sources = _build_context_and_sources(
        hits,
        selected_doc,
        query_text=seed_query,
        max_context_chars=int(getattr(CONFIG, "MAX_CONTEXT_CHARS", 4500)),
        per_doc_quota=2,
        min_docs_quota=1,
        enable_debug=bool(getattr(CONFIG, "RETRIEVAL_DEBUG", True)),
        use_rerank=False,
    )

    if not sources:
        return []

    texts = [str(s.get("snippet") or "").strip() for s in sources if str(s.get("snippet") or "").strip()]
    joined = "\n\n".join(texts)[:3000]

    if not joined:
        return []

    llm = get_llm()
    prompt = f"""Berdasarkan kutipan dokumen berikut, buat {max_examples} contoh pertanyaan yang wajar diajukan user.

ATURAN:
1. Semua pertanyaan harus dalam Bahasa Indonesia.
2. Pertanyaan harus spesifik, relevan, dan bisa dijawab dari kutipan.
3. Jangan gunakan bullet dengan simbol *.
4. Keluarkan HANYA daftar bernomor 1. sampai {max_examples}.
5. Jangan terlalu panjang. Setiap pertanyaan maksimal 18 kata.

KUTIPAN:
{joined}
"""

    raw = _llm_chat_safe(
        llm,
        "Anda membuat contoh pertanyaan singkat berdasarkan kutipan dokumen.",
        prompt,
        max_out=220,
    )

    lines = [re.sub(r"^\s*\d+[\.\)]\s*", "", x).strip() for x in raw.splitlines()]
    out: List[str] = []
    seen = set()
    for line in lines:
        if not line:
            continue
        if len(line) < 8:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= max_examples:
            break

    if out:
        return out

    fallback: List[str] = []
    if not _is_all_docs(selected_doc):
        fallback = [
            f"Apa ringkasan utama dari dokumen {selected_doc}?",
            f"Apa temuan penting dalam dokumen {selected_doc}?",
            f"Bagian mana yang membahas risiko pada {selected_doc}?",
            f"Apa rekomendasi utama dalam dokumen {selected_doc}?",
        ]
    else:
        fallback = [
            "Apa ringkasan isi dokumen yang terindeks?",
            "Apa temuan atau risiko yang paling penting?",
            "Dokumen mana yang membahas kebijakan atau prosedur?",
            "Apa rekomendasi utama dari dokumen yang tersedia?",
        ]
    return fallback[:max_examples]


def answer_question(message: str, selected_doc: str = "Semua dokumen") -> Tuple[str, List[Dict[str, Any]]]:
    store = get_store()
    llm = get_llm()

    if not _store_ready(store):
        return ("Index belum ada. Silakan index dokumen dulu.", [])

    retrieval_q = _clean_query_for_retrieval(message)
    hits = _retrieve_hits(retrieval_q, selected_doc=selected_doc, use_multi_query=bool(getattr(CONFIG, "USE_MULTI_QUERY", True)))

    context, sources = _build_context_and_sources(
        hits,
        selected_doc,
        query_text=retrieval_q,
        max_context_chars=int(getattr(CONFIG, "MAX_CONTEXT_CHARS", 7000)),
        per_doc_quota=int(getattr(CONFIG, "PER_DOC_QUOTA", 3)),
        min_docs_quota=int(getattr(CONFIG, "MIN_DOCS_QUOTA", 1)),
        enable_debug=bool(getattr(CONFIG, "RETRIEVAL_DEBUG", True)),
        use_rerank=bool(getattr(CONFIG, "USE_RERANK", True)),
    )

    scope_note = f"[SCOPE_DOKUMEN_AKTIF: {selected_doc}]"
    scoped_message = f"{scope_note}\n{message}"

    user_prompt = build_user_prompt(
        scoped_message,
        context_blocks=context if context else "(Tidak ada konteks yang ditemukan.)",
    )

    max_out = int(getattr(CONFIG, "LLM_MAX_OUTPUT_TOKENS", 768))
    answer = _llm_chat_safe(llm, SYSTEM_PROMPT_ID, user_prompt, max_out=max_out)
    answer = _strip_llm_sources_section(answer)

    max_sources = int(getattr(CONFIG, "MAX_RETURN_SOURCES", 6))
    return answer, sources[:max_sources]


def stream_answer_tokens(
    message: str,
    selected_doc: str = "Semua dokumen",
    *,
    retrieval_query: Optional[str] = None,
) -> Tuple[Iterable[str], List[Dict[str, Any]], Dict[str, Any]]:
    store = get_store()
    llm = get_llm()

    if not _store_ready(store):

        def _one():
            yield "Index belum ada. Silakan index dokumen dulu."

        eval_meta = {
            "status": "index_not_ready",
            "selected_doc": selected_doc,
            "retrieval_query": _clean_query_for_retrieval(retrieval_query if retrieval_query is not None else message),
            "input_chars": len(message or ""),
            "input_tokens_est": _estimate_tokens(message or ""),
            "output_tokens_est": 0,
            "total_tokens_est": _estimate_tokens(message or ""),
        }
        return _one(), [], eval_meta

    eval_meta: Dict[str, Any] = {
        "status": "started",
        "selected_doc": selected_doc,
        "message_chars": len(message or ""),
        "message_tokens_est": _estimate_tokens(message or ""),
    }

    t0 = time.perf_counter()

    rq = _clean_query_for_retrieval(retrieval_query if retrieval_query is not None else message)
    eval_meta["retrieval_query"] = rq

    t_retrieval_start = time.perf_counter()
    hits = _retrieve_hits(
        rq,
        selected_doc=selected_doc,
        use_multi_query=bool(getattr(CONFIG, "USE_MULTI_QUERY", True)),
    )
    context, sources = _build_context_and_sources(
        hits,
        selected_doc,
        query_text=rq,
        max_context_chars=int(getattr(CONFIG, "MAX_CONTEXT_CHARS", 7000)),
        per_doc_quota=int(getattr(CONFIG, "PER_DOC_QUOTA", 3)),
        min_docs_quota=int(getattr(CONFIG, "MIN_DOCS_QUOTA", 1)),
        enable_debug=bool(getattr(CONFIG, "RETRIEVAL_DEBUG", True)),
        use_rerank=bool(getattr(CONFIG, "USE_RERANK", True)),
    )
    t_retrieval_end = time.perf_counter()

    eval_meta["retrieval_ms"] = round((t_retrieval_end - t_retrieval_start) * 1000, 2)
    eval_meta["retrieved_hits"] = len(hits)
    eval_meta["returned_sources"] = min(len(sources), int(getattr(CONFIG, "MAX_RETURN_SOURCES", 6)))
    eval_meta["context_chars"] = len(context or "")

    scope_note = f"[SCOPE_DOKUMEN_AKTIF: {selected_doc}]"
    scoped_message = f"{scope_note}\n{message}"

    t_prompt_start = time.perf_counter()
    user_prompt = build_user_prompt(
        scoped_message,
        context_blocks=context if context else "(Tidak ada konteks yang ditemukan.)",
    )
    t_prompt_end = time.perf_counter()

    eval_meta["prompt_build_ms"] = round((t_prompt_end - t_prompt_start) * 1000, 2)
    eval_meta["system_prompt_chars"] = len(SYSTEM_PROMPT_ID or "")
    eval_meta["user_prompt_chars"] = len(user_prompt or "")
    eval_meta["input_chars"] = len((SYSTEM_PROMPT_ID or "")) + len((user_prompt or ""))
    eval_meta["input_tokens_est"] = _estimate_tokens(SYSTEM_PROMPT_ID or "") + _estimate_tokens(user_prompt or "")

    max_out = int(getattr(CONFIG, "LLM_MAX_OUTPUT_TOKENS", 768))
    eval_meta["max_output_tokens"] = max_out

    max_sources = int(getattr(CONFIG, "MAX_RETURN_SOURCES", 6))
    final_sources = sources[:max_sources]

    def locked_token_iter():
        output_parts: List[str] = []
        t_llm_start = time.perf_counter()
        first_token_sent = False

        try:
            with STATE.llm_lock:
                try:
                    raw_iter = llm.stream_chat(SYSTEM_PROMPT_ID, user_prompt, max_tokens=max_out)
                except TypeError:
                    try:
                        raw_iter = llm.stream_chat(SYSTEM_PROMPT_ID, user_prompt, max_new_tokens=max_out)
                    except TypeError:
                        raw_iter = llm.stream_chat(SYSTEM_PROMPT_ID, user_prompt)

                for t in raw_iter:
                    if not first_token_sent:
                        eval_meta["first_token_ms"] = round((time.perf_counter() - t_llm_start) * 1000, 2)
                        first_token_sent = True
                    output_parts.append(t)
                    yield t

            final_answer = _strip_llm_sources_section("".join(output_parts).strip())
            eval_meta["output_chars"] = len(final_answer)
            eval_meta["output_tokens_est"] = _estimate_tokens(final_answer)
            eval_meta["total_tokens_est"] = int(eval_meta.get("input_tokens_est", 0)) + int(eval_meta.get("output_tokens_est", 0))
            eval_meta["llm_total_ms"] = round((time.perf_counter() - t_llm_start) * 1000, 2)
            eval_meta["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            eval_meta["status"] = "done"

            _append_eval_log(
                {
                    "ts": int(time.time()),
                    "selected_doc": selected_doc,
                    "retrieval_query": rq,
                    "message": message,
                    "metrics": dict(eval_meta),
                }
            )
        except Exception as e:
            eval_meta["status"] = "error"
            eval_meta["error"] = str(e)
            eval_meta["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            _append_eval_log(
                {
                    "ts": int(time.time()),
                    "selected_doc": selected_doc,
                    "retrieval_query": rq,
                    "message": message,
                    "metrics": dict(eval_meta),
                }
            )
            raise

    return locked_token_iter(), final_sources, eval_meta


def stream_answer_tokens_with_history(
    message: str,
    selected_doc: str = "Semua dokumen",
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Iterable[str], List[Dict[str, Any]], Dict[str, Any]]:
    hist_text = _format_history_for_prompt(history or [])
    llm_message = (
        f"(Percakapan sebelumnya)\n{hist_text}\n\n(Pertanyaan sekarang)\n{message}"
        if hist_text
        else message
    )

    token_iter, sources, eval_meta = stream_answer_tokens(
        llm_message,
        selected_doc=selected_doc,
        retrieval_query=message,
    )

    eval_meta["history_chars"] = len(hist_text or "")
    eval_meta["history_tokens_est"] = _estimate_tokens(hist_text or "")

    return token_iter, sources, eval_meta

