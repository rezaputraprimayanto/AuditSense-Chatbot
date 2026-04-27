from __future__ import annotations
import json
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import CONFIG

SESS_DIR = CONFIG.STORAGE_DIR / "sessions"
SESS_DIR.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()

@dataclass
class Session:
    session_id: str
    title: str = "Obrolan Baru"
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = time.time()

def _safe_session_id(session_id: str) -> str:
    return Path(session_id).name.strip()

def _session_path(session_id: str) -> Path:
    sid = _safe_session_id(session_id)
    return SESS_DIR / f"{sid}.json"

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def new_session_id() -> str:
    return secrets.token_hex(6)

def load_session(session_id: str) -> Dict[str, Any]:
    data = _read_json(_session_path(session_id))
    if not data:
        return {"error": "not_found"}
    return data

def save_session(sess_dict: Dict[str, Any]) -> None:
    sid = _safe_session_id(str(sess_dict.get("session_id") or ""))
    if not sid:
        return

    with _LOCK:
        path = _session_path(sid)
        cur = _read_json(path)
        if not cur:
            cur = asdict(Session(session_id=sid))

        cur["session_id"] = sid
        cur["title"] = sess_dict.get("title", cur.get("title", "Obrolan Baru"))
        cur["created_at"] = sess_dict.get("created_at", cur.get("created_at", time.time()))
        cur["updated_at"] = sess_dict.get("updated_at", cur.get("updated_at", time.time()))
        cur["messages"] = sess_dict.get("messages", cur.get("messages", []))

        _write_json(path, cur)

def append_message(
    session_id: str,
    role: str,
    content: str,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> None:
    sid = _safe_session_id(session_id)
    if not sid:
        return

    with _LOCK:
        path = _session_path(sid)
        cur = _read_json(path)
        if not cur:
            cur = asdict(Session(session_id=sid))

        msg: Dict[str, Any] = {
            "role": role,
            "content": content,
            "ts": time.time(),
        }
        if citations is not None:
            msg["citations"] = citations

        cur_msgs = cur.get("messages") or []
        cur_msgs.append(msg)
        cur["messages"] = cur_msgs
        cur["updated_at"] = time.time()

        _write_json(path, cur)

def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for p in SESS_DIR.glob("*.json"):
        data = _read_json(p)
        if not data:
            continue

        msgs = data.get("messages") or []
        items.append(
            {
                "session_id": data.get("session_id") or p.stem,
                "title": data.get("title", "Obrolan Baru"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "num_messages": len(msgs),
            }
        )

    items.sort(key=lambda x: (x.get("updated_at") or 0), reverse=True)
    return items[:limit]