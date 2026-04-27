from __future__ import annotations
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class Job:
    job_id: str
    created_at: float
    message: str
    selected_pdf: str

class ChatQueue:
    def __init__(self):
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._lock = threading.Lock()
        self._pending: List[Job] = []

    def enqueue(self, message: str, selected_pdf: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            job_id=job_id,
            created_at=time.time(),
            message=message,
            selected_pdf=selected_pdf,
        )
        with self._lock:
            self._pending.append(job)
        self._q.put(job)
        return job_id

    def get_position(self, job_id: str) -> Optional[int]:
        with self._lock:
            for i, j in enumerate(self._pending):
                if j.job_id == job_id:
                    return i
        return None

    def pop_next(self) -> Job:
        job = self._q.get()
        with self._lock:
            self._pending = [j for j in self._pending if j.job_id != job.job_id]
        return job

    def mark_done(self) -> None:
        self._q.task_done()

CHAT_QUEUE = ChatQueue()