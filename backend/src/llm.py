from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

from llama_cpp import Llama


@dataclass(frozen=True)
class LLMConfig:
    model_path: str
    n_ctx: int
    n_threads: int
    n_gpu_layers: int
    temperature: float
    max_tokens: int


class LocalLlama:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._llm: Optional[Llama] = None

    def load(self) -> "LocalLlama":
        if self._llm is not None:
            return self

        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.cfg.n_ctx,
            n_threads=self.cfg.n_threads,
            n_gpu_layers=self.cfg.n_gpu_layers,
            verbose=False,
        )
        return self

    def _resolve_max_tokens(
        self,
        *,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> int:
        if max_tokens is not None:
            return int(max_tokens)
        if max_new_tokens is not None:
            return int(max_new_tokens)
        return int(self.cfg.max_tokens)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stop: Optional[Sequence[str]] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        self.load()

        resolved_max_tokens = self._resolve_max_tokens(
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )

        resp = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=resolved_max_tokens,
            stop=stop,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()

    def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stop: Optional[Sequence[str]] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        self.load()

        resolved_max_tokens = self._resolve_max_tokens(
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )

        stream = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=resolved_max_tokens,
            stop=stop,
            stream=True,
        )

        for chunk in stream:
            try:
                delta = chunk["choices"][0]["delta"]
                token = delta.get("content")
                if token:
                    yield token
            except Exception:
                continue

