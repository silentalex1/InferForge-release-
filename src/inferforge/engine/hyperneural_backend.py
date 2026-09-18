from __future__ import annotations

import time
from typing import Any, Iterator

import httpx

from inferforge.core.registry import ModelRecord
from inferforge.engine.base import ChatEngine, ChatMessage


class HyperNeuralEngine(ChatEngine):
    def __init__(self, model: ModelRecord, timeout: float = 30.0):
        self.model_name = model.name
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[ChatMessage], system: str | None = None, options: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if system:
            payload["messages"] = [{"role": "system", "content": system}] + payload["messages"]
        time.sleep(0.35)
        r = self._client.post("https://hyperneural.cfd/api/ai/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content") or data.get("response") or ""

    def stream_chat(self, messages: list[ChatMessage], system: str | None = None, options: dict[str, Any] | None = None) -> Iterator[str]:
        text = self.chat(messages, system, options)
        for chunk in [text[i:i+24] for i in range(0, len(text), 24)]:
            time.sleep(0.04)
            yield chunk

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
