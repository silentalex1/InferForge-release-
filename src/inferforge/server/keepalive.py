from __future__ import annotations

import threading
import time
from typing import Any, Callable

import httpx

from inferforge.core.config import load_settings


def _ollama_host() -> str:
    settings = load_settings()
    return (settings.get("ollama_host") or "http://127.0.0.1:11434").rstrip("/")


def warm_model(model: str, host: str | None = None, keep_alive: int | str = -1, timeout: float = 120.0) -> bool:
    target = (host or _ollama_host()).rstrip("/")
    try:
        r = httpx.post(
            f"{target}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": keep_alive, "stream": False},
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


class KeepAliveDaemon:
    def __init__(
        self,
        models: list[str],
        host: str | None = None,
        interval: float | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        settings = load_settings()
        self.models = list(dict.fromkeys(models))
        self.host = (host or _ollama_host()).rstrip("/")
        self.interval = float(interval if interval is not None else settings.get("keep_alive_interval", 240))
        self.on_event = on_event
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client = httpx.Client(base_url=self.host, timeout=120.0)

    def _emit(self, msg: str) -> None:
        if self.on_event:
            try:
                self.on_event(msg)
            except Exception:
                pass

    def _ping(self, model: str) -> bool:
        try:
            r = self._client.post(
                "/api/generate",
                json={"model": model, "prompt": "", "keep_alive": -1, "stream": False},
                timeout=120.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            for model in self.models:
                if self._stop.is_set():
                    break
                if not self._ping(model):
                    self._emit(f"keep-alive ping failed for {model} — retrying next cycle")
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        for model in self.models:
            self._ping(model)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="forge-keepalive", daemon=True)
        self._thread.start()
        self._emit(f"keep-alive active for {len(self.models)} model(s), every {int(self.interval)}s")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self._client.close()
        except Exception:
            pass

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())


_daemon: KeepAliveDaemon | None = None


def start_keepalive(models: list[str], host: str | None = None, interval: float | None = None, on_event: Callable[[str], None] | None = None) -> KeepAliveDaemon:
    global _daemon
    if _daemon and _daemon.is_running():
        merged = list(dict.fromkeys(_daemon.models + list(models)))
        _daemon.models = merged
        return _daemon
    _daemon = KeepAliveDaemon(models, host=host, interval=interval, on_event=on_event)
    _daemon.start()
    return _daemon


def stop_keepalive() -> None:
    global _daemon
    if _daemon:
        _daemon.stop()
        _daemon = None
