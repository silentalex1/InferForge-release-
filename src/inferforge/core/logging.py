from __future__ import annotations

import logging
import os
from pathlib import Path

_configured = False


def _log_dir() -> Path:
    from inferforge.core.config import data_dir

    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(verbose: bool = False) -> None:
    global _configured
    level_name = os.environ.get("INFERFORGE_LOG_LEVEL", "DEBUG" if verbose else "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger("inferforge")
    root.setLevel(level)
    root.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(_log_dir() / "inferforge.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    if verbose or os.environ.get("INFERFORGE_DEBUG"):
        stream = logging.StreamHandler()
        stream.setLevel(level)
        stream.setFormatter(formatter)
        root.addHandler(stream)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure_logging(verbose=bool(os.environ.get("INFERFORGE_DEBUG")))
    return logging.getLogger(f"inferforge.{name}")
