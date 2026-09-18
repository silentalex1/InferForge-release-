from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "InferForge"
APP_AUTHOR = "InferForge"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11435
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def data_dir() -> Path:
    path = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir() -> Path:
    path = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def registry_path() -> Path:
    return data_dir() / "registry.json"


def settings_path() -> Path:
    return config_dir() / "settings.json"


def ollama_models_dir() -> Path:
    override = os.environ.get("OLLAMA_MODELS")
    if override:
        return Path(override)
    home = Path.home()
    if platform.system() == "Darwin":
        candidates = [home / ".ollama" / "models", home / ".ollama" / "models"]
    elif platform.system() == "Windows":
        candidates = [home / ".ollama" / "models"]
    else:
        candidates = [home / ".ollama" / "models", Path("/usr/share/ollama/.ollama/models")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return home / ".ollama" / "models"


def load_settings() -> dict[str, Any]:
    path = settings_path()
    if not path.exists():
        defaults = {
            "host": DEFAULT_HOST,
            "port": DEFAULT_PORT,
            "ollama_host": OLLAMA_HOST,
            "animation": True,
            "backend": "auto",
            "theme": "forge",
            "storage_enabled": False,
            "storage_type": "local",
            "storage_endpoint": "",
            "storage_access_key": "",
            "storage_secret_key": "",
            "storage_bucket": "",
            "storage_region": "",
            "storage_max_size_tb": 20.0,
            "storage_chunk_size_mb": 100,
            "remote_enabled": False,
            "remote_endpoint": "https://api.inferforge.com",
            "remote_api_key": "",
            "remote_timeout": 600.0,
            "prefer_remote": False,
            "n_ctx": 2048,
            "n_gpu_layers": -1,
            "verbose": False,
            "cache_models": False,
            "cache_dir": str(data_dir() / "cache"),
            "training_enabled": True,
            "training_max_examples": 64,
            "training_save_path": str(data_dir() / "trained_models"),
            "default_model": "inferforge-beta",
            "always_on_models": [],
            "keep_alive_interval": 240,
            "embed_keys": {},
        }
        save_settings(defaults)
        return defaults
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_settings(settings: dict[str, Any]) -> None:
    path = settings_path()
    with path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def cache_dir() -> Path:
    settings = load_settings()
    cache_path = Path(settings.get("cache_dir", str(data_dir() / "cache")))
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_storage_config() -> dict[str, Any]:
    settings = load_settings()
    return {
        "enabled": settings.get("storage_enabled", False),
        "type": settings.get("storage_type", "s3"),
        "endpoint": settings.get("storage_endpoint", ""),
        "access_key": settings.get("storage_access_key", ""),
        "secret_key": settings.get("storage_secret_key", ""),
        "bucket": settings.get("storage_bucket", "inferforge-models"),
        "region": settings.get("storage_region", "us-east-1"),
        "max_size_tb": settings.get("storage_max_size_tb", 10.0),
        "chunk_size_mb": settings.get("storage_chunk_size_mb", 100),
    }


def get_remote_config() -> dict[str, Any]:
    settings = load_settings()
    return {
        "enabled": settings.get("remote_enabled", False),
        "endpoint": settings.get("remote_endpoint", ""),
        "api_key": settings.get("remote_api_key", ""),
        "timeout": settings.get("remote_timeout", 600.0),
        "prefer_remote": settings.get("prefer_remote", False),
    }


def get_training_config() -> dict[str, Any]:
    settings = load_settings()
    return {
        "enabled": settings.get("training_enabled", True),
        "max_examples": settings.get("training_max_examples", 50),
        "save_path": settings.get("training_save_path", str(data_dir() / "trained_models")),
    }


def trained_models_dir() -> Path:
    settings = load_settings()
    path = Path(settings.get("training_save_path", str(data_dir() / "trained_models")))
    path.mkdir(parents=True, exist_ok=True)
    return path
