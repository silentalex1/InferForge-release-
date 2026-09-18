from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import httpx

from inferforge.core.config import load_settings, ollama_models_dir
from inferforge.core.registry import ModelRecord, Registry

ProgressCb = Callable[[str, int, int], None]


def _blob_path(digest: str, blobs_dir: Path) -> Path | None:
    hexpart = digest.split(":", 1)[-1]
    candidate = blobs_dir / f"sha256-{hexpart}"
    if candidate.exists():
        return candidate
    alt = blobs_dir / hexpart
    return alt if alt.exists() else None


def _model_blob_from_manifest(name: str, manifests_root: Path, blobs_dir: Path) -> tuple[str, Path | None]:
    if ":" in name:
        repo, tag = name.rsplit(":", 1)
    else:
        repo, tag = name, "latest"

    prefixes = [
        manifests_root / "registry.ollama.ai" / "library",
        manifests_root / "registry.ollama.ai",
    ]
    manifest_file: Path | None = None

    for base in prefixes:
        simple = repo.split("/")[-1] if repo.startswith("library/") is False and "/" not in repo else repo
        candidates = [
            base / repo / tag,
            base / simple / tag,
            manifests_root / "registry.ollama.ai" / "library" / simple / tag,
            manifests_root / "registry.ollama.ai" / repo / tag,
        ]
        for c in candidates:
            if c.is_file():
                manifest_file = c
                break
        if manifest_file:
            break

    if not manifest_file:
        return "", None

    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", None

    digest = ""
    path: Path | None = None
    for layer in data.get("layers") or []:
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer.get("digest") or ""
            path = _blob_path(digest, blobs_dir) if digest else None
            break
    return digest, path


def fetch_ollama_models(host: str | None = None) -> list[dict]:
    settings = load_settings()
    base = (host or settings.get("ollama_host") or "http://127.0.0.1:11434").rstrip("/")
    with httpx.Client(base_url=base, timeout=30.0) as client:
        r = client.get("/api/tags")
        r.raise_for_status()
        return list((r.json() or {}).get("models") or [])


def import_from_ollama(
    registry: Registry | None = None,
    host: str | None = None,
    progress: ProgressCb | None = None,
    link_blobs: bool = True,
) -> tuple[int, list[str]]:
    reg = registry or Registry()
    models = fetch_ollama_models(host=host)
    total = len(models)
    imported: list[str] = []

    models_root = ollama_models_dir()
    manifests = models_root / "manifests"
    blobs = models_root / "blobs"

    for i, raw in enumerate(models, start=1):
        name = raw.get("name") or raw.get("model") or ""
        if not name:
            continue
        if progress:
            progress(name, i, total)

        details = raw.get("details") or {}
        digest = raw.get("digest") or ""
        size = int(raw.get("size") or 0)
        blob_digest, blob_path = ("", None)
        if manifests.exists() and blobs.exists():
            blob_digest, blob_path = _model_blob_from_manifest(name, manifests, blobs)

        has_local_blob = bool(blob_path and link_blobs)
        record = ModelRecord(
            name=name,
            source="ollama",
            backend="native" if has_local_blob else "ollama",
            digest=digest or blob_digest,
            size=size,
            format=(details.get("format") or "").lower() or ("gguf" if has_local_blob else ""),
            family=details.get("family") or "",
            parameter_size=details.get("parameter_size") or "",
            quantization=details.get("quantization_level") or "",
            context_length=int(details.get("context_length") or 0),
            path=str(blob_path) if has_local_blob else "",
            ollama_name=name,
            capabilities=list(raw.get("capabilities") or []),
            meta={
                "parent_model": details.get("parent_model") or "",
                "remote_model": raw.get("remote_model") or "",
                "remote_host": raw.get("remote_host") or "",
                "modified_at": raw.get("modified_at") or "",
                "direct_blob": has_local_blob,
            },
        )
        reg.upsert(record)
        imported.append(name)

    return len(imported), imported


def stream_ollama_pull(
    model_name: str,
    host: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> bool:
    """Stream model pull from Ollama daemon via REST API."""
    settings = load_settings()
    base = (host or settings.get("ollama_host") or "http://127.0.0.1:11434").rstrip("/")
    url = f"{base}/api/pull"
    try:
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", url, json={"name": model_name}) as response:
                if response.status_code != 200:
                    return False
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if progress_callback:
                            progress_callback(data)
                    except Exception:
                        pass
        return True
    except Exception:
        return False
