from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from inferforge.core.config import registry_path


@dataclass
class ModelRecord:
    name: str
    source: str = "local"
    backend: str = "ollama"
    digest: str = ""
    size: int = 0
    format: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    context_length: int = 0
    path: str = ""
    ollama_name: str = ""
    capabilities: list[str] = field(default_factory=list)
    imported_at: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def display_size(self) -> str:
        if self.size <= 0:
            return "—"
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(self.size)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024
        return f"{self.size} B"


class Registry:
    def __init__(self, path: Path | None = None, seed_defaults: bool | None = None) -> None:
        self.path = path or registry_path()
        # Only seed the premade catalogue when creating the user's default registry
        # for the first time; explicit paths (tests, tooling) start empty.
        self._seed = seed_defaults if seed_defaults is not None else path is None
        self._models: dict[str, ModelRecord] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._models = {}
            if self._seed:
                self._seed_defaults()
                self.save()
            return
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        models = raw.get("models", {})
        self._models = {}
        for name, data in models.items():
            self._models[name] = ModelRecord(**data)

    def _seed_defaults(self) -> None:
        defaults = [
            ("advanced-merged", "forge", "llama", "4.5 GB", "Q4_K_M", "forge", ""),
            ("auto-optimized-merge", "forge", "llama", "4.8 GB", "Q4_K_M", "forge", ""),
            ("dolphin-mixtral:8x7b", "ollama", "llama", "24.6 GB", "Q4_0", "ollama", "dolphin-mixtral:8x7b"),
            ("embeddinggemma:latest", "ollama", "gemma3", "593.1 MB", "BF16", "ollama", "embeddinggemma:latest"),
            ("gemini-3-flash-preview:latest", "ollama", "", "367 B", "_", "ollama", "gemini-3-flash-preview:latest"),
            ("gemma2:9b", "ollama", "gemma2", "5.1 GB", "Q4_0", "ollama", "gemma2:9b"),
            ("glm-5.2:cloud", "ollama", "glm", "338 B", "_0", "ollama", "glm-5.2:cloud"),
            ("glm-5.3-flash:cloud", "ollama", "", "317 B", "FP8_K", "ollama", "glm-5.3-flash:cloud"),
            ("inferforge-beta", "forge", "inferforge", "5.6 GB", "Q4_K_M", "forge", "inferforge-beta"),
            ("inferforge-beta:latest", "ollama", "qwen2", "8.4 GB", "Q4_K_M", "ollama", "inferforge-beta:latest"),
            ("llama-gemma-slerp", "forge", "llama", "4.8 GB", "Q4_K_M", "forge", ""),
            ("llama-qwen-merged", "forge", "llama", "4.5 GB", "Q4_K_M", "forge", ""),
            ("llama3.1:8b", "ollama", "llama", "4.8 GB", "Q4_K_M", "ollama", "llama3.1:8b"),
            ("microsoft:Phi-3-mini-4k-instruct-gguf", "huggingface", "text-generation", "9.3 GB", "unknown", "huggingface", "microsoft/Phi-3-mini-4k-instruct-gguf"),
            ("my-custom-model", "forge", "llama", "—", "Q4_0", "forge", ""),
            ("optimized-model", "forge", "llama", "—", "Q4_0", "forge", ""),
            ("oroboroslabs/claude-fable-5Q:latest", "ollama", "qwen2", "5.4 GB", "Q2_K", "ollama", "oroboroslabs/claude-fable-5Q:latest"),
            ("prysmisai:latest", "ollama", "qwen2", "8.4 GB", "Q4_K_M", "ollama", "prysmisai:latest"),
            ("prysmis:latest", "ollama", "qwen2", "4.4 GB", "Q4_K_M", "ollama", "prysmis:latest"),
            ("PrysmisAI", "ollama", "qwen2", "—", "Q4_K_M", "ollama", "PrysmisAI"),
            ("prysmisai-fast:latest", "ollama", "qwen2", "—", "Q4_K_M", "ollama", "prysmisai-fast:latest"),
            ("prysmisai-ft:latest", "ollama", "qwen2", "—", "Q4_K_M", "ollama", "prysmisai-ft:latest"),
        ]
        for name, source, family, size_str, quant, fmt, ollama_name in defaults:
            if name in self._models:
                continue
            size = 0
            if size_str not in ("—", ""):
                try:
                    num, unit = size_str.split()
                    mult = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit]
                    size = int(float(num) * mult)
                except Exception:
                    pass
            flags = ""
            if name == "inferforge-beta":
                flags = "beta, agent"
            self._models[name] = ModelRecord(
                name=name,
                source=source,
                backend="ollama" if source == "ollama" else "huggingface" if source == "huggingface" else "native",
                digest="",
                size=size,
                format=fmt,
                family=family,
                parameter_size=size_str,
                quantization=quant,
                context_length=4096,
                path="",
                ollama_name=ollama_name,
                capabilities=[],
                imported_at=time.time(),
                meta={"premade": True, "flags": flags},
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "models": {name: asdict(m) for name, m in sorted(self._models.items())},
        }
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def upsert(self, record: ModelRecord) -> None:
        if not record.imported_at:
            record.imported_at = time.time()
        self._models[record.name] = record
        self.save()

    def upsert_many(self, records: Iterable[ModelRecord]) -> int:
        count = 0
        now = time.time()
        for record in records:
            if not record.imported_at:
                record.imported_at = now
            self._models[record.name] = record
            count += 1
        self.save()
        return count

    def get(self, name: str) -> ModelRecord | None:
        if name in self._models:
            return self._models[name]
        if ":" not in name:
            latest = self._models.get(f"{name}:latest")
            if latest:
                return latest
            for key, rec in self._models.items():
                if key.split(":", 1)[0] == name:
                    return rec
        for rec in self._models.values():
            if rec.digest and (rec.digest == name or rec.digest.startswith(name)):
                return rec
        return None

    def remove(self, name: str) -> bool:
        if name in self._models:
            del self._models[name]
            self.save()
            return True
        return False

    def list(self) -> list[ModelRecord]:
        return sorted(self._models.values(), key=lambda m: m.name.lower())

    def names(self) -> list[str]:
        return sorted(self._models.keys(), key=str.lower)

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None
