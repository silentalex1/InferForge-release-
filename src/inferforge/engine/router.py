from __future__ import annotations

from pathlib import Path
from typing import Any

import psutil

from inferforge.core.config import load_settings
from inferforge.core.registry import ModelRecord
from inferforge.engine.base import ChatEngine
from inferforge.engine.native_backend import NativeEngine
from inferforge.engine.ollama_backend import OllamaEngine
from inferforge.remote.http_backend import HTTPRemoteBackend


class ExecutionRouter:
    def __init__(self) -> None:
        self.settings = load_settings()
        self._cache: dict[str, ChatEngine] = {}

    def resolve(self, model: ModelRecord) -> ChatEngine:
        if model.name in self._cache:
            return self._cache[model.name]

        execution_mode = self._determine_execution_mode(model)
        engine = self._create_engine(model, execution_mode)
        self._cache[model.name] = engine
        return engine

    def _has_local_weights(self, model: ModelRecord) -> bool:
        if not model.path:
            return False
        path = Path(model.path)
        if path.is_file():
            return True
        if path.is_dir():
            return any(
                (path / name).exists()
                for name in ("model.gguf", f"{model.name}.gguf", "Modelfile")
            ) and (
                (path / "model.gguf").exists()
                or (path / f"{model.name}.gguf").exists()
                or any(path.glob("*.gguf"))
            )
        return False

    def _has_hf_weights(self, model: ModelRecord) -> bool:
        """Local HF-format weights (config.json + safetensors/bin), e.g. from `forge train`."""
        if not model.path:
            return False
        path = Path(model.path)
        if not path.is_dir() or not (path / "config.json").exists():
            return False
        return (path / "model.safetensors").exists() or any(path.glob("*.safetensors")) or any(
            path.glob("pytorch_model*.bin")
        )

    def _determine_execution_mode(self, model: ModelRecord) -> str:
        preferred = model.backend or self.settings.get("backend", "auto")

        # Locally fine-tuned HF models (forge train --finetune/--agent) run via transformers.
        if self._has_hf_weights(model) or preferred == "huggingface":
            return "huggingface"

        # Own forge models (Modelfile-derived) always prefer ollama under their own tag
        if model.source == "forge" and model.meta.get("own_model"):
            return "ollama"

        if preferred != "auto":
            if preferred == "native" and not self._has_local_weights(model):
                return "ollama" if model.ollama_name or model.source in {"ollama", "forge"} else "native"
            return preferred

        if self._has_local_weights(model) and model.source != "forge":
            model_size_gb = model.size / (1024**3) if model.size else 0
            available_ram_gb = psutil.virtual_memory().available / (1024**3)

            if model_size_gb > 0 and model_size_gb > available_ram_gb * 0.85:
                return "remote" if self.settings.get("prefer_remote", False) else "native"
            return "native"

        if self.settings.get("prefer_remote", False):
            return "remote"

        return "ollama"

    def _check_gpu_available(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _try_native(self, model: ModelRecord) -> ChatEngine | None:
        if not self._has_local_weights(model):
            return None
        try:
            return NativeEngine(
                model,
                n_ctx=model.context_length or self.settings.get("n_ctx", 2048),
                n_gpu_layers=self.settings.get("n_gpu_layers", -1),
                verbose=self.settings.get("verbose", False),
            )
        except (RuntimeError, ValueError, FileNotFoundError):
            return None

    def _create_engine(self, model: ModelRecord, mode: str) -> ChatEngine:
        if mode == "native":
            engine = self._try_native(model)
            if engine is not None:
                return engine

        if mode == "huggingface":
            try:
                from inferforge.engine.huggingface_backend import HuggingFaceEngine

                return HuggingFaceEngine(model)
            except (ImportError, RuntimeError) as e:
                if model.ollama_name:
                    return OllamaEngine(model)
                raise RuntimeError(f"Cannot load {model.name} via HuggingFace: {e}") from e

        if mode == "remote":
            remote = HTTPRemoteBackend(
                endpoint=self.settings.get("remote_endpoint", ""),
                api_key=self.settings.get("remote_api_key", ""),
                model_name=model.name,
                timeout=self.settings.get("remote_timeout", 600.0),
            )
            remote.initialize()
            return remote

        # Ollama path — forge own-models run under THEIR name (inferforge-beta), not the base tag
        if model.source == "forge":
            if model.meta.get("own_model") or model.ollama_name:
                # Prefer the derived Ollama model tag
                run_as = ModelRecord(
                    name=model.name,
                    source="forge",
                    backend="ollama",
                    family=model.family,
                    parameter_size=model.parameter_size,
                    quantization=model.quantization,
                    format=model.format,
                    context_length=model.context_length,
                    ollama_name=model.ollama_name or model.name,
                    meta=model.meta,
                )
                try:
                    return OllamaEngine(run_as)
                except RuntimeError as e:
                    if "Cannot connect to Ollama" in str(e):
                        return self._hyperneural_fallback(model)
                    # Fall back to base only if derived tag missing
                    base_model_name = model.meta.get("base_model")
                    if base_model_name and base_model_name != model.name:
                        base_record = ModelRecord(
                            name=base_model_name,
                            source="ollama",
                            backend="ollama",
                            family=model.family,
                            parameter_size=model.parameter_size,
                            quantization=model.quantization,
                            format=model.format,
                            context_length=model.context_length,
                            ollama_name=base_model_name,
                        )
                        try:
                            return OllamaEngine(base_record)
                        except RuntimeError as e2:
                            if "Cannot connect to Ollama" in str(e2):
                                return self._hyperneural_fallback(model)
                            raise
                    raise

        try:
            return OllamaEngine(model)
        except RuntimeError as e:
            if "Cannot connect to Ollama" in str(e):
                return self._hyperneural_fallback(model)
            raise

    def _hyperneural_fallback(self, model: ModelRecord) -> ChatEngine:
        try:
            return HTTPRemoteBackend(
                endpoint="https://hyperneural.cfd",
                model_name=model.name,
                timeout=30.0,
            )
        except Exception:
            pass
        # Final fallback — still try hyperneural via direct httpx in ChatEngine wrapper
        from inferforge.engine.hyperneural_backend import HyperNeuralEngine
        return HyperNeuralEngine(model)

    def clear_cache(self) -> None:
        for engine in self._cache.values():
            try:
                engine.close()
            except Exception:
                pass
        self._cache.clear()

    def get_execution_info(self, model: ModelRecord) -> dict[str, Any]:
        mode = self._determine_execution_mode(model)
        return {
            "model": model.name,
            "mode": mode,
            "local_weights": self._has_local_weights(model),
            "size_gb": model.size / (1024**3) if model.size else 0,
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
            "gpu_available": self._check_gpu_available(),
        }


_router_instance: ExecutionRouter | None = None


def get_router() -> ExecutionRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = ExecutionRouter()
    return _router_instance


def reset_router() -> None:
    global _router_instance
    if _router_instance:
        _router_instance.clear_cache()
    _router_instance = None
