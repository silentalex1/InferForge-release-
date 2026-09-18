"""Unified inference router with intelligent backend selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import psutil

from inferforge.core.config import load_settings
from inferforge.core.registry import ModelRecord
from inferforge.engine.base import ChatEngine, ChatMessage, GenerationConfig
from inferforge.engine.huggingface_backend import HuggingFaceEngine
from inferforge.engine.native_backend import NativeEngine
from inferforge.engine.ollama_backend import OllamaEngine
from inferforge.remote.http_backend import HTTPRemoteBackend

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available backend types."""
    NATIVE = "native"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    REMOTE = "remote"
    AUTO = "auto"


@dataclass
class BackendCapabilities:
    """Capabilities and requirements of a backend."""
    name: BackendType
    available: bool
    requires_local_weights: bool
    supports_streaming: bool
    supports_gpu: bool
    memory_efficient: bool
    startup_time: str  # "fast", "medium", "slow"
    quality: str  # "high", "medium", "low"


class UnifiedRouter:
    """Intelligent router that selects optimal backend for each model."""
    
    def __init__(self):
        self.settings = load_settings()
        self._cache: dict[str, ChatEngine] = {}
        self._backend_health: dict[BackendType, bool] = {}
        self._initialize_backends()
    
    def _initialize_backends(self) -> None:
        """Check availability of each backend."""
        # Check Ollama
        try:
            import httpx
            response = httpx.get(
                self.settings.get("ollama_host", "http://127.0.0.1:11434") + "/api/tags",
                timeout=2.0,
            )
            self._backend_health[BackendType.OLLAMA] = response.status_code == 200
        except Exception:
            self._backend_health[BackendType.OLLAMA] = False
        
        # Check HuggingFace
        try:
            import torch
            import transformers
            self._backend_health[BackendType.HUGGINGFACE] = True
        except ImportError:
            self._backend_health[BackendType.HUGGINGFACE] = False
        
        # Check Native (llama-cpp-python)
        try:
            import llama_cpp
            self._backend_health[BackendType.NATIVE] = True
        except ImportError:
            self._backend_health[BackendType.NATIVE] = False
        
        # Check Remote
        remote_endpoint = self.settings.get("remote_endpoint")
        self._backend_health[BackendType.REMOTE] = bool(remote_endpoint)
    
    def get_backend_capabilities(self) -> list[BackendCapabilities]:
        """Get capabilities of all backends."""
        return [
            BackendCapabilities(
                name=BackendType.NATIVE,
                available=self._backend_health.get(BackendType.NATIVE, False),
                requires_local_weights=True,
                supports_streaming=True,
                supports_gpu=True,
                memory_efficient=True,
                startup_time="fast",
                quality="high",
            ),
            BackendCapabilities(
                name=BackendType.OLLAMA,
                available=self._backend_health.get(BackendType.OLLAMA, False),
                requires_local_weights=False,  # Ollama manages weights
                supports_streaming=True,
                supports_gpu=True,
                memory_efficient=True,
                startup_time="fast",
                quality="high",
            ),
            BackendCapabilities(
                name=BackendType.HUGGINGFACE,
                available=self._backend_health.get(BackendType.HUGGINGFACE, False),
                requires_local_weights=True,
                supports_streaming=True,
                supports_gpu=True,
                memory_efficient=False,  # Can be memory-intensive
                startup_time="slow",  # Model loading is slow
                quality="high",
            ),
            BackendCapabilities(
                name=BackendType.REMOTE,
                available=self._backend_health.get(BackendType.REMOTE, False),
                requires_local_weights=False,
                supports_streaming=True,
                supports_gpu=False,  # Handled remotely
                memory_efficient=True,
                startup_time="fast",
                quality="high",
            ),
        ]
    
    def select_backend(
        self,
        model: ModelRecord,
        prefer_backend: BackendType | None = None,
    ) -> BackendType:
        """Intelligently select the best backend for a model.
        
        Selection criteria:
        1. User preference (if specified and available)
        2. Model source and format
        3. System resources
        4. Backend availability
        """
        # Honor user preference if valid
        if prefer_backend and prefer_backend != BackendType.AUTO:
            if self._backend_health.get(prefer_backend, False):
                if self._is_backend_suitable(model, prefer_backend):
                    return prefer_backend
        
        # Check explicit backend specification
        if model.backend:
            try:
                specified = BackendType(model.backend)
                if self._backend_health.get(specified, False):
                    return specified
            except ValueError:
                pass
        
        # InferForge-trained models prefer Ollama
        if model.source == "forge" and model.meta.get("own_model"):
            if self._backend_health.get(BackendType.OLLAMA, False):
                return BackendType.OLLAMA
        
        # Models from Ollama library prefer Ollama
        if model.source == "ollama" and self._backend_health.get(BackendType.OLLAMA, False):
            return BackendType.OLLAMA
        
        # Check if we have local weights
        has_local_weights = self._has_local_weights(model)
        
        # HuggingFace models prefer HuggingFace backend
        if model.meta.get("hf_model_id") and self._backend_health.get(BackendType.HUGGINGFACE, False):
            return BackendType.HUGGINGFACE
        
        # If we have local GGUF weights, prefer native for efficiency
        if has_local_weights and model.format == "gguf":
            if self._backend_health.get(BackendType.NATIVE, False):
                # Check if we have enough memory
                if self._can_fit_in_memory(model):
                    return BackendType.NATIVE
        
        # Try Ollama as fallback
        if self._backend_health.get(BackendType.OLLAMA, False):
            return BackendType.OLLAMA
        
        # Try HuggingFace with remote download
        if self._backend_health.get(BackendType.HUGGINGFACE, False):
            return BackendType.HUGGINGFACE
        
        # Last resort: remote
        if self._backend_health.get(BackendType.REMOTE, False):
            return BackendType.REMOTE
        
        # No backend available
        raise RuntimeError(
            "No suitable backend available. Install at least one of:\n"
            "- Ollama (ollama.com)\n"
            "- llama-cpp-python (pip install llama-cpp-python)\n"
            "- transformers (pip install transformers torch)"
        )
    
    def _is_backend_suitable(self, model: ModelRecord, backend: BackendType) -> bool:
        """Check if a backend is suitable for a model."""
        if backend == BackendType.NATIVE:
            return self._has_local_weights(model) and model.format in {"gguf", "ggml"}
        elif backend == BackendType.OLLAMA:
            return True  # Ollama can handle most models
        elif backend == BackendType.HUGGINGFACE:
            return True  # HF can download and load most models
        elif backend == BackendType.REMOTE:
            return True  # Remote can handle anything
        return False
    
    def _has_local_weights(self, model: ModelRecord) -> bool:
        """Check if model has local weight files."""
        if not model.path:
            return False
        
        path = Path(model.path)
        if path.is_file():
            return path.suffix in {".gguf", ".ggml", ".bin", ".safetensors"}
        
        if path.is_dir():
            # Check for GGUF files
            if any(path.glob("*.gguf")) or any(path.glob("*.ggml")):
                return True
            # Check for HF format
            if (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists():
                return True
        
        return False
    
    def _can_fit_in_memory(self, model: ModelRecord) -> bool:
        """Check if model can fit in available memory."""
        if not model.size:
            return True  # Unknown size, assume it fits
        
        model_size_gb = model.size / (1024 ** 3)
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        
        # Need at least 2x model size for loading + inference
        return available_ram_gb > (model_size_gb * 2)
    
    def get_engine(
        self,
        model: ModelRecord,
        backend: BackendType | None = None,
    ) -> ChatEngine:
        """Get or create engine for model."""
        cache_key = f"{model.name}:{backend.value if backend else 'auto'}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Select backend
        selected_backend = self.select_backend(model, backend)
        logger.info(f"Selected backend {selected_backend.value} for model {model.name}")
        
        # Create engine
        engine = self._create_engine(model, selected_backend)
        self._cache[cache_key] = engine
        
        return engine
    
    def _create_engine(self, model: ModelRecord, backend: BackendType) -> ChatEngine:
        """Create engine instance for specified backend."""
        if backend == BackendType.NATIVE:
            return NativeEngine(
                model,
                n_ctx=model.context_length or self.settings.get("n_ctx", 2048),
                n_gpu_layers=self.settings.get("n_gpu_layers", -1),
                verbose=self.settings.get("verbose", False),
            )
        
        elif backend == BackendType.OLLAMA:
            return OllamaEngine(model)
        
        elif backend == BackendType.HUGGINGFACE:
            return HuggingFaceEngine(
                model,
                device=self.settings.get("device", "auto"),
                load_in_8bit=self.settings.get("load_in_8bit", False),
                load_in_4bit=self.settings.get("load_in_4bit", False),
                use_flash_attention=self.settings.get("use_flash_attention", False),
            )
        
        elif backend == BackendType.REMOTE:
            remote = HTTPRemoteBackend(
                endpoint=self.settings.get("remote_endpoint", ""),
                api_key=self.settings.get("remote_api_key", ""),
                model_name=model.name,
                timeout=self.settings.get("remote_timeout", 600.0),
            )
            remote.initialize()
            return remote
        
        raise ValueError(f"Unknown backend: {backend}")
    
    def generate(
        self,
        model: ModelRecord,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        backend: BackendType | None = None,
    ) -> str:
        """Generate response using optimal backend."""
        engine = self.get_engine(model, backend)
        return engine.generate(messages, config)
    
    def stream(
        self,
        model: ModelRecord,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        backend: BackendType | None = None,
    ) -> Iterator[str]:
        """Stream response using optimal backend."""
        engine = self.get_engine(model, backend)
        return engine.stream(messages, config)
    
    def clear_cache(self) -> None:
        """Clear all cached engines."""
        for engine in self._cache.values():
            try:
                engine.close()
            except Exception as e:
                logger.warning(f"Error closing engine: {e}")
        self._cache.clear()
    
    def get_status(self) -> dict[str, Any]:
        """Get router status and backend health."""
        return {
            "backends": {
                backend.value: healthy
                for backend, healthy in self._backend_health.items()
            },
            "cached_engines": len(self._cache),
            "capabilities": [
                {
                    "name": cap.name.value,
                    "available": cap.available,
                    "quality": cap.quality,
                    "startup": cap.startup_time,
                }
                for cap in self.get_backend_capabilities()
            ],
        }


# Global router instance
_unified_router: UnifiedRouter | None = None


def get_unified_router() -> UnifiedRouter:
    """Get global unified router instance."""
    global _unified_router
    if _unified_router is None:
        _unified_router = UnifiedRouter()
    return _unified_router


def reset_unified_router() -> None:
    """Reset global router instance."""
    global _unified_router
    if _unified_router:
        _unified_router.clear_cache()
    _unified_router = None
