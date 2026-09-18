from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

ProgressFn = Callable[[str, int, int], None]

GGUF_TO_HF = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
    "rope_freqs.weight": None,
}

GGUF_LAYER = [
    (re.compile(r"^blk\.(\d+)\.attn_q\.weight$"), "model.layers.{}.self_attn.q_proj.weight"),
    (re.compile(r"^blk\.(\d+)\.attn_k\.weight$"), "model.layers.{}.self_attn.k_proj.weight"),
    (re.compile(r"^blk\.(\d+)\.attn_v\.weight$"), "model.layers.{}.self_attn.v_proj.weight"),
    (re.compile(r"^blk\.(\d+)\.attn_output\.weight$"), "model.layers.{}.self_attn.o_proj.weight"),
    (re.compile(r"^blk\.(\d+)\.ffn_gate\.weight$"), "model.layers.{}.mlp.gate_proj.weight"),
    (re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"), "model.layers.{}.mlp.up_proj.weight"),
    (re.compile(r"^blk\.(\d+)\.ffn_down\.weight$"), "model.layers.{}.mlp.down_proj.weight"),
    (re.compile(r"^blk\.(\d+)\.attn_norm\.weight$"), "model.layers.{}.input_layernorm.weight"),
    (re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"), "model.layers.{}.post_attention_layernorm.weight"),
    (re.compile(r"^blk\.(\d+)\.attn_q\.bias$"), "model.layers.{}.self_attn.q_proj.bias"),
    (re.compile(r"^blk\.(\d+)\.attn_k\.bias$"), "model.layers.{}.self_attn.k_proj.bias"),
    (re.compile(r"^blk\.(\d+)\.attn_v\.bias$"), "model.layers.{}.self_attn.v_proj.bias"),
    (re.compile(r"^blk\.(\d+)\.ffn_gate\.bias$"), "model.layers.{}.mlp.gate_proj.bias"),
    (re.compile(r"^blk\.(\d+)\.ffn_up\.bias$"), "model.layers.{}.mlp.up_proj.bias"),
    (re.compile(r"^blk\.(\d+)\.ffn_down\.bias$"), "model.layers.{}.mlp.down_proj.bias"),
    (re.compile(r"^blk\.(\d+)\.attn_q_norm\.weight$"), "model.layers.{}.self_attn.q_norm.weight"),
    (re.compile(r"^blk\.(\d+)\.attn_k_norm\.weight$"), "model.layers.{}.self_attn.k_norm.weight"),
]


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for model merging. Install with: pip install 'inferforge[merging]'"
        ) from exc


def _is_gguf_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def _is_safetensors_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            if header_len <= 0 or header_len > 100_000_000:
                return False
            return handle.read(min(header_len, 8)).startswith(b"{")
    except OSError:
        return False


def detect_format(path: str | Path) -> str:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Model path not found: {target}")
    if target.is_dir():
        if list(target.glob("*.safetensors")):
            return "safetensors"
        if (target / "pytorch_model.bin").exists() or list(target.glob("*.bin")):
            return "pytorch"
        if list(target.glob("*.gguf")):
            return "gguf"
        if (target / "config.json").exists():
            return "huggingface"
        raise ValueError(f"No supported weight files in directory: {target}")
    suffix = target.suffix.lower()
    if suffix == ".safetensors":
        return "safetensors"
    if suffix in {".bin", ".pt", ".pth"}:
        return "pytorch"
    if suffix == ".gguf" or _is_gguf_file(target):
        return "gguf"
    if _is_safetensors_file(target):
        return "safetensors"
    raise ValueError(f"Unsupported model format: {target}")


def map_gguf_key(name: str) -> str | None:
    if name in GGUF_TO_HF:
        return GGUF_TO_HF[name]
    for pattern, template in GGUF_LAYER:
        match = pattern.match(name)
        if match:
            return template.format(match.group(1))
    return name


def detect_model_architecture(weights: dict[str, Any]) -> str:
    keys = list(weights.keys())
    joined = " ".join(keys)
    if any("pre_feedforward_layernorm" in key for key in keys):
        return "gemma2"
    if any("q_norm" in key or "k_norm" in key for key in keys) and "q_proj" in joined:
        return "qwen2"
    if "blk." in joined and "attn_q" in joined:
        return "llama"
    if "transformer.wte" in joined or "transformer.h." in joined:
        return "gpt2"
    if "model.decoder.layers" in joined:
        return "opt"
    if "model.layers" in joined and "mlp.gate_proj" in joined:
        if any("block_sparse_moe" in key for key in keys):
            return "mixtral"
        return "llama"
    if "model.layers" in joined and "self_attn.q_proj" in joined:
        return "llama"
    return "llama"


def _embed_shape(weights: dict[str, Any]) -> tuple[int, int]:
    for key in (
        "model.embed_tokens.weight",
        "token_embd.weight",
        "transformer.wte.weight",
        "embed_tokens.weight",
        "lm_head.weight",
        "output.weight",
    ):
        tensor = weights.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[0]), int(tensor.shape[1])
    for tensor in weights.values():
        if hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[0]), int(tensor.shape[1])
    return 32000, 4096


def _layer_count(weights: dict[str, Any]) -> int:
    indices: set[int] = set()
    for key in weights:
        match = re.search(r"(?:layers|blk|h)\.(\d+)", key)
        if match:
            indices.add(int(match.group(1)))
    return (max(indices) + 1) if indices else 1


def _head_dim(hidden: int) -> int:
    for candidate in (128, 80, 64, 96, 256):
        if hidden % candidate == 0:
            return candidate
    return 64


def build_hf_config(weights: dict[str, Any], architecture: str) -> dict[str, Any]:
    vocab, hidden = _embed_shape(weights)
    layers = _layer_count(weights)
    head_dim = _head_dim(hidden)
    num_heads = max(1, hidden // head_dim)
    kv_heads = num_heads
    intermediate_size = hidden * 4

    for key, tensor in weights.items():
        if "k_proj" in key and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            kv_out = int(tensor.shape[0])
            if kv_out % head_dim == 0:
                kv_heads = max(1, kv_out // head_dim)
        if "q_proj" in key and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            q_out = int(tensor.shape[0])
            if q_out % head_dim == 0:
                num_heads = max(1, q_out // head_dim)
        if ("gate_proj" in key or "up_proj" in key) and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            intermediate_size = int(tensor.shape[0])

    # Check for Qwen2 specific keys
    if any("q_norm" in k or "k_norm" in k for k in weights):
        architecture = "qwen2"
    elif any("pre_feedforward_layernorm" in k for k in weights):
        architecture = "gemma2"

    arch_map = {
        "llama": ("LlamaForCausalLM", "llama"),
        "mistral": ("MistralForCausalLM", "mistral"),
        "mixtral": ("MixtralForCausalLM", "mixtral"),
        "qwen2": ("Qwen2ForCausalLM", "qwen2"),
        "gemma2": ("Gemma2ForCausalLM", "gemma2"),
        "gpt2": ("GPT2LMHeadModel", "gpt2"),
        "opt": ("OPTForCausalLM", "opt"),
    }
    hf_arch, model_type = arch_map.get(architecture, ("LlamaForCausalLM", "llama"))
    return {
        "architectures": [hf_arch],
        "model_type": model_type,
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": kv_heads,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": "lm_head.weight" not in weights and "output.weight" not in weights,
    }


def load_safetensors_weights(path: str | Path, progress: ProgressFn | None = None) -> dict[str, Any]:
    torch = _require_torch()
    target = Path(path)
    files: list[Path]
    if target.is_dir():
        files = sorted(target.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors files in {target}")
    else:
        files = [target]
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required. Install with: pip install safetensors") from exc
    weights: dict[str, Any] = {}
    total = len(files)
    for index, file_path in enumerate(files, start=1):
        if progress:
            progress(f"Loading {file_path.name}", index - 1, total)
        shard = load_file(str(file_path), device="cpu")
        for key, tensor in shard.items():
            weights[key] = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
        del shard
    if progress:
        progress("Loaded safetensors", total, total)
    return weights


def load_pytorch_weights(path: str | Path, progress: ProgressFn | None = None) -> dict[str, Any]:
    torch = _require_torch()
    target = Path(path)
    files: list[Path] = []
    if target.is_dir():
        index_file = target / "pytorch_model.bin.index.json"
        if index_file.exists():
            index = json.loads(index_file.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") or {}
            files = [target / name for name in sorted(set(weight_map.values()))]
        else:
            files = sorted(target.glob("*.bin")) + sorted(target.glob("*.pt")) + sorted(target.glob("*.pth"))
        if not files:
            raise FileNotFoundError(f"No PyTorch weight files in {target}")
    else:
        files = [target]
    weights: dict[str, Any] = {}
    total = len(files)
    for index, file_path in enumerate(files, start=1):
        if progress:
            progress(f"Loading {file_path.name}", index - 1, total)
        payload = torch.load(str(file_path), map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected checkpoint structure in {file_path}")
        for key, tensor in payload.items():
            if hasattr(tensor, "shape"):
                weights[key] = tensor
        del payload
    if progress:
        progress("Loaded pytorch", total, total)
    return weights


def load_gguf_weights(path: str | Path, progress: ProgressFn | None = None) -> dict[str, Any]:
    torch = _require_torch()
    errors: list[str] = []
    arrays = None
    try:
        from inferforge.merger.core.gguf_reader import load_gguf_native

        arrays, _arch, _tok = load_gguf_native(path, progress=progress)
    except Exception as exc:
        errors.append(f"native: {exc}")
        arrays = None
    if arrays is None:
        try:
            import numpy as np
            from gguf import GGUFReader

            reader = GGUFReader(str(path))
            arrays = {}
            tensors = list(reader.tensors)
            total = max(len(tensors), 1)
            for index, tensor in enumerate(tensors, start=1):
                if progress:
                    progress(f"Loading {tensor.name}", index - 1, total)
                arrays[tensor.name] = np.array(tensor.data, copy=False)
        except Exception as exc:
            errors.append(f"gguf-package: {exc}")
    if arrays is None:
        raise RuntimeError(
            "Failed to load GGUF weights from "
            f"{path}. Tried built-in reader and gguf package. {'; '.join(errors)}"
        )
    import numpy as np

    weights: dict[str, Any] = {}
    total = max(len(arrays), 1)
    for index, (name, data) in enumerate(arrays.items(), start=1):
        if progress:
            progress(f"Mapping {name}", index - 1, total)
        mapped = map_gguf_key(name)
        if mapped is None:
            continue
        array = np.ascontiguousarray(data)
        weights[mapped] = torch.from_numpy(array)
    if not weights:
        raise ValueError(f"No tensors loaded from GGUF file: {path}")
    if progress:
        progress("Loaded GGUF", total, total)
    return weights


def load_huggingface_weights(path: str | Path, progress: ProgressFn | None = None) -> dict[str, Any]:
    target = Path(path)
    if list(target.glob("*.safetensors")) if target.is_dir() else target.suffix == ".safetensors":
        return load_safetensors_weights(target, progress=progress)
    return load_pytorch_weights(target, progress=progress)


def load_model_weights(
    model_path: str | Path,
    model_format: str | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    target = Path(model_path)
    fmt = model_format or detect_format(target)
    if fmt == "gguf":
        return load_gguf_weights(target, progress=progress)
    if fmt == "safetensors":
        return load_safetensors_weights(target, progress=progress)
    if fmt in {"pytorch", "huggingface"}:
        return load_huggingface_weights(target, progress=progress)
    raise ValueError(f"Unsupported model format '{fmt}' for {target}")


def load_tokenizer_payload(model_path: str | Path) -> dict[str, Any] | None:
    target = Path(model_path)
    search_dirs = [target] if target.is_dir() else [target.parent]
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
        "merges.txt",
    )
    payload: dict[str, Any] = {}
    for directory in search_dirs:
        for name in names:
            file_path = directory / name
            if not file_path.exists():
                continue
            if name.endswith(".json"):
                try:
                    payload[name] = json.loads(file_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
            else:
                payload[name] = file_path.read_text(encoding="utf-8")
    if payload:
        return payload
    if target.is_file() and _is_gguf_file(target):
        return _load_gguf_tokenizer(target)
    return None


def _load_gguf_tokenizer(path: Path) -> dict[str, Any] | None:
    try:
        from inferforge.merger.core.gguf_reader import NativeGGUFReader

        with NativeGGUFReader(path) as reader:
            payload = reader.tokenizer_payload()
            if payload:
                return payload
    except Exception:
        pass
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    reader = GGUFReader(str(path))
    fields = {field.name: field for field in reader.fields.values()}
    tokens_field = fields.get("tokenizer.ggml.tokens")
    if tokens_field is None:
        return None
    tokens: list[str] = []
    for part in tokens_field.contents():
        if isinstance(part, bytes):
            tokens.append(part.decode("utf-8", errors="replace"))
        else:
            tokens.append(str(part))
    vocab = {token: index for index, token in enumerate(tokens)}
    return {
        "tokenizer.json": {
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": [],
        },
        "tokenizer_config.json": {
            "vocab_size": len(tokens),
            "model_max_length": 4096,
        },
        "vocab": vocab,
    }


def _is_valid_weight_target(p: Path) -> bool:
    if not p.exists():
        return False
    if p.is_file():
        return True
    if p.is_dir():
        # Directory must contain actual weight files
        if list(p.glob("*.safetensors")) or list(p.glob("*.bin")) or list(p.glob("*.pt")) or list(p.glob("*.gguf")):
            return True
        if (p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists() or (p / "model.gguf").exists():
            return True
        return False
    return False


def resolve_weight_path(record: Any) -> Path:
    candidates: list[Path] = []
    path_value = ""
    name = "model"
    ollama_name = ""

    if isinstance(record, (str, Path)):
        s = str(record)
        p = Path(s)
        try:
            from inferforge.core.registry import Registry
            reg = Registry()
            rec = reg.get(s)
            if rec is not None:
                record = rec
            elif p.exists():
                path_value = str(p)
                name = p.stem
                ollama_name = name
            else:
                name = s
                ollama_name = s
        except Exception:
            path_value = str(p) if p.exists() else ""
            name = p.stem if p.exists() else s
            ollama_name = name

    if not isinstance(record, (str, Path)):
        path_value = getattr(record, "path", "") or ""
        name = getattr(record, "name", "") or "model"
        ollama_name = getattr(record, "ollama_name", "") or name

    if path_value:
        candidates.append(Path(path_value))
    meta = getattr(record, "meta", {}) or {}
    for key in ("merged_model_path", "weights_path", "local_path", "hf_path"):
        extra = meta.get(key)
        if extra:
            candidates.append(Path(str(extra)))

    # If path_value is a directory without weights (e.g. Modelfile trained model), check for base model
    if path_value and Path(path_value).is_dir() and not _is_valid_weight_target(Path(path_value)):
        dir_p = Path(path_value)
        cfg_p = dir_p / "config.json"
        mf_p = dir_p / "Modelfile"
        base_ref = None
        if cfg_p.exists():
            try:
                cdata = json.loads(cfg_p.read_text(encoding="utf-8"))
                base_ref = cdata.get("ollama_base") or cdata.get("base_model")
            except Exception:
                pass
        if not base_ref and mf_p.exists():
            try:
                for line in mf_p.read_text(encoding="utf-8").splitlines():
                    if line.strip().upper().startswith("FROM "):
                        base_ref = line.strip().split(None, 1)[1].strip()
                        break
            except Exception:
                pass
        if base_ref:
            try:
                from inferforge.core.registry import Registry
                reg = Registry()
                base_rec = reg.get(base_ref)
                if base_rec and base_rec.path:
                    candidates.append(Path(base_rec.path))
            except Exception:
                pass
            ollama_name = base_ref

    try:
        from inferforge.core.config import models_dir, ollama_models_dir

        model_root = models_dir()
        candidates.extend(
            [
                model_root / name,
                model_root / f"{name}.gguf",
                model_root / f"{name}.safetensors",
                model_root / name / "model.gguf",
                model_root / name / "model.safetensors",
            ]
        )
        blobs = ollama_models_dir() / "blobs"
        digest = getattr(record, "digest", "") or ""
        if digest and blobs.exists():
            hexpart = digest.split(":", 1)[-1]
            candidates.extend([blobs / f"sha256-{hexpart}", blobs / hexpart])
        if ollama_name:
            from inferforge.importers.ollama import _model_blob_from_manifest

            manifests = ollama_models_dir() / "manifests"
            if manifests.exists() and blobs.exists():
                _, blob_path = _model_blob_from_manifest(ollama_name, manifests, blobs)
                if blob_path:
                    candidates.append(blob_path)
                # Also check with :latest or without :latest
                alt_tag = ollama_name + ":latest" if ":" not in ollama_name else ollama_name.split(":", 1)[0]
                _, alt_blob = _model_blob_from_manifest(alt_tag, manifests, blobs)
                if alt_blob:
                    candidates.append(alt_blob)
    except Exception:
        pass

    # Also check Registry for related records (e.g. name:latest)
    try:
        from inferforge.core.registry import Registry
        reg = Registry()
        for alt_name in (f"{name}:latest", name.split(":", 1)[0] if ":" in name else name):
            alt_rec = reg.get(alt_name)
            if alt_rec and getattr(alt_rec, "path", ""):
                candidates.append(Path(alt_rec.path))
    except Exception:
        pass

    for candidate in candidates:
        if _is_valid_weight_target(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not locate weights for model '{getattr(record, 'name', record)}'. "
        "Import or pull the model first: forge import ollama  |  forge pull <model>"
    )


def load_record_weights(record: Any, progress: ProgressFn | None = None) -> tuple[dict[str, Any], Path, str]:
    path = resolve_weight_path(record)
    fmt = (getattr(record, "format", "") or "").lower() or detect_format(path)
    if fmt in {"", "unknown"}:
        fmt = detect_format(path)
    weights = load_model_weights(path, fmt, progress=progress)
    return weights, path, fmt


class LazyModelReader:
    """Memory-efficient streaming reader for model weights.
    
    Loads individual tensors on demand using memory mapping (mmap)
    without loading entire multi-gigabyte models into RAM.
    Supports GGUF, SafeTensors, and PyTorch formats.
    """

    def __init__(self, target: Any, format: str | None = None, use_mmap: bool = True):
        self.record = target if hasattr(target, "name") and not isinstance(target, (str, Path)) else None
        self.path = resolve_weight_path(target) if self.record else Path(target)
        self.format = (format or (getattr(self.record, "format", "") or "")).lower() or detect_format(self.path)
        self.use_mmap = use_mmap
        self._gguf_reader = None
        self._safetensor_handles = {}
        self._safetensor_key_to_file = {}
        self._pytorch_weights = None
        self._key_map = {}  # hf_key -> raw_key
        self._reverse_map = {}  # raw_key -> hf_key
        self._tensor_names = []
        self._init_reader()

    def _init_reader(self) -> None:
        if self.format == "gguf":
            from inferforge.merger.core.gguf_reader import NativeGGUFReader

            self._gguf_reader = NativeGGUFReader(self.path, use_mmap=self.use_mmap)
            raw_names = self._gguf_reader.get_tensor_names()
            for raw in raw_names:
                mapped = map_gguf_key(raw)
                if mapped:
                    self._key_map[mapped] = raw
                    self._reverse_map[raw] = mapped
                    self._tensor_names.append(mapped)
        elif self.format == "safetensors":
            from safetensors import safe_open

            files = []
            if self.path.is_dir():
                files = sorted(list(self.path.glob("*.safetensors")))
            elif self.path.is_file():
                files = [self.path]
            for f in files:
                handle = safe_open(str(f), framework="pt", device="cpu")
                self._safetensor_handles[str(f)] = handle
                for key in handle.keys():
                    self._safetensor_key_to_file[key] = str(f)
                    self._tensor_names.append(key)
        else:
            # Fallback for PyTorch
            torch = _require_torch()
            self._pytorch_weights = load_pytorch_weights(self.path)
            self._tensor_names = list(self._pytorch_weights.keys())

    def get_tensor_names(self) -> list[str]:
        return list(self._tensor_names)

    def has_tensor(self, name: str) -> bool:
        if self.format == "gguf":
            raw = self._key_map.get(name, name)
            return self._gguf_reader.has_tensor(raw)
        elif self.format == "safetensors":
            return name in self._safetensor_key_to_file
        elif self._pytorch_weights is not None:
            return name in self._pytorch_weights
        return False

    def get_tensor(self, name: str, dtype: Any = None) -> Any:
        """Load a single tensor on demand with zero unnecessary copies."""
        torch = _require_torch()
        tensor = None
        if self.format == "gguf":
            raw_name = self._key_map.get(name, name)
            arr = self._gguf_reader.load_tensor_by_name(raw_name)
            tensor = torch.from_numpy(arr)
        elif self.format == "safetensors":
            file_path = self._safetensor_key_to_file.get(name)
            if not file_path:
                raise KeyError(f"Tensor '{name}' not found in safetensors")
            handle = self._safetensor_handles[file_path]
            tensor = handle.get_tensor(name)
        elif self._pytorch_weights is not None:
            tensor = self._pytorch_weights.get(name)
            if tensor is None:
                raise KeyError(f"Tensor '{name}' not found in PyTorch weights")

        if tensor is not None and dtype is not None:
            tensor = tensor.to(dtype)
        return tensor

    def get_tokenizer_payload(self) -> dict[str, Any] | None:
        if self.format == "gguf" and self._gguf_reader:
            payload = self._gguf_reader.tokenizer_payload()
            if payload:
                return payload
        return load_tokenizer_payload(self.path)

    def get_architecture(self) -> str:
        if self.format == "gguf" and self._gguf_reader:
            return self._gguf_reader.architecture_fields().get("architecture", "llama")
        torch = _require_torch()
        dummy_weights = {name: torch.zeros(1) for name in self._tensor_names[:25]}
        return detect_model_architecture(dummy_weights)

    def close(self) -> None:
        if self._gguf_reader:
            try:
                self._gguf_reader.close()
            except Exception:
                pass
            self._gguf_reader = None
        for h in list(self._safetensor_handles.values()):
            try:
                del h
            except Exception:
                pass
        self._safetensor_handles.clear()
        self._safetensor_key_to_file.clear()
        if self._pytorch_weights is not None:
            del self._pytorch_weights
            self._pytorch_weights = None
        import gc
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def estimate_merge_memory(
    model_records: list[Any],
    strategy: str = "ties",
    layer_wise: bool = True,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Calculate RAM and disk space requirements before merging.
    
    Provides accurate memory consumption estimates and determines
    whether the system has sufficient resources.
    """
    import shutil
    try:
        import psutil
        avail_ram = psutil.virtual_memory().available
        total_ram = psutil.virtual_memory().total
    except Exception:
        avail_ram = 8 * (1024**3)
        total_ram = 16 * (1024**3)

    total_weight_bytes = 0
    for r in model_records:
        try:
            p = resolve_weight_path(r)
            if p.is_file():
                total_weight_bytes += p.stat().st_size
            elif p.is_dir():
                total_weight_bytes += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        except Exception:
            total_weight_bytes += 4 * (1024**3)

    # In-memory merge requires loading both models (often converted to FP32 for operations) + output buffer
    in_memory_ram_needed = int(total_weight_bytes * 2.2)
    # Layer-wise merge only requires active layer tensors + blender buffers (~300MB - 1GB)
    # For small models, streaming RAM is at most in_memory_ram_needed
    streaming_ram_needed = min(in_memory_ram_needed, int(min(1.2 * (1024**3), max(300 * (1024**2), total_weight_bytes * 0.05))))

    # Disk space check
    dest = Path(output_dir) if output_dir else Path.cwd()
    p = dest.resolve()
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        free_disk = shutil.disk_usage(p).free
    except Exception:
        free_disk = 50 * (1024**3)

    # Estimated output model size
    est_output_size = int(total_weight_bytes / max(len(model_records), 1))
    disk_ok = free_disk > (est_output_size * 1.2)

    can_in_memory = (avail_ram > in_memory_ram_needed)
    recommended_mode = "in_memory" if can_in_memory and not layer_wise else "streaming"

    warning = None
    if not can_in_memory and not layer_wise:
        warning = (
            f"Insufficient RAM for in-memory merge (need {in_memory_ram_needed / (1024**3):.1f} GB, "
            f"have {avail_ram / (1024**3):.1f} GB). Streaming layer-wise merge is recommended."
        )
    if not disk_ok:
        warning = (
            f"Low disk space on {dest} (need {est_output_size / (1024**3):.1f} GB, "
            f"only {free_disk / (1024**3):.1f} GB available)."
        )

    return {
        "can_in_memory": can_in_memory,
        "in_memory_ram_gb": round(in_memory_ram_needed / (1024**3), 2),
        "streaming_ram_gb": round(streaming_ram_needed / (1024**3), 2),
        "available_ram_gb": round(avail_ram / (1024**3), 2),
        "total_ram_gb": round(total_ram / (1024**3), 2),
        "estimated_output_gb": round(est_output_size / (1024**3), 2),
        "available_disk_gb": round(free_disk / (1024**3), 2),
        "disk_ok": disk_ok,
        "recommended_mode": recommended_mode,
        "warning": warning,
    }
