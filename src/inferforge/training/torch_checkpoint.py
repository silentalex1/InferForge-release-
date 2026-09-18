"""PyTorch checkpoint loading, validation, and conversion utilities.

Supports:
- state_dict checkpoints ({"state_dict": ...} or plain tensors)
- full-model checkpoints (pickled nn.Module)
- torch.save zip format (modern) and legacy serialization
- safetensors files
- sharded checkpoints (pytorch_model-00001-of-000N.bin style)

Everything degrades gracefully when torch is not installed.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint of any supported format into a tensor dict.

    Handles:
      - plain state_dict: {layer_name: tensor}
      - wrapped: {"state_dict": {...}} / {"model": {...}} / {"module": {...}}
      - full pickled models (extracts .state_dict() when possible)
      - safetensors
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    if path.suffix.lower() == ".safetensors":
        return _load_safetensors(path)

    torch = _load_torch()
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Run: pip install torch")

    # weights_only=True is the safe default for modern torch; fall back for
    # full-model pickles and older serialization formats.
    obj: Any = None
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        try:
            obj = torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            try:  # very old torch versions have no weights_only kwarg
                obj = torch.load(path, map_location=map_location)
            except Exception as exc:
                raise ValueError(f"could not load checkpoint {path.name}: {exc}") from exc

    return unwrap_state_dict(obj)


def unwrap_state_dict(obj: Any) -> dict[str, Any]:
    """Extract the raw tensor dict from any checkpoint shape."""
    torch = _load_torch()

    if torch is not None and hasattr(obj, "state_dict"):
        obj = obj.state_dict()

    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module", "model_state_dict", "weights"):
            inner = obj.get(key)
            if isinstance(inner, dict):
                obj = inner
                break

    if torch is not None and isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            # strip DDP "module." prefixes
            name = k
            while name.startswith("module."):
                name = name[len("module."):]
            cleaned[name] = v
        return cleaned

    if isinstance(obj, dict):
        return obj
    raise ValueError("checkpoint does not contain a tensor state_dict")


def _load_safetensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
        return dict(load_file(str(path)))
    except ImportError:
        raise RuntimeError("safetensors is required for .safetensors files. Run: pip install safetensors")


def is_sharded_checkpoint(directory: Path) -> bool:
    """True when the directory holds a sharded checkpoint (HF style)."""
    if not directory.is_dir():
        return False
    return any(re.match(r"pytorch_model-\d+-of-\d+\.bin", f.name) for f in directory.iterdir())


def load_sharded_checkpoint(directory: Path) -> dict[str, Any]:
    """Load a sharded checkpoint (pytorch_model-00001-of-00002.bin, ...)."""
    shards = sorted(
        f for f in directory.iterdir()
        if re.match(r"pytorch_model-\d+-of-\d+\.bin", f.name)
    )
    if not shards:
        raise FileNotFoundError(f"no shards found in {directory}")
    merged: dict[str, Any] = {}
    for shard in shards:
        merged.update(load_checkpoint(shard))
    return merged


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

@dataclass
class CheckpointInfo:
    path: str
    format: str
    keys: int
    total_params: int
    size_bytes: int
    dtype_counts: dict = field(default_factory=dict)
    layers: int = 0
    embed_dim: int | None = None
    vocab_size: int | None = None
    has_lora: bool = False
    torch_version: str | None = None
    notes: list[str] = field(default_factory=list)


def inspect_checkpoint(path: str | Path) -> CheckpointInfo:
    """Gather detailed info about a checkpoint without loading tensors to GPU."""
    path = Path(path)
    state = load_checkpoint(path)
    torch = _load_torch()

    total_params = 0
    dtype_counts: dict[str, int] = {}
    layers = 0
    embed_dim = None
    vocab_size = None
    has_lora = False

    for key, value in state.items():
        if torch is not None and torch.is_tensor(value):
            total_params += value.numel()
            dt = str(value.dtype).replace("torch.", "")
            dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
            if "." in key or key.endswith(".weight") or key.endswith(".bias"):
                layers += 1
            if embed_dim is None and (key.endswith("weight") and value.dim() == 2):
                embed_dim = value.shape[1]
            if vocab_size is None and ("embed" in key.lower() or "lm_head" in key.lower()) and value.dim() == 2:
                vocab_size = value.shape[0]
        if "lora" in key.lower():
            has_lora = True

    fmt = "safetensors" if path.suffix.lower() == ".safetensors" else "torch"
    info = CheckpointInfo(
        path=str(path),
        format=fmt,
        keys=len(state),
        total_params=total_params,
        size_bytes=path.stat().st_size,
        dtype_counts=dtype_counts,
        layers=layers,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        has_lora=has_lora,
    )
    if not torch_available():
        info.notes.append("torch not installed — loaded via safetensors reader only" if fmt == "safetensors" else "torch not installed")
    return info


def validate_checkpoint(path: str | Path) -> tuple[bool, list[str]]:
    """Validate a checkpoint. Returns (ok, issues)."""
    issues: list[str] = []
    path = Path(path)
    if not path.exists():
        return False, [f"file not found: {path}"]
    if path.suffix.lower() not in CHECKPOINT_EXTENSIONS:
        issues.append(f"unusual extension: {path.suffix} (expected one of {sorted(CHECKPOINT_EXTENSIONS)})")
    try:
        state = load_checkpoint(path)
    except Exception as exc:
        return False, [f"failed to load: {exc}"]

    torch = _load_torch()
    if not state:
        issues.append("checkpoint is empty")
    for key, value in state.items():
        if torch is not None and torch.is_tensor(value):
            if not torch.isfinite(value.float()).all():
                issues.append(f"non-finite values in '{key}' (NaN/Inf)")
            if value.numel() == 0:
                issues.append(f"empty tensor '{key}'")
            if str(value.dtype) == "torch.bfloat16" and not torch_available():
                issues.append("bfloat16 tensor without torch")
    if issues:
        return False, issues
    return True, []


def compare_checkpoints(a: str | Path, b: str | Path) -> dict[str, Any]:
    """Compare two checkpoints: missing keys, shape mismatches, parameter deltas."""
    sa, sb = load_checkpoint(a), load_checkpoint(b)
    torch = _load_torch()

    only_a = sorted(set(sa) - set(sb))
    only_b = sorted(set(sb) - set(sa))
    shared = sorted(set(sa) & set(sb))
    shape_mismatches = []
    drift = []
    for key in shared:
        ta, tb = sa[key], sb[key]
        if torch is not None and torch.is_tensor(ta) and torch.is_tensor(tb):
            if ta.shape != tb.shape:
                shape_mismatches.append({"key": key, "a": list(ta.shape), "b": list(tb.shape)})
            else:
                try:
                    diff = (ta.float() - tb.float()).abs().mean().item()
                    drift.append((key, diff))
                except Exception:
                    continue

    drift.sort(key=lambda kv: kv[1], reverse=True)
    same_tensors = 0
    if torch is not None:
        same_tensors = sum(
            1 for k in shared if torch.is_tensor(sa[k]) and torch.is_tensor(sb[k]) and torch.equal(sa[k], sb[k])
        )

    return {
        "a": str(a),
        "b": str(b),
        "shared_keys": len(shared),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "shape_mismatches": shape_mismatches,
        "identical_tensors": same_tensors,
        "max_drift_keys": [{"key": k, "mean_abs_diff": round(v, 6)} for k, v in drift[:10]],
        "mean_drift": round(sum(v for _, v in drift) / len(drift), 8) if drift else 0.0,
    }


# ---------------------------------------------------------------------------
# Saving / conversion
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict[str, Any], path: str | Path, compress: bool = False) -> Path:
    """Save a state_dict, optionally gzip-compressed."""
    torch = _load_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch is None:
        raise RuntimeError("PyTorch is required to save torch checkpoints")
    try:
        torch.save(state, path, _use_new_zipfile_serialization=True)
    except TypeError:
        torch.save(state, path)
    if compress:
        import gzip
        raw = path.read_bytes()
        out = path.with_suffix(path.suffix + ".gz")
        with gzip.open(out, "wb", compresslevel=6) as fh:
            fh.write(raw)
        return out
    return path


def merge_checkpoints(paths: list[str | Path], output: str | Path, strategy: str = "average") -> Path:
    """Merge multiple state_dicts into one.

    Strategies:
      average  — element-wise mean of all checkpoints
      sum      — element-wise sum
      first    — first checkpoint wins for any key
    """
    torch = _load_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required to merge checkpoints")
    states = [load_checkpoint(p) for p in paths]
    if not states:
        raise ValueError("no checkpoints given")

    keys = set(states[0])
    for s in states[1:]:
        keys &= set(s)

    merged: dict[str, Any] = {}
    for key in sorted(keys):
        tensors = [s[key].float() for s in states if torch.is_tensor(s[key])]
        if not tensors:
            merged[key] = states[0][key]
            continue
        if len(tensors) != len(states):
            merged[key] = tensors[0]
            continue
        if strategy == "average":
            acc = tensors[0].clone()
            for t in tensors[1:]:
                acc += t
            merged[key] = (acc / len(tensors)).to(states[0][key].dtype)
        elif strategy == "sum":
            acc = tensors[0].clone()
            for t in tensors[1:]:
                acc += t
            merged[key] = acc.to(states[0][key].dtype)
        else:  # first
            merged[key] = tensors[0].to(states[0][key].dtype)
    return save_checkpoint(merged, output)


def prune_checkpoint(path: str | Path, output: str | Path, ratio: float = 0.5) -> Path:
    """Magnitude-prune weights (unstructured, per-tensor) by a ratio."""
    torch = _load_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required to prune checkpoints")
    state = load_checkpoint(path)
    pruned: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.dim() >= 2 and value.is_floating_point():
            flat = value.float().abs().flatten()
            k = max(1, int(flat.numel() * ratio))
            threshold = torch.kthvalue(flat, k).values
            mask = value.float().abs() > threshold
            pruned[key] = (value.float() * mask).to(value.dtype)
        else:
            pruned[key] = value
    return save_checkpoint(pruned, output)


def export_gguf(state: dict[str, Any], output: str | Path, architecture: str = "inferforge") -> Path:
    """Minimal GGUF export: writes tensor data with a valid header.

    Supports F32 / F16 tensors. This is a lightweight exporter for small
    models; full GGUF conversion for llama-family architectures should use
    llama.cpp's convert script.
    """
    torch = _load_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for GGUF export")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    GGUF_MAGIC = b"GGUF"
    VERSION = 3

    def _encode_str(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    kv: list[bytes] = [(_encode_str("general.architecture") + struct.pack("<I", 8) + _encode_str(architecture))]

    tensor_names = sorted(state.keys())
    with output.open("wb") as fh:
        fh.write(GGUF_MAGIC)
        fh.write(struct.pack("<I", VERSION))
        fh.write(struct.pack("<Q", len(kv)))
        fh.write(struct.pack("<Q", len(tensor_names)))
        for kvb in kv:
            fh.write(kvb)
        # align to 32 bytes for tensor data
        pos = fh.tell()
        pad = (-pos) % 32
        fh.write(b"\x00" * pad)
        for name in tensor_names:
            t = state[name]
            if not torch.is_tensor(t):
                continue
            t = t.contiguous().cpu()
            if t.dtype == torch.bfloat16:
                t = t.float()
            if t.dtype == torch.float32:
                gtype = 0
                data = t.numpy().tobytes()
            elif t.dtype == torch.float16:
                gtype = 1
                data = t.numpy().tobytes()
            else:
                t = t.float()
                gtype = 0
                data = t.numpy().tobytes()
            fh.write(_encode_str(name))
            fh.write(struct.pack("<I", len(t.shape)))
            for dim in reversed(t.shape):
                fh.write(struct.pack("<Q", dim))
            fh.write(struct.pack("<I", gtype))
            fh.write(struct.pack("<Q", 0))  # offset, resolved on read
            fh.write(data)
    return output


def export_onnx(model: Any, output: str | Path, example_input: Any | None = None) -> Path:
    """Export a full nn.Module to ONNX."""
    torch = _load_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for ONNX export")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if example_input is None:
        example_input = torch.zeros(1, 8, dtype=torch.long)
    torch.onnx.export(model, example_input, str(output), opset_version=17)
    return output


def auto_detect_device() -> str:
    torch = _load_torch()
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_latest_checkpoint(directory: Path) -> Path | None:
    """Find the newest checkpoint file in a directory (by step number then mtime)."""
    if not directory.is_dir():
        return None
    candidates = [f for f in directory.iterdir() if f.suffix.lower() in CHECKPOINT_EXTENSIONS]
    if not candidates:
        return None

    def step_of(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    candidates.sort(key=lambda p: (step_of(p), p.stat().st_mtime), reverse=True)
    return candidates[0]
