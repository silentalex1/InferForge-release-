from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inferforge.merger.core.loader import (
    build_hf_config,
    detect_format,
    detect_model_architecture,
    load_model_weights,
    load_tokenizer_payload,
    map_gguf_key,
)


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for format conversion. Install with: pip install 'inferforge[merging]'"
        ) from exc


def prepare_weights_for_saving(weights: dict[str, Any], target_format: str = "bfloat16") -> dict[str, Any]:
    torch = _require_torch()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(target_format.lower(), torch.bfloat16)
    ready: dict[str, Any] = {}
    for key, tensor in weights.items():
        if not hasattr(tensor, "contiguous"):
            tensor = torch.as_tensor(tensor)
        if tensor.dtype == target_dtype and tensor.device.type == "cpu" and tensor.is_contiguous():
            ready[key] = tensor.detach()
        else:
            ready[key] = tensor.detach().to(target_dtype).contiguous().cpu()
    return ready


def _to_cpu_contiguous(weights: dict[str, Any]) -> dict[str, Any]:
    return prepare_weights_for_saving(weights, "float32")


def quantize_to_format(weights: dict[str, Any], target_format: str) -> dict[str, Any]:
    return prepare_weights_for_saving(weights, target_format)


def gguf_to_safetensors(gguf_path: str | Path, output_path: str | Path | None = None) -> Path:
    source = Path(gguf_path)
    weights = load_model_weights(source, "gguf")
    mapped = {}
    for key, tensor in weights.items():
        mapped_key = map_gguf_key(key) or key
        mapped[mapped_key] = tensor
    destination = Path(output_path) if output_path else source.with_suffix(".safetensors")
    if destination.suffix.lower() != ".safetensors" and destination.suffix:
        destination = destination.with_suffix(".safetensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_merged_model(
        mapped,
        destination.parent if destination.suffix == "" else destination.parent,
        architecture=detect_model_architecture(mapped),
        tokenizer_payload=load_tokenizer_payload(source),
        filename=destination.name if destination.suffix else "model.safetensors",
    )
    if destination.suffix == ".safetensors":
        file_path = destination.parent / destination.name
        if file_path.exists():
            return file_path
    produced = destination.parent / "model.safetensors"
    if produced.exists():
        return produced
    return destination


def save_merged_model(
    weights: dict[str, Any],
    output_dir: str | Path,
    architecture: str | None = None,
    tokenizer_payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    filename: str = "model.safetensors",
    precision: str = "bfloat16",
) -> Path:
    from safetensors.torch import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arch = architecture or detect_model_architecture(weights)
    converted = prepare_weights_for_saving(weights, precision)
    weight_path = out / filename
    save_file(converted, str(weight_path))
    config = build_hf_config(converted, arch)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    generation = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 2,
        "max_length": config.get("max_position_embeddings", 4096),
    }
    (out / "generation_config.json").write_text(json.dumps(generation, indent=2), encoding="utf-8")
    if tokenizer_payload:
        _write_tokenizer(out, tokenizer_payload)
    if metadata:
        (out / "merge_manifest.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return out


def _write_tokenizer(output_dir: Path, payload: dict[str, Any]) -> None:
    tokenizer_json = payload.get("tokenizer.json")
    if isinstance(tokenizer_json, dict):
        (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json, indent=2), encoding="utf-8")
    tokenizer_config = payload.get("tokenizer_config.json")
    if isinstance(tokenizer_config, dict):
        (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2), encoding="utf-8")
    vocab = payload.get("vocab.json") or payload.get("vocab")
    if isinstance(vocab, dict) and "tokenizer.json" not in payload:
        (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    special = payload.get("special_tokens_map.json")
    if isinstance(special, dict):
        (output_dir / "special_tokens_map.json").write_text(json.dumps(special, indent=2), encoding="utf-8")
    merges = payload.get("merges.txt")
    if isinstance(merges, str):
        (output_dir / "merges.txt").write_text(merges, encoding="utf-8")


def convert_model(source: str | Path, dest: str | Path, target_format: str = "safetensors") -> Path:
    src = Path(source)
    fmt = detect_format(src)
    weights = load_model_weights(src, fmt)
    architecture = detect_model_architecture(weights)
    tokenizer_payload = load_tokenizer_payload(src)
    dest_path = Path(dest)
    if target_format == "safetensors":
        if dest_path.suffix == ".safetensors":
            return save_merged_model(
                weights,
                dest_path.parent,
                architecture=architecture,
                tokenizer_payload=tokenizer_payload,
                filename=dest_path.name,
            ) / dest_path.name
        return save_merged_model(
            weights,
            dest_path,
            architecture=architecture,
            tokenizer_payload=tokenizer_payload,
        )
    raise ValueError(f"Unsupported conversion target: {target_format}")
