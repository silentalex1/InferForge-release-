from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Callable

from inferforge.merger.alignment import DynamicSVD, ProcrustesAligner
from inferforge.merger.core.converter import save_merged_model
from inferforge.merger.core.loader import (
    LazyModelReader,
    detect_model_architecture,
    estimate_merge_memory,
    load_record_weights,
    load_tokenizer_payload,
    resolve_weight_path,
)
from inferforge.merger.core.tensor_utils import TensorUtils
from inferforge.merger.core.tokenizer_aligner import TokenizerAligner
from inferforge.merger.core.weight_blender import MergeConfig, MergeStrategy, WeightBlender
from inferforge.merger.execution import FisherImportanceMask
from inferforge.merger.validation import MergeEvaluator, MergeQuality

ProgressFn = Callable[[str, int, int, dict[str, Any] | None], None]


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for model merging. Install with: pip install 'inferforge[merging]'"
        ) from exc


def _layer_index(name: str) -> int:
    import re

    match = re.search(r"(?:layers|blk|h)\.(\d+)", name)
    return int(match.group(1)) if match else -1


def _is_embed(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("embed_tokens", "token_embd", "wte", "lm_head", "output.weight"))


class MergePipeline:
    def __init__(
        self,
        config: MergeConfig | None = None,
        enable_procrustes: bool = False,
        enable_fisher: bool = False,
        enable_evaluation: bool = False,
        enable_svd: bool = True,
        per_layer_coeffs: str | None = None,
        auto_optimize: bool = False,
        progress: ProgressFn | None = None,
        device: str = "cpu",
        lazy_load: bool = True,
        parallel_layers: bool = False,
        num_workers: int = 0,
        use_mmap: bool = True,
        resume: bool = False,
        slices: list[dict] | None = None,
        base_model: Any | None = None,
        skip_validation: bool = False,
    ) -> None:
        self.config = config or MergeConfig(normalize=False)
        self.enable_procrustes = enable_procrustes
        self.enable_fisher = enable_fisher
        self.enable_evaluation = enable_evaluation
        self.enable_svd = enable_svd
        self.per_layer_coeffs = per_layer_coeffs or "uniform"
        self.auto_optimize = auto_optimize
        self.progress = progress
        self.device = device
        self.lazy_load = lazy_load
        self.parallel_layers = parallel_layers
        self.num_workers = num_workers
        self.use_mmap = use_mmap
        self.resume = resume
        self.slices = slices
        self.base_model = base_model
        self.skip_validation = skip_validation
        self.blender = WeightBlender(self.config)
        self.procrustes = ProcrustesAligner() if enable_procrustes else None
        self.fisher = FisherImportanceMask() if enable_fisher else None
        self.evaluator = MergeEvaluator() if enable_evaluation else None
        self.svd = DynamicSVD() if enable_svd else None

    def _emit(self, stage: str, current: int, total: int, extra: dict[str, Any] | None = None) -> None:
        if self.progress:
            self.progress(stage, current, total, extra)

    def _check_environment(self, model_records: list[Any], output_dir: Path) -> None:
        """Validate file permissions, path accessibility, and disk space."""
        import shutil

        # Verify output dir permission
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            probe = output_dir / f".write_test_{int(time.time() * 1000)}"
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            try:
                probe.unlink(missing_ok=True)
            except Exception:
                pass
        except (PermissionError, OSError) as exc:
            raise PermissionError(
                f"Cannot write to output directory {output_dir}. Please check folder permissions: {exc}"
            ) from exc

        # Verify model paths exist and are accessible
        for rec in model_records:
            try:
                p = resolve_weight_path(rec)
                if not p.exists():
                    raise FileNotFoundError(f"Model weight path not found: {p}")
                # Test read access
                if p.is_file():
                    with open(p, "rb") as f:
                        f.read(16)
            except PermissionError as exc:
                raise PermissionError(f"Permission denied reading model '{getattr(rec, 'name', rec)}': {exc}") from exc
            except Exception as exc:
                if not isinstance(exc, FileNotFoundError):
                    raise RuntimeError(f"Error accessing model weights for '{getattr(rec, 'name', rec)}': {exc}") from exc
                raise

        # Check free disk space
        try:
            p = output_dir.resolve()
            while not p.exists() and p.parent != p:
                p = p.parent
            free_disk = shutil.disk_usage(p).free
            if free_disk < 200 * (1024**2):  # Less than 200MB
                raise RuntimeError(
                    f"Insufficient disk space on {output_dir}: only {free_disk / (1024**3):.2f} GB free."
                )
        except Exception as exc:
            if "Insufficient disk space" in str(exc):
                raise

    def run(self, model_records: list[Any], output_dir: str | Path) -> Path:
        torch = _require_torch()
        if len(model_records) < 2:
            raise ValueError("At least two models are required for merging")

        out_path = Path(output_dir)
        self._check_environment(model_records, out_path)

        # Pre-merge memory estimation
        mem_info = estimate_merge_memory(model_records, strategy=str(self.config.strategy.value if hasattr(self.config.strategy, "value") else self.config.strategy))
        if not mem_info["can_in_memory"] and not self.lazy_load:
            self._emit("Low RAM detected — auto-switching to streaming layer-wise mode", 0, 1)
            self.lazy_load = True

        # If any model is GGUF or lazy_load is active, execute streaming pipeline
        formats = [getattr(r, "format", "").lower() for r in model_records]
        is_gguf = any("gguf" in f for f in formats)
        if self.lazy_load or is_gguf:
            return self._run_streaming(model_records, out_path)

        loaded: list[dict[str, Any]] = []
        sources: list[Path] = []
        tokenizers: list[dict[str, Any]] = []
        architectures: list[str] = []
        total_models = len(model_records)
        for index, record in enumerate(model_records):
            self._emit(f"Loading {getattr(record, 'name', index)}", index, total_models)
            weights, path, _fmt = load_record_weights(record)
            if not weights:
                raise ValueError(f"No weights loaded from {getattr(record, 'name', path)}")
            loaded.append(weights)
            sources.append(path)
            payload = load_tokenizer_payload(path)
            if payload:
                tokenizers.append(payload)
            architectures.append(detect_model_architecture(weights))
            TensorUtils.clear_cache()
        self._emit("Loaded source models", total_models, total_models, {"architectures": architectures})
        loaded = self._align_tokenizers(loaded, tokenizers)
        keys = self._union_keys(loaded)
        if not keys:
            raise ValueError("Source models have no mergeable tensors")
        layer_count = max((_layer_index(key) for key in keys), default=0) + 1
        coefficients = []
        if self.evaluator:
            coefficients = self.evaluator.generate_per_layer_coefficients(
                layer_count,
                base_ratio=self.config.interpolation,
                pattern=self.per_layer_coeffs,
            )
        merged: dict[str, Any] = {}
        baseline = {key: tensor.clone() for key, tensor in loaded[0].items()}
        total_keys = len(keys)
        for index, key in enumerate(keys):
            self._emit(f"Merging {key}", index, total_keys)
            tensors = [weights[key] for weights in loaded if key in weights]
            if not tensors:
                continue
            if len(tensors) == 1:
                merged[key] = tensors[0].detach().contiguous()
                continue
            tensors = self._align_tensor_group(tensors)
            layer_i = _layer_index(key)
            if coefficients and 0 <= layer_i < len(coefficients):
                self.blender.config.interpolation = float(coefficients[layer_i])
            if self.enable_procrustes and len(tensors) >= 2 and tensors[0].ndim >= 2:
                aligned = [tensors[0]]
                for extra in tensors[1:]:
                    try:
                        _rotation, rotated = self.procrustes.orthogonal_procrustes(extra, tensors[0])
                        aligned.append(rotated)
                    except Exception:
                        aligned.append(extra)
                tensors = aligned
            if self.enable_fisher and len(tensors) >= 2:
                result = tensors[0]
                for extra in tensors[1:]:
                    result = self.fisher.adaptive_merge(
                        result,
                        extra,
                        merge_ratio=self.blender.config.interpolation,
                    )
                merged_tensor = result
            else:
                merged_tensor = self.blender.merge_weights(tensors, base_weights=tensors[0])
            if isinstance(merged_tensor, list):
                merged_tensor = self.blender.simple_average(merged_tensor)
            if not torch.isfinite(merged_tensor.float().sum()):
                merged_tensor = tensors[0]
            merged[key] = merged_tensor.detach().contiguous()
            if self.enable_evaluation and key in baseline and index % 8 == 0:
                evaluation = self.evaluator.evaluate_layer_merge(
                    {key: merged[key]},
                    {key: baseline[key]},
                    layer_index=max(layer_i, 0),
                )
                if evaluation.should_rollback:
                    merged[key] = baseline[key]
                elif self.auto_optimize:
                    self.blender.config.interpolation = self.evaluator.auto_tune_merge_ratio(
                        self.blender.config.interpolation,
                        self.evaluator.quality_history,
                    )
            if index % 16 == 0:
                TensorUtils.clear_cache()
        self._emit("Saving merged model", total_keys, total_keys)
        architecture = architectures[0] if architectures else detect_model_architecture(merged)
        tokenizer_payload = tokenizers[0] if tokenizers else None
        metadata = {
            "merged": True,
            "strategy": self.config.strategy.value if isinstance(self.config.strategy, MergeStrategy) else str(self.config.strategy),
            "precision": self.config.precision,
            "interpolation": self.config.interpolation,
            "ties_param_k": self.config.ties_param_k,
            "source_models": [getattr(record, "name", str(record)) for record in model_records],
            "source_paths": [str(path) for path in sources],
            "architectures": architectures,
            "tensor_count": len(merged),
            "parameter_count": int(sum(tensor.numel() for tensor in merged.values())),
            "procrustes": self.enable_procrustes,
            "fisher": self.enable_fisher,
            "evaluation": self.enable_evaluation,
            "svd": self.enable_svd,
            "created_at": int(time.time()),
        }
        if self.evaluator:
            metadata["final_metrics"] = self.evaluator.evaluate_final_model(merged)
            metadata["quality"] = (
                self.evaluator.quality_history[-1].quality.value
                if self.evaluator.quality_history
                else MergeQuality.GOOD.value
            )
        output = save_merged_model(
            merged,
            output_dir,
            architecture=architecture,
            tokenizer_payload=tokenizer_payload,
            metadata=metadata,
            precision=self.config.precision,
        )
        self._verify_saved(output)
        del loaded, merged, baseline
        TensorUtils.clear_cache()
        gc.collect()
        return output

    def _run_streaming(self, model_records: list[Any], output_dir: Path) -> Path:
        torch = _require_torch()
        import json
        import time

        readers: list[LazyModelReader] = []
        sources: list[Path] = []
        try:
            total_models = len(model_records)
            for index, record in enumerate(model_records):
                self._emit(f"Indexing {getattr(record, 'name', index)}", index, total_models)
                reader = LazyModelReader(record, use_mmap=self.use_mmap)
                readers.append(reader)
                sources.append(reader.path)

            architectures = [r.get_architecture() for r in readers]
            tokenizers = [r.get_tokenizer_payload() for r in readers if r.get_tokenizer_payload()]
            self._emit("Indexed source models (streaming mode)", total_models, total_models, {"architectures": architectures})

            # Base model determination
            base_reader = readers[0]
            if self.base_model:
                for r in readers:
                    if getattr(r.record, "name", "") == str(self.base_model) or str(self.base_model) in str(r.path):
                        base_reader = r
                        break

            def _get_tensor_canonical(r, k):
                if r.has_tensor(k):
                    return r.get_tensor(k)
                if k.startswith("model.") and r.has_tensor(k[6:]):
                    return r.get_tensor(k[6:])
                if not k.startswith("model.") and r.has_tensor("model." + k):
                    return r.get_tensor("model." + k)
                if k == "lm_head.weight" and r.has_tensor("model.lm_head.weight"):
                    return r.get_tensor("model.lm_head.weight")
                if k == "model.lm_head.weight" and r.has_tensor("lm_head.weight"):
                    return r.get_tensor("lm_head.weight")
                return None

            canonical_keys = set()
            for r in readers:
                for k in r.get_tensor_names():
                    if k.startswith("layers."):
                        canonical_keys.add("model." + k)
                    else:
                        canonical_keys.add(k)
            
            # Apply layer filter if defined
            layer_filter = getattr(self, "layer_filter", None)
            if callable(layer_filter):
                canonical_keys = {k for k in canonical_keys if layer_filter(k)}

            keys = sorted(list(canonical_keys), key=lambda k: (_layer_index(k), k))

            # Checkpoint resumption
            ckpt_file = output_dir / ".merge_checkpoint.json"
            merged: dict[str, torch.Tensor] = {}
            completed_keys: set[str] = set()
            if self.resume and ckpt_file.exists():
                try:
                    ckpt_data = json.loads(ckpt_file.read_text(encoding="utf-8"))
                    completed_keys = set(ckpt_data.get("completed_keys", []))
                    self._emit(
                        f"Resuming merge from checkpoint ({len(completed_keys)} tensors already done)",
                        len(completed_keys),
                        len(keys),
                    )
                except Exception:
                    completed_keys = set()

            total_keys = len(keys)
            layer_count = max((_layer_index(key) for key in keys), default=0) + 1
            coefficients = []
            if self.evaluator:
                coefficients = self.evaluator.generate_per_layer_coefficients(
                    layer_count,
                    base_ratio=self.config.interpolation,
                    pattern=self.per_layer_coeffs,
                )

            target_dtype = self.blender._get_target_dtype()

            for index, key in enumerate(keys):
                if key in completed_keys and key in merged:
                    continue
                self._emit(f"Streaming & merging {key}", index, total_keys)

                tensors = []
                for r in readers:
                    t = _get_tensor_canonical(r, key)
                    if t is not None:
                        tensors.append(t)
                if not tensors:
                    continue
                if len(tensors) == 1:
                    merged[key] = tensors[0].to(target_dtype).detach().contiguous().cpu()
                    del tensors
                    continue

                base_tensor = _get_tensor_canonical(base_reader, key)
                if base_tensor is None:
                    base_tensor = tensors[0]

                tensors = self._align_tensor_group(tensors)
                layer_i = _layer_index(key)
                if coefficients and 0 <= layer_i < len(coefficients):
                    self.blender.config.interpolation = float(coefficients[layer_i])

                if self.enable_procrustes and len(tensors) >= 2 and tensors[0].ndim >= 2:
                    aligned = [tensors[0]]
                    for extra in tensors[1:]:
                        try:
                            _rotation, rotated = self.procrustes.orthogonal_procrustes(extra, tensors[0])
                            aligned.append(rotated)
                        except Exception:
                            aligned.append(extra)
                    tensors = aligned

                if self.enable_fisher and len(tensors) >= 2:
                    result = tensors[0]
                    for extra in tensors[1:]:
                        result = self.fisher.adaptive_merge(
                            result,
                            extra,
                            merge_ratio=self.blender.config.interpolation,
                        )
                    merged_tensor = result
                else:
                    merged_tensor = self.blender.merge_weights(tensors, base_weights=base_tensor, strategy=self.config.strategy)

                if isinstance(merged_tensor, list):
                    merged_tensor = self.blender.simple_average(merged_tensor)
                if not torch.isfinite(merged_tensor.float().sum()):
                    merged_tensor = tensors[0]

                merged[key] = merged_tensor.to(target_dtype).detach().contiguous().cpu()

                # Clean up memory immediately
                del tensors, base_tensor, merged_tensor
                if index % 8 == 0:
                    gc.collect()
                    TensorUtils.clear_cache()

                # Periodic checkpointing
                if (index + 1) % 25 == 0:
                    try:
                        ckpt_file.write_text(
                            json.dumps({
                                "completed_keys": list(merged.keys()),
                                "last_key": key,
                                "timestamp": time.time(),
                            }),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

            self._emit("Saving merged model", total_keys, total_keys)
            architecture = architectures[0] if architectures else detect_model_architecture(merged)
            tokenizer_payload = tokenizers[0] if tokenizers else None
            metadata = {
                "merged": True,
                "streaming": True,
                "strategy": self.config.strategy.value if isinstance(self.config.strategy, MergeStrategy) else str(self.config.strategy),
                "precision": self.config.precision,
                "interpolation": self.config.interpolation,
                "ties_param_k": self.config.ties_param_k,
                "source_models": [getattr(record, "name", str(record)) for record in model_records],
                "source_paths": [str(path) for path in sources],
                "architectures": architectures,
                "tensor_count": len(merged),
                "parameter_count": int(sum(tensor.numel() for tensor in merged.values())),
                "procrustes": self.enable_procrustes,
                "fisher": self.enable_fisher,
                "created_at": int(time.time()),
            }
            output = save_merged_model(
                merged,
                output_dir,
                architecture=architecture,
                tokenizer_payload=tokenizer_payload,
                metadata=metadata,
                precision=self.config.precision,
            )
            self._verify_saved(output)

            # Cleanup checkpoint on completion
            if ckpt_file.exists():
                try:
                    ckpt_file.unlink()
                except Exception:
                    pass

            del merged
            TensorUtils.clear_cache()
            gc.collect()
            return output
        finally:
            for r in readers:
                try:
                    r.close()
                except Exception:
                    pass

    def _verify_saved(self, output: Path) -> None:
        from inferforge.merger.core.loader import load_model_weights

        self._emit("Verifying merged weights", 0, 1)
        config_path = output / "config.json"
        weight_path = output / "model.safetensors"
        if not config_path.exists():
            raise RuntimeError(f"Merged model is missing config.json at {output}")
        if not weight_path.exists():
            raise RuntimeError(f"Merged model is missing model.safetensors at {output}")
        reloaded = load_model_weights(output)
        if not reloaded:
            raise RuntimeError("Merged model saved but reloading produced no tensors")
        self._emit("Verified merged weights", 1, 1, {"tensors": len(reloaded)})

    def _union_keys(self, loaded: list[dict[str, Any]]) -> list[str]:
        counts: dict[str, int] = {}
        for weights in loaded:
            for key in weights:
                counts[key] = counts.get(key, 0) + 1
        shared = [key for key, count in counts.items() if count >= 2]
        unique = [key for key, count in counts.items() if count == 1]
        shared.sort()
        unique.sort()
        return shared + unique

    def _align_tensor_group(self, tensors: list[Any]) -> list[Any]:
        torch = _require_torch()
        reference = tensors[0]
        aligned = [reference]
        for tensor in tensors[1:]:
            if tensor.shape == reference.shape:
                aligned.append(tensor)
                continue
            if self.svd is not None and tensor.ndim == 2 and reference.ndim == 2:
                try:
                    if tensor.shape[1] != reference.shape[1]:
                        if tensor.shape[1] > reference.shape[1]:
                            tensor = self.svd.compress_via_svd(tensor, reference.shape[1])
                        else:
                            tensor = self.svd.expand_via_svd(tensor, reference.shape[1])
                    if tensor.shape[0] != reference.shape[0]:
                        rows = min(tensor.shape[0], reference.shape[0])
                        if tensor.shape[0] > reference.shape[0]:
                            tensor = tensor[:rows]
                        else:
                            pad = torch.zeros(
                                (reference.shape[0] - tensor.shape[0], tensor.shape[1]),
                                dtype=tensor.dtype,
                                device=tensor.device,
                            )
                            tensor = torch.cat([tensor, pad], dim=0)
                    aligned.append(tensor)
                    continue
                except Exception:
                    pass
            is_norm = (tensor.ndim == 1 and reference.ndim == 1 and abs(tensor.float().mean().item() - 1.0) < 0.35)
            aligned.append(TensorUtils.match_shape(tensor, reference.shape, is_norm_scale=is_norm))
        return aligned

    def _align_tokenizers(
        self,
        loaded: list[dict[str, Any]],
        tokenizers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        embed_keys = []
        head_keys = []
        for weights in loaded:
            embed = next((key for key in weights if _is_embed(key) and "lm_head" not in key and "output" not in key), None)
            head = next((key for key in weights if "lm_head" in key or key.endswith("output.weight")), None)
            embed_keys.append(embed)
            head_keys.append(head)
        if not all(embed_keys) or not all(head_keys):
            return loaded
        embeddings = [loaded[i][embed_keys[i]] for i in range(len(loaded))]
        heads = [loaded[i][head_keys[i]] for i in range(len(loaded))]
        if len({tuple(tensor.shape) for tensor in embeddings}) == 1:
            return loaded
        try:
            aligner = TokenizerAligner([])
            if tokenizers:
                vocabs = []
                for payload in tokenizers:
                    vocab = None
                    tokenizer_json = payload.get("tokenizer.json") or {}
                    if isinstance(tokenizer_json, dict):
                        model = tokenizer_json.get("model") or tokenizer_json.get("tokenizer") or {}
                        vocab = model.get("vocab") if isinstance(model, dict) else None
                    if vocab is None:
                        vocab = payload.get("vocab")
                    if isinstance(vocab, dict):
                        vocabs.append(vocab)
                if len(vocabs) == len(loaded):
                    aligner.tokenizers = tokenizers
                    master = aligner.build_master_vocabulary(vocabs)
                    aligned_embeds = []
                    aligned_heads = []
                    for embed, head in zip(embeddings, heads):
                        new_indices = list(range(embed.shape[0], len(master)))
                        aligned_embeds.append(
                            aligner.initialize_new_embeddings(embed, embed.shape[0], len(master), new_indices)
                        )
                        aligned_heads.append(
                            aligner.initialize_new_embeddings(head, head.shape[0], len(master), new_indices)
                        )
                    embeddings = aligned_embeds
                    heads = aligned_heads
            else:
                target_vocab = max(tensor.shape[0] for tensor in embeddings)
                resized_embeds = []
                resized_heads = []
                for embed, head in zip(embeddings, heads):
                    resized_embeds.append(
                        TokenizerAligner([]).initialize_new_embeddings(
                            embed, embed.shape[0], target_vocab, list(range(embed.shape[0], target_vocab))
                        )
                    )
                    resized_heads.append(
                        TokenizerAligner([]).initialize_new_embeddings(
                            head, head.shape[0], target_vocab, list(range(head.shape[0], target_vocab))
                        )
                    )
                embeddings = resized_embeds
                heads = resized_heads
            for index in range(len(loaded)):
                loaded[index][embed_keys[index]] = embeddings[index]
                loaded[index][head_keys[index]] = heads[index]
        except Exception:
            return loaded
        return loaded


def merge_models(
    model_records: list[Any],
    output_dir: str | Path,
    config: MergeConfig | None = None,
    **kwargs: Any,
) -> Path:
    pipeline = MergePipeline(config=config, **kwargs)
    return pipeline.run(model_records, output_dir)
