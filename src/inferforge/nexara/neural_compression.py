from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CompressionResult:
    original_size: int
    compressed_size: int
    compression_ratio: float
    accuracy_preserved: float
    method: str


class NeuralCompressionEngine:
    def __init__(self):
        self.compression_methods = {
            "pruning": self._prune_weights,
            "quantization": self._quantize_weights,
            "knowledge_distillation": self._distill_knowledge,
            "low_rank": self._low_rank_approximation
        }
    
    def compress_model(self, weights: dict[str, np.ndarray], target_ratio: float = 0.5, method: str = "auto") -> CompressionResult:
        original_size = sum(w.nbytes for w in weights.values())
        
        if method == "auto":
            method = self._select_best_method(weights, target_ratio)
        
        compressed_weights = self.compression_methods[method](weights, target_ratio)
        compressed_size = sum(w.nbytes for w in compressed_weights.values())
        
        accuracy_preserved = self._estimate_accuracy_preservation(weights, compressed_weights)
        
        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            accuracy_preserved=accuracy_preserved,
            method=method
        )
    
    def _select_best_method(self, weights: dict[str, np.ndarray], target_ratio: float) -> str:
        avg_sparsity = self._calculate_sparsity(weights)
        
        if avg_sparsity > 0.3:
            return "pruning"
        elif target_ratio < 0.3:
            return "quantization"
        elif target_ratio < 0.6:
            return "low_rank"
        else:
            return "knowledge_distillation"
    
    def _calculate_sparsity(self, weights: dict[str, np.ndarray]) -> float:
        total_zeros = 0
        total_elements = 0
        
        for w in weights.values():
            total_zeros += np.sum(np.abs(w) < 1e-6)
            total_elements += w.size
        
        return total_zeros / total_elements if total_elements > 0 else 0.0
    
    def _prune_weights(self, weights: dict[str, np.ndarray], target_ratio: float) -> dict[str, np.ndarray]:
        pruned = {}
        threshold = self._calculate_pruning_threshold(weights, target_ratio)
        
        for name, w in weights.items():
            mask = np.abs(w) > threshold
            pruned[name] = w * mask
        
        return pruned
    
    def _calculate_pruning_threshold(self, weights: dict[str, np.ndarray], target_ratio: float) -> float:
        all_weights = np.concatenate([w.flatten() for w in weights.values()])
        sorted_weights = np.sort(np.abs(all_weights))
        threshold_idx = int(len(sorted_weights) * target_ratio)
        return sorted_weights[threshold_idx]
    
    def _quantize_weights(self, weights: dict[str, np.ndarray], target_ratio: float) -> dict[str, np.ndarray]:
        quantized = {}
        
        for name, w in weights.items():
            if target_ratio < 0.25:
                quantized[name] = self._quantize_to_int4(w)
            elif target_ratio < 0.5:
                quantized[name] = self._quantize_to_int8(w)
            else:
                quantized[name] = self._quantize_to_fp16(w)
        
        return quantized
    
    def _quantize_to_int4(self, weights: np.ndarray) -> np.ndarray:
        scale = np.max(np.abs(weights)) / 7.0
        quantized = np.round(weights / scale).astype(np.int8)
        return quantized.astype(np.int8)
    
    def _quantize_to_int8(self, weights: np.ndarray) -> np.ndarray:
        scale = np.max(np.abs(weights)) / 127.0
        quantized = np.round(weights / scale).astype(np.int8)
        return quantized.astype(np.int8)
    
    def _quantize_to_fp16(self, weights: np.ndarray) -> np.ndarray:
        return weights.astype(np.float16)
    
    def _distill_knowledge(self, weights: dict[str, np.ndarray], target_ratio: float) -> dict[str, np.ndarray]:
        distilled = {}
        
        for name, w in weights.items():
            if len(w.shape) == 2:
                U, S, V = np.linalg.svd(w, full_matrices=False)
                k = int(min(w.shape) * target_ratio)
                distilled[name] = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
            else:
                distilled[name] = w
        
        return distilled
    
    def _low_rank_approximation(self, weights: dict[str, np.ndarray], target_ratio: float) -> dict[str, np.ndarray]:
        approximated = {}
        
        for name, w in weights.items():
            if len(w.shape) == 2:
                rank = int(min(w.shape) * target_ratio)
                approximated[name] = self._low_rank_svd(w, rank)
            else:
                approximated[name] = w
        
        return approximated
    
    def _low_rank_svd(self, matrix: np.ndarray, rank: int) -> np.ndarray:
        U, S, V = np.linalg.svd(matrix, full_matrices=False)
        return U[:, :rank] @ np.diag(S[:rank]) @ V[:rank, :]
    
    def _estimate_accuracy_preservation(self, original: dict[str, np.ndarray], compressed: dict[str, np.ndarray]) -> float:
        total_error = 0.0
        total_norm = 0.0
        
        for name in original.keys():
            if name in compressed:
                error = np.linalg.norm(original[name] - compressed[name])
                norm = np.linalg.norm(original[name])
                total_error += error
                total_norm += norm
        
        return 1.0 - (total_error / total_norm) if total_norm > 0 else 0.0
    
    def dynamic_neural_activation(self, activations: dict[str, np.ndarray], threshold: float = 0.1) -> dict[str, np.ndarray]:
        activated = {}
        
        for name, activation in activations.items():
            importance = np.mean(np.abs(activation))
            if importance > threshold:
                activated[name] = activation
        
        return activated
