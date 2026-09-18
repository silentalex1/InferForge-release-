from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MergeConfig:
    method: str
    weights: list[float]
    layer_mapping: dict[str, str]
    interpolation: str


@dataclass
class MergeResult:
    merged_weights: dict[str, np.ndarray]
    method_used: str
    performance_estimate: float
    conflicts_resolved: int


class ModelMergingEngine:
    def __init__(self):
        self.merge_methods = {
            "linear": self._linear_merge,
            "weighted_average": self._weighted_average,
            "frankenstein": self._frankenstein_merge,
            "task_arithmetic": self._task_arithmetic_merge,
            "ties_merging": self._ties_merging,
            "dare": self._dare_merge
        }
    
    def merge_models(self, models: list[dict[str, np.ndarray]], config: MergeConfig) -> MergeResult:
        if len(models) < 2:
            raise ValueError("At least 2 models required for merging")
        
        method = self.merge_methods.get(config.method, self._weighted_average)
        merged_weights, conflicts = method(models, config)
        
        performance_estimate = self._estimate_merged_performance(models, merged_weights)
        
        return MergeResult(
            merged_weights=merged_weights,
            method_used=config.method,
            performance_estimate=performance_estimate,
            conflicts_resolved=conflicts
        )
    
    def _linear_merge(self, models: list[dict[str, np.ndarray]], config: MergeConfig) -> tuple[dict[str, np.ndarray], int]:
        merged = {}
        conflicts = 0
        weights = config.weights or [1.0 / len(models)] * len(models)
        
        all_layer_names = set()
        for model in models:
            all_layer_names.update(model.keys())
        
        for layer_name in all_layer_names:
            layer_weights = []
            for i, model in enumerate(models):
                if layer_name in model:
                    layer_weights.append(model[layer_name] * weights[i])
            
            if layer_weights:
                merged[layer_name] = sum(layer_weights)
            else:
                conflicts += 1
        
        return merged, conflicts
    
    def _weighted_average(self, models: list[dict[str, np.ndarray]], config: MergeConfig) -> tuple[dict[str, np.ndarray], int]:
        merged = {}
        conflicts = 0
        weights = config.weights or [1.0 / len(models)] * len(models)
        
        all_layer_names = set()
        for model in models:
            all_layer_names.update(model.keys())
        
        for layer_name in all_layer_names:
            layer_weights = []
            total_weight = 0.0
            
            for i, model in enumerate(models):
                if layer_name in model:
                    layer_weights.append(model[layer_name] * weights[i])
                    total_weight += weights[i]
            
            if layer_weights and total_weight > 0:
                merged[layer_name] = sum(layer_weights) / total_weight
            else:
                conflicts += 1
        
        return merged, conflicts
    
    def _frankenstein_merge(self, models: list[dict[str, np.ndarray]], config: MergeConfig) -> tuple[dict[str, np.ndarray], int]:
        merged = {}
        conflicts = 0
        
        layer_mapping = config.layer_mapping or {}
        
        for layer_name in layer_mapping.keys():
            source_model_idx = int(layer_mapping[layer_name])
            if source_model_idx < len(models) and layer_name in models[source_model_idx]:
                merged[layer_name] = models[source_model_idx][layer_name].copy()
            else:
                conflicts += 1
        
        for layer_name in models[0].keys():
            if layer_name not in merged:
                merged[layer_name] = models[0][layer_name].copy()
        
        return merged, conflicts
    
    def _task_arithmetic_merge(self, models: list[dict[str, np.ndarray]], config: MergeConfig) -> tuple[dict[str, np.ndarray], int]:
        base_model = models[0]
        task_vectors = []
        
        for i in range(1, len(models)):
            task_vector = {}
            for layer_name in base_model.keys():
                if layer_name in models[i]:
                    task_vector[layer_name] = models[i][layer_name] - base_model[layer_name]
            task_vectors.append(task_vector)
        
        merged = {}
        conflicts = 0
        weights = config.weights or [1.0 / len(task_vectors)] * len(task_vectors)
        
        for layer_name in base_model.keys():
            merged[layer_name] = base_model[layer_name].copy()
            
            for i, task_vector in enumerate(task_vectors):
                if layer_name in task_vector:
                    merged[layer_name] += task_vector[layer_name] * weights[i]
        
        return merged, conflicts
    
    def _ties_merging(self, models: list[dict[str, np.ndarray]], config: MergeConfig) -> tuple[dict[str, np.ndarray], int]:
        merged = {}
        conflicts = 0
        top_k_ratio = 0.2
        
        all_layer_names = set()
        for model in models:
            all_layer_names.update(model.keys())
        
        for layer_name in all_layer_names:
            layer_weights = []
            for model in models:
                if layer_name in model:
                    layer_weights.append(model[layer_name])
            
            if not layer_weights:
                conflicts += 1
                continue
            
            stacked = np.stack(layer_weights)
            
            if len(stacked.shape) == 2:
                for i in range(stacked.shape[0]):
                    top_k = int(stacked.shape[1] * top_k_ratio)
                    indices = np.argsort(np.abs(stacked[i]))[-top_k:]
                    mask = np.zeros_like(stacked[i])
                    mask[indices] = 1
                    stacked[i] *= mask
            
            merged[layer_name] = np.mean(stacked, axis=0)
        
        return merged, conflicts
    
    def _dare_merge(self, models: list[dict[str, np.ndarray]], config: MergeConfig) -> tuple[dict[str, np.ndarray], int]:
        merged = {}
        conflicts = 0
        drop_rate = 0.5
        
        all_layer_names = set()
        for model in models:
            all_layer_names.update(model.keys())
        
        for layer_name in all_layer_names:
            layer_weights = []
            for model in models:
                if layer_name in model:
                    layer_weights.append(model[layer_name])
            
            if not layer_weights:
                conflicts += 1
                continue
            
            stacked = np.stack(layer_weights)
            
            mask = np.random.binomial(1, 1 - drop_rate, size=stacked.shape)
            stacked = stacked * mask / (1 - drop_rate)
            
            merged[layer_name] = np.mean(stacked, axis=0)
        
        return merged, conflicts
    
    def _estimate_merged_performance(self, models: list[dict[str, np.ndarray]], merged: dict[str, np.ndarray]) -> float:
        base_scores = [0.8, 0.75, 0.7][:len(models)]
        avg_base = sum(base_scores) / len(base_scores)
        
        diversity_score = self._calculate_diversity(models)
        
        merged_estimate = avg_base + (diversity_score * 0.1)
        
        return min(merged_estimate, 0.95)
    
    def _calculate_diversity(self, models: list[dict[str, np.ndarray]]) -> float:
        if len(models) < 2:
            return 0.0
        
        total_diversity = 0.0
        comparisons = 0
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                common_layers = set(models[i].keys()) & set(models[j].keys())
                
                for layer in common_layers:
                    diff = np.mean(np.abs(models[i][layer] - models[j][layer]))
                    total_diversity += diff
                    comparisons += 1
        
        return total_diversity / comparisons if comparisons > 0 else 0.0
    
    def create_ensemble(self, models: list[dict[str, np.ndarray]], voting_strategy: str = "weighted") -> dict[str, Any]:
        return {
            "models_count": len(models),
            "voting_strategy": voting_strategy,
            "ensemble_type": "soft_voting",
            "weights": [1.0 / len(models)] * len(models),
            "config": {
                "temperature": 0.8,
                "top_p": 0.95,
                "diversity_penalty": 0.1
            }
        }
    
    def optimize_merge_config(self, models: list[dict[str, np.ndarray]]) -> MergeConfig:
        diversity = self._calculate_diversity(models)
        
        if diversity > 0.5:
            method = "ties_merging"
        elif diversity > 0.3:
            method = "task_arithmetic"
        else:
            method = "weighted_average"
        
        return MergeConfig(
            method=method,
            weights=[1.0 / len(models)] * len(models),
            layer_mapping={},
            interpolation="linear"
        )
