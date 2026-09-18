from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InitializationConfig:
    method: str
    scale: float
    pretrained_patterns: bool
    language_aware: bool
    reasoning_aware: bool


@dataclass
class InitializationResult:
    weights: dict[str, np.ndarray]
    method_used: str
    quality_score: float


class IntelligentInitialization:
    def __init__(self):
        self.language_patterns = self._load_language_patterns()
        self.reasoning_patterns = self._load_reasoning_patterns()
        self.init_methods = {
            "xavier": self._xavier_init,
            "he": self._he_init,
            "kaiming": self._kaiming_init,
            "language_aware": self._language_aware_init,
            "reasoning_aware": self._reasoning_aware_init,
            "meta_learning": self._meta_learning_init
        }
    
    def _load_language_patterns(self) -> dict[str, np.ndarray]:
        return {
            "attention_bias": np.array([0.1, -0.1, 0.2, -0.2]),
            "ffn_gate": np.array([0.5, 0.3, 0.7, 0.1]),
            "layer_norm_scale": np.ones(4)
        }
    
    def _load_reasoning_patterns(self) -> dict[str, np.ndarray]:
        return {
            "reasoning_gate": np.array([0.8, 0.6, 0.9, 0.4]),
            "memory_bias": np.array([0.2, 0.1, 0.3, 0.05]),
            "logic_scale": np.array([1.2, 1.1, 1.3, 1.0])
        }
    
    def initialize_weights(self, architecture: dict[str, tuple[int, ...]], config: InitializationConfig) -> InitializationResult:
        if config.method == "auto":
            config.method = self._select_best_method(architecture, config)
        
        weights = {}
        
        for layer_name, shape in architecture.items():
            if "attention" in layer_name:
                weights[layer_name] = self._init_attention_weights(shape, config)
            elif "ffn" in layer_name or "feedforward" in layer_name:
                weights[layer_name] = self._init_ffn_weights(shape, config)
            elif "embedding" in layer_name:
                weights[layer_name] = self._init_embedding_weights(shape, config)
            elif "layer_norm" in layer_name:
                weights[layer_name] = self._init_layer_norm_weights(shape, config)
            else:
                weights[layer_name] = self._init_generic_weights(shape, config)
        
        quality_score = self._evaluate_initialization_quality(weights)
        
        return InitializationResult(
            weights=weights,
            method_used=config.method,
            quality_score=quality_score
        )
    
    def _select_best_method(self, architecture: dict[str, tuple[int, ...]], config: InitializationConfig) -> str:
        has_attention = any("attention" in name for name in architecture.keys())
        has_reasoning = any("reasoning" in name or "logic" in name for name in architecture.keys())
        
        if config.reasoning_aware and has_reasoning:
            return "reasoning_aware"
        elif config.language_aware and has_attention:
            return "language_aware"
        elif config.pretrained_patterns:
            return "meta_learning"
        else:
            return "kaiming"
    
    def _init_attention_weights(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        if config.language_aware:
            return self._language_aware_init(shape, config)
        return self._kaiming_init(shape, config)
    
    def _init_ffn_weights(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        return self._kaiming_init(shape, config)
    
    def _init_embedding_weights(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        return self._xavier_init(shape, config)
    
    def _init_layer_norm_weights(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        scale = np.ones(shape[0])
        bias = np.zeros(shape[0])
        return np.stack([scale, bias])
    
    def _init_generic_weights(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        return self.init_methods.get(config.method, self._kaiming_init)(shape, config)
    
    def _xavier_init(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        fan_in, fan_out = self._calculate_fan_in_out(shape)
        scale = config.scale * np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * scale
    
    def _he_init(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        fan_in, _ = self._calculate_fan_in_out(shape)
        scale = config.scale * np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * scale
    
    def _kaiming_init(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        return self._he_init(shape, config)
    
    def _language_aware_init(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        base_weights = self._kaiming_init(shape, config)
        
        if len(shape) == 2 and shape[0] == shape[1]:
            pattern = self.language_patterns["attention_bias"]
            if len(pattern) <= min(shape):
                for i in range(len(pattern)):
                    base_weights[i, i] += pattern[i] * config.scale
        
        return base_weights
    
    def _reasoning_aware_init(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        base_weights = self._kaiming_init(shape, config)
        
        if len(shape) == 2:
            pattern = self.reasoning_patterns["reasoning_gate"]
            if len(pattern) <= min(shape):
                for i in range(len(pattern)):
                    base_weights[i, :] *= pattern[i] * config.scale
        
        return base_weights
    
    def _meta_learning_init(self, shape: tuple[int, ...], config: InitializationConfig) -> np.ndarray:
        base_weights = self._kaiming_init(shape, config)
        
        if len(shape) == 2:
            for i in range(min(shape[0], 10)):
                base_weights[i, :] *= 1.2
                base_weights[:, i] *= 1.1
        
        return base_weights
    
    def _calculate_fan_in_out(self, shape: tuple[int, ...]) -> tuple[int, int]:
        if len(shape) == 2:
            fan_in, fan_out = shape[1], shape[0]
        elif len(shape) == 4:
            fan_in = shape[1] * shape[2] * shape[3]
            fan_out = shape[0] * shape[2] * shape[3]
        else:
            fan_in = fan_out = np.prod(shape)
        
        return fan_in, fan_out
    
    def _evaluate_initialization_quality(self, weights: dict[str, np.ndarray]) -> float:
        score = 0.0
        
        for w in weights.values():
            mean = np.mean(w)
            std = np.std(w)
            
            if abs(mean) < 0.1:
                score += 0.3
            if 0.05 < std < 1.0:
                score += 0.3
            if not np.any(np.isnan(w)) and not np.any(np.isinf(w)):
                score += 0.4
        
        return score / len(weights) if weights else 0.0
    
    def warm_start_from_patterns(self, base_weights: dict[str, np.ndarray], task_type: str) -> dict[str, np.ndarray]:
        warmed = {}
        
        for name, w in base_weights.items():
            if task_type == "reasoning" and "attention" in name:
                warmed[name] = w * 1.2
            elif task_type == "coding" and "ffn" in name:
                warmed[name] = w * 1.1
            else:
                warmed[name] = w
        
        return warmed
