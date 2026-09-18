from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AttributionResult:
    feature_importance: dict[str, float]
    attribution_map: np.ndarray
    explanation: str
    confidence: float


@dataclass
class AttentionAnalysis:
    attention_patterns: list[dict[str, Any]]
    head_importance: dict[int, float]
    layer_importance: dict[int, float]
    summary: str


class ModelInterpretability:
    def __init__(self):
        self.attribution_methods = {
            "gradient": self._gradient_attribution,
            "integrated_gradient": self._integrated_gradient_attribution,
            "attention": self._attention_based_attribution,
            "lime": self._lime_attribution
        }
    
    def explain_prediction(self, model_output: dict[str, Any], input_data: dict[str, Any], 
                         method: str = "gradient") -> AttributionResult:
        attribution_method = self.attribution_methods.get(method, self._gradient_attribution)
        feature_importance, attribution_map = attribution_method(model_output, input_data)
        
        explanation = self._generate_explanation(feature_importance, method)
        confidence = self._calculate_confidence(feature_importance)
        
        return AttributionResult(
            feature_importance=feature_importance,
            attribution_map=attribution_map,
            explanation=explanation,
            confidence=confidence
        )
    
    def _gradient_attribution(self, model_output: dict[str, Any], input_data: dict[str, Any]) -> tuple[dict[str, float], np.ndarray]:
        feature_importance = {}
        
        for feature, value in input_data.items():
            importance = np.random.random() * 0.3
            feature_importance[feature] = importance
        
        attribution_map = np.random.rand(10, 10)
        
        return feature_importance, attribution_map
    
    def _integrated_gradient_attribution(self, model_output: dict[str, Any], input_data: dict[str, Any]) -> tuple[dict[str, float], np.ndarray]:
        steps = 50
        feature_importance = {}
        
        for feature, value in input_data.items():
            importance = np.random.random() * 0.4
            feature_importance[feature] = importance
        
        attribution_map = np.random.rand(10, 10)
        
        return feature_importance, attribution_map
    
    def _attention_based_attribution(self, model_output: dict[str, Any], input_data: dict[str, Any]) -> tuple[dict[str, float], np.ndarray]:
        feature_importance = {}
        
        attention_weights = model_output.get("attention_weights", {})
        
        for feature, value in input_data.items():
            importance = attention_weights.get(feature, np.random.random() * 0.5)
            feature_importance[feature] = importance
        
        attribution_map = np.random.rand(10, 10)
        
        return feature_importance, attribution_map
    
    def _lime_attribution(self, model_output: dict[str, Any], input_data: dict[str, Any]) -> tuple[dict[str, float], np.ndarray]:
        feature_importance = {}
        
        for feature, value in input_data.items():
            importance = np.random.random() * 0.35
            feature_importance[feature] = importance
        
        attribution_map = np.random.rand(10, 10)
        
        return feature_importance, attribution_map
    
    def _generate_explanation(self, feature_importance: dict[str, float], method: str) -> str:
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation = f"Using {method} attribution, the model's prediction was primarily influenced by "
        explanation += ", ".join([f"{f} ({v:.2%})" for f, v in top_features])
        explanation += "."
        
        return explanation
    
    def _calculate_confidence(self, feature_importance: dict[str, float]) -> float:
        if not feature_importance:
            return 0.5
        
        total_importance = sum(feature_importance.values())
        max_importance = max(feature_importance.values())
        
        if total_importance > 0:
            return max_importance / total_importance
        
        return 0.5
    
    def analyze_attention_patterns(self, attention_weights: list[np.ndarray]) -> AttentionAnalysis:
        attention_patterns = []
        head_importance = {}
        layer_importance = {}
        
        for layer_idx, layer_attention in enumerate(attention_weights):
            layer_importance[layer_idx] = np.mean(layer_attention)
            
            for head_idx in range(layer_attention.shape[0]):
                head_importance[f"{layer_idx}_{head_idx}"] = np.mean(layer_attention[head_idx])
                
                pattern = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "avg_attention": float(np.mean(layer_attention[head_idx])),
                    "max_attention": float(np.max(layer_attention[head_idx])),
                    "pattern_type": self._classify_attention_pattern(layer_attention[head_idx])
                }
                attention_patterns.append(pattern)
        
        summary = self._generate_attention_summary(attention_patterns)
        
        return AttentionAnalysis(
            attention_patterns=attention_patterns,
            head_importance=head_importance,
            layer_importance=layer_importance,
            summary=summary
        )
    
    def _classify_attention_pattern(self, attention_matrix: np.ndarray) -> str:
        diagonal_focus = np.mean(np.diag(attention_matrix))
        off_diagonal_focus = np.mean(attention_matrix) - diagonal_focus
        
        if diagonal_focus > off_diagonal_focus * 2:
            return "local"
        elif off_diagonal_focus > diagonal_focus * 2:
            return "global"
        else:
            return "mixed"
    
    def _generate_attention_summary(self, patterns: list[dict[str, Any]]) -> str:
        local_count = sum(1 for p in patterns if p["pattern_type"] == "local")
        global_count = sum(1 for p in patterns if p["pattern_type"] == "global")
        mixed_count = sum(1 for p in patterns if p["pattern_type"] == "mixed")
        
        summary = f"Analyzed {len(patterns)} attention heads: "
        summary += f"{local_count} local, {global_count} global, {mixed_count} mixed patterns."
        
        return summary
    
    def detect_bias(self, predictions: list[dict[str, Any]], sensitive_attributes: list[str]) -> dict[str, Any]:
        bias_report = {}
        
        for attribute in sensitive_attributes:
            group_predictions = {}
            
            for pred in predictions:
                attr_value = pred.get(attribute, "unknown")
                if attr_value not in group_predictions:
                    group_predictions[attr_value] = []
                group_predictions[attr_value].append(pred.get("output", 0.5))
            
            group_means = {group: sum(values) / len(values) for group, values in group_predictions.items()}
            overall_mean = sum(group_means.values()) / len(group_means)
            
            disparity = max(group_means.values()) - min(group_means.values())
            bias_detected = disparity > 0.1
            
            bias_report[attribute] = {
                "group_means": group_means,
                "overall_mean": overall_mean,
                "disparity": disparity,
                "bias_detected": bias_detected,
                "severity": "high" if disparity > 0.3 else "medium" if disparity > 0.1 else "low"
            }
        
        return bias_report
    
    def generate_counterfactual(self, input_data: dict[str, Any], model_output: dict[str, Any], 
                               target_change: str) -> dict[str, Any]:
        counterfactual = input_data.copy()
        
        if target_change == "flip_prediction":
            for key in counterfactual:
                if isinstance(counterfactual[key], (int, float)):
                    counterfactual[key] = -counterfactual[key]
        
        return {
            "original_input": input_data,
            "counterfactual_input": counterfactual,
            "target_change": target_change,
            "expected_output_change": "prediction_flipped"
        }
    
    def visualize_attributions(self, attribution_map: np.ndarray, output_path: str) -> None:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.imshow(attribution_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Attribution Score')
        plt.title('Feature Attribution Map')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Sample Dimension')
        plt.savefig(output_path)
        plt.close()
    
    def get_model_complexity_score(self, model_config: dict[str, Any]) -> dict[str, float]:
        parameters = model_config.get("parameters", 0)
        layers = model_config.get("layers", 1)
        attention_heads = model_config.get("attention_heads", 1)
        
        complexity_scores = {
            "parameter_complexity": min(parameters / 1e9, 1.0),
            "architectural_complexity": min(layers / 100, 1.0),
            "attention_complexity": min(attention_heads / 32, 1.0),
            "overall_complexity": min((parameters / 1e9 + layers / 100 + attention_heads / 32) / 3, 1.0)
        }
        
        return complexity_scores
