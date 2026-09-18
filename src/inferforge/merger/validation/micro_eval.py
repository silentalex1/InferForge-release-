"""
Merge Evaluation Feedback Loops for Autonomous Model Merging.

Implements real-time runtime checking to prevent model corruption and
automatic hyperparameter tuning for optimal merge ratios.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch


class MergeQuality(Enum):
    """Quality assessment of merge attempt."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CORRUPTED = "corrupted"


@dataclass
class MergeCheckpoint:
    """Checkpoint for potential rollback."""
    layer_index: int
    weights: Dict[str, torch.Tensor]
    merge_ratio: float
    perplexity: float
    timestamp: float


@dataclass
class EvaluationResult:
    """Result of merge evaluation."""
    quality: MergeQuality
    perplexity: float
    layer_index: int
    recommendations: List[str]
    should_rollback: bool


class MergeEvaluator:
    """Real-time merge evaluation with automatic optimization."""
    
    def __init__(
        self,
        perplexity_threshold: float = 10.0,
        quality_history_size: int = 5
    ):
        """
        Initialize merge evaluator.
        
        Args:
            perplexity_threshold: Threshold for considering merge as corrupted
            quality_history_size: Number of recent evaluations to track
        """
        self.perplexity_threshold = perplexity_threshold
        self.quality_history_size = quality_history_size
        self.quality_history: List[EvaluationResult] = []
        self.checkpoints: List[MergeCheckpoint] = []
        self.best_checkpoint: Optional[MergeCheckpoint] = None
        self.best_perplexity = float('inf')
    
    def evaluate_layer_merge(
        self,
        merged_weights: Dict[str, torch.Tensor],
        baseline_weights: Dict[str, torch.Tensor],
        layer_index: int,
        calibration_data: Optional[torch.Tensor] = None
    ) -> EvaluationResult:
        """
        Evaluate quality of a layer merge using micro-benchmarks.
        
        Args:
            merged_weights: Merged layer weights
            baseline_weights: Original layer weights for comparison
            layer_index: Current layer being evaluated
            calibration_data: Optional calibration data for forward pass
        
        Returns:
            Evaluation result with quality assessment
        """
                                          
        perplexity = self._calculate_weight_perplexity(merged_weights, baseline_weights)
        
                        
        quality = self._assess_quality(perplexity)
        
                                  
        recommendations = self._generate_recommendations(perplexity, quality)
        
                                         
        should_rollback = quality == MergeQuality.CORRUPTED
        
        result = EvaluationResult(
            quality=quality,
            perplexity=perplexity,
            layer_index=layer_index,
            recommendations=recommendations,
            should_rollback=should_rollback
        )
        
                        
        self.quality_history.append(result)
        if len(self.quality_history) > self.quality_history_size:
            self.quality_history.pop(0)
        
        return result
    
    def _calculate_weight_perplexity(
        self,
        merged_weights: Dict[str, torch.Tensor],
        baseline_weights: Dict[str, torch.Tensor]
    ) -> float:
        """
        Calculate perplexity-like metric based on weight statistics.
        
        Uses weight magnitude and distribution changes as a proxy for
        model quality when full forward pass isn't available.
        """
        total_perplexity = 0.0
        weight_count = 0
        
        for key in merged_weights:
            if key not in baseline_weights:
                continue
            
            merged = merged_weights[key]
            baseline = baseline_weights[key]
            
                                       
            if merged.shape != baseline.shape:
                min_shape = torch.minimum(merged.shape, baseline.shape)
                merged = merged[:min_shape[0]] if merged.dim() == 1 else merged[:min_shape[0], :min_shape[1]]
                baseline = baseline[:min_shape[0]] if baseline.dim() == 1 else baseline[:min_shape[0], :min_shape[1]]
            
                                                
            weight_diff = merged - baseline
            relative_change = torch.abs(weight_diff) / (torch.abs(baseline) + 1e-8)
            
                                                                                         
            layer_perplexity = torch.mean(relative_change).item()
            
            total_perplexity += layer_perplexity
            weight_count += 1
        
        return total_perplexity / max(weight_count, 1)
    
    def _assess_quality(self, perplexity: float) -> MergeQuality:
        """Assess merge quality based on perplexity score."""
        if perplexity > self.perplexity_threshold * 2:
            return MergeQuality.CORRUPTED
        elif perplexity > self.perplexity_threshold:
            return MergeQuality.POOR
        elif perplexity > self.perplexity_threshold * 0.5:
            return MergeQuality.ACCEPTABLE
        elif perplexity > self.perplexity_threshold * 0.25:
            return MergeQuality.GOOD
        else:
            return MergeQuality.EXCELLENT
    
    def _generate_recommendations(
        self,
        perplexity: float,
        quality: MergeQuality
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        if quality == MergeQuality.CORRUPTED:
            recommendations.append("IMMEDIATE ROLLBACK: Merge corrupted model weights")
            recommendations.append("Reduce merge ratio significantly (try 0.1-0.2)")
            recommendations.append("Check for architecture compatibility")
        elif quality == MergeQuality.POOR:
            recommendations.append("Consider rollback and reduce merge ratio")
            recommendations.append("Try ratio 0.3-0.4 instead of current")
            recommendations.append("Enable importance masking to protect critical weights")
        elif quality == MergeQuality.ACCEPTABLE:
            recommendations.append("Current merge is acceptable but could be improved")
            recommendations.append("Try fine-tuning merge ratio around current value")
        elif quality == MergeQuality.GOOD:
            recommendations.append("Good merge quality, continue with current parameters")
        else:             
            recommendations.append("Excellent merge quality, optimal parameters")
        
        return recommendations
    
    def create_checkpoint(
        self,
        layer_index: int,
        weights: Dict[str, torch.Tensor],
        merge_ratio: float
    ) -> MergeCheckpoint:
        """
        Create a checkpoint for potential rollback.
        
        Args:
            layer_index: Current layer index
            weights: Current merged weights
            merge_ratio: Current merge ratio
        
        Returns:
            Merge checkpoint
        """
                                      
        baseline_weights = {}                                          
        perplexity = self._calculate_weight_perplexity(weights, baseline_weights)
        
        checkpoint = MergeCheckpoint(
            layer_index=layer_index,
            weights={k: v.clone() for k, v in weights.items()},
            merge_ratio=merge_ratio,
            perplexity=perplexity,
            timestamp=time.time()
        )
        
        self.checkpoints.append(checkpoint)
        
                               
        if perplexity < self.best_perplexity:
            self.best_perplexity = perplexity
            self.best_checkpoint = checkpoint
        
        return checkpoint
    
    def rollback_to_checkpoint(self, checkpoint: MergeCheckpoint) -> Dict[str, torch.Tensor]:
        """
        Rollback to a previous checkpoint.
        
        Args:
            checkpoint: Checkpoint to rollback to
        
        Returns:
            Weights from checkpoint
        """
        return {k: v.clone() for k, v in checkpoint.weights.items()}
    
    def auto_tune_merge_ratio(
        self,
        current_ratio: float,
        recent_evaluations: List[EvaluationResult]
    ) -> float:
        """
        Automatically adjust merge ratio based on recent evaluations.
        
        Uses evolutionary logic to find optimal merge parameters.
        
        Args:
            current_ratio: Current merge ratio
            recent_evaluations: Recent evaluation results
        
        Returns:
            Adjusted merge ratio
        """
        if not recent_evaluations:
            return current_ratio
        
                                   
        poor_count = sum(1 for e in recent_evaluations if e.quality in [MergeQuality.POOR, MergeQuality.CORRUPTED])
        good_count = sum(1 for e in recent_evaluations if e.quality in [MergeQuality.GOOD, MergeQuality.EXCELLENT])
        
        total = len(recent_evaluations)
        
        if poor_count > total * 0.6:
                                                             
            new_ratio = max(0.1, current_ratio * 0.5)
        elif poor_count > total * 0.3:
                                                        
            new_ratio = max(0.2, current_ratio * 0.8)
        elif good_count > total * 0.7:
                                                          
            new_ratio = min(0.9, current_ratio * 1.1)
        else:
                                               
            new_ratio = current_ratio
        
        return new_ratio
    
    def generate_per_layer_coefficients(
        self,
        num_layers: int,
        base_ratio: float = 0.5,
        pattern: str = "uniform"
    ) -> List[float]:
        """
        Generate layer-specific merge coefficients.
        
        Instead of applying a blanket blend to the whole model, different
        layers may need different ratios for optimal performance.
        
        Args:
            num_layers: Number of layers in model
            base_ratio: Base merge ratio
            pattern: Coefficient pattern ("uniform", "increasing", "decreasing", "attention-heavy")
        
        Returns:
            List of merge coefficients per layer
        """
        if pattern == "uniform":
            return [base_ratio] * num_layers
        
        elif pattern == "increasing":
                                                              
            return [base_ratio * (0.5 + 0.5 * i / num_layers) for i in range(num_layers)]
        
        elif pattern == "decreasing":
                                                              
            return [base_ratio * (1.5 - 0.5 * i / num_layers) for i in range(num_layers)]
        
        elif pattern == "attention-heavy":
                                                                         
            coefficients = []
            for i in range(num_layers):
                if i % 2 == 0:                                    
                    coefficients.append(base_ratio * 1.2)
                else:
                    coefficients.append(base_ratio * 0.8)
            return coefficients
        
        else:
            return [base_ratio] * num_layers
    
    def evaluate_final_model(
        self,
        final_weights: Dict[str, torch.Tensor],
        test_prompts: List[str] = None
    ) -> Dict[str, float]:
        """
        Final evaluation of merged model.
        
        Args:
            final_weights: Final merged model weights
            test_prompts: Optional test prompts for functional evaluation
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "weight_perplexity": 0.0,
            "parameter_stability": 0.0,
            "gradient_health": 0.0,
            "overall_score": 0.0
        }
        
                                     
        metrics["weight_perplexity"] = self._calculate_overall_perplexity(final_weights)
        
                                       
        metrics["parameter_stability"] = self._calculate_parameter_stability(final_weights)
        
                                                            
        metrics["gradient_health"] = self._calculate_gradient_health(final_weights)
        
                       
        metrics["overall_score"] = (
            metrics["weight_perplexity"] * 0.4 +
            metrics["parameter_stability"] * 0.3 +
            metrics["gradient_health"] * 0.3
        )
        
        return metrics
    
    def _calculate_overall_perplexity(self, weights: Dict[str, torch.Tensor]) -> float:
        """Calculate overall perplexity across all weights."""
        total_perplexity = 0.0
        count = 0
        
        for weight in weights.values():
            if weight.dim() > 1:
                weight_std = torch.std(weight).item()
                weight_mean = torch.abs(torch.mean(weight)).item()
                layer_perplexity = weight_std / (weight_mean + 1e-8)
                total_perplexity += layer_perplexity
                count += 1
        
        return total_perplexity / max(count, 1)
    
    def _calculate_parameter_stability(self, weights: Dict[str, torch.Tensor]) -> float:
        """Calculate parameter stability (inverse of outlier count)."""
        outlier_count = 0
        total_params = 0
        
        for weight in weights.values():
            if weight.dim() > 1:
                mean = torch.mean(weight)
                std = torch.std(weight)
                outliers = torch.abs(weight - mean) > 3 * std
                outlier_count += outliers.sum().item()
                total_params += weight.numel()
        
        stability = 1.0 - (outlier_count / max(total_params, 1))
        return max(0.0, stability)
    
    def _calculate_gradient_health(self, weights: Dict[str, torch.Tensor]) -> float:
        """Calculate gradient health (proxy using weight distribution)."""
        total_health = 0.0
        count = 0
        
        for weight in weights.values():
            if weight.dim() > 1:
                                                                               
                kurtosis = self._calculate_kurtosis(weight)
                                                                    
                health = 1.0 / (1.0 + abs(kurtosis - 3))
                total_health += health
                count += 1
        
        return total_health / max(count, 1)
    
    def _calculate_kurtosis(self, tensor: torch.Tensor) -> float:
        """Calculate kurtosis of tensor values."""
        flattened = tensor.flatten().float()
        mean = torch.mean(flattened)
        std = torch.std(flattened)
        if std < 1e-8:
            return 0.0
        standardized = (flattened - mean) / std
        kurtosis = torch.mean(standardized ** 4).item() - 3
        return kurtosis
