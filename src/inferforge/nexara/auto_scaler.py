"""Automatic scaling for premium distributed training."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from inferforge.core.premium import Tier, get_premium_manager


class ScalingStrategy(Enum):
    """Automatic scaling strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class ScalingMetrics:
    """Current training metrics for scaling decisions."""
    current_loss: float = 0.0
    loss_trend: str = "stable"
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    estimated_time_remaining: float = 0.0
    cost_per_hour: float = 0.0


@dataclass
class ScalingConfig:
    """Configuration for automatic scaling."""
    strategy: ScalingStrategy = ScalingStrategy.BALANCED
    min_nodes: int = 1
    max_nodes: int = 8
    target_gpu_utilization: float = 0.85
    target_memory_usage: float = 0.80
    scale_up_threshold: float = 0.90
    scale_down_threshold: float = 0.60
    cooldown_seconds: int = 300
    max_cost_per_hour: float = 10.0
    priority_weight: float = 1.0


@dataclass
class ScalingDecision:
    """Decision made by auto-scaler."""
    action: str  # "scale_up", "scale_down", "maintain", "terminate"
    target_nodes: int
    reason: str
    estimated_cost_impact: float
    estimated_time_impact: float


class AutoScaler:
    """Automatic scaling engine for distributed training."""
    
    def __init__(self, config: ScalingConfig | None = None):
        self.config = config or ScalingConfig()
        self.premium_manager = get_premium_manager()
        self.metrics_history: list[ScalingMetrics] = []
        self.last_scale_time = 0.0
        self.current_nodes = 1
        self.scaling_history: list[ScalingDecision] = []
    
    def _get_max_allowed_nodes(self) -> int:
        """Get maximum nodes allowed by premium tier."""
        limits = self.premium_manager.get_limits()
        return min(self.config.max_nodes, limits.max_gpu_nodes)
    
    def _get_priority_multiplier(self) -> float:
        """Get priority multiplier for scaling decisions."""
        return self.premium_manager.get_priority_score()
    
    def _calculate_utilization_score(self, metrics: ScalingMetrics) -> float:
        """Calculate overall utilization score (0-1)."""
        gpu_score = metrics.gpu_utilization / 100.0
        memory_score = metrics.memory_usage / 100.0
        
        # Weighted average
        return (gpu_score * 0.6 + memory_score * 0.4)
    
    def _estimate_optimal_nodes(self, metrics: ScalingMetrics) -> int:
        """Estimate optimal number of nodes based on metrics."""
        utilization = self._calculate_utilization_score(metrics)
        target_util = self.config.target_gpu_utilization
        
        if utilization <= 0:
            return self.config.min_nodes
        
        # Simple scaling: nodes = current_nodes * (current_util / target_util)
        optimal = math.ceil(self.current_nodes * (utilization / target_util))
        
        # Apply strategy modifiers
        if self.config.strategy == ScalingStrategy.CONSERVATIVE:
            optimal = max(self.config.min_nodes, optimal - 1)
        elif self.config.strategy == ScalingStrategy.AGGRESSIVE:
            optimal = optimal + 1
        
        # Clamp to allowed range
        max_nodes = self._get_max_allowed_nodes()
        return max(self.config.min_nodes, min(optimal, max_nodes))
    
    def _check_cooldown(self) -> bool:
        """Check if cooldown period has passed."""
        return time.time() - self.last_scale_time >= self.config.cooldown_seconds
    
    def _estimate_cost_impact(self, current_nodes: int, target_nodes: int) -> float:
        """Estimate cost impact of scaling."""
        # Assume $2/hour per GPU node (adjust based on actual cloud pricing)
        cost_per_node = 2.0
        return (target_nodes - current_nodes) * cost_per_node
    
    def _estimate_time_impact(self, metrics: ScalingMetrics, current_nodes: int, target_nodes: int) -> float:
        """Estimate time impact of scaling."""
        if current_nodes == 0 or target_nodes == 0:
            return 0.0
        
        # Assume linear scaling (simplified)
        speedup = target_nodes / current_nodes
        current_eta = metrics.estimated_time_remaining
        return current_eta - (current_eta / speedup)
    
    def should_scale(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Determine if scaling is needed."""
        if not self._check_cooldown():
            return ScalingDecision(
                action="maintain",
                target_nodes=self.current_nodes,
                reason="Cooldown period active",
                estimated_cost_impact=0.0,
                estimated_time_impact=0.0,
            )
        
        max_nodes = self._get_max_allowed_nodes()
        if max_nodes <= 1:
            return ScalingDecision(
                action="maintain",
                target_nodes=self.current_nodes,
                reason="Multi-node training not available in current tier",
                estimated_cost_impact=0.0,
                estimated_time_impact=0.0,
            )
        
        utilization = self._calculate_utilization_score(metrics)
        optimal_nodes = self._estimate_optimal_nodes(metrics)
        
        # Check cost constraints
        estimated_cost = optimal_nodes * 2.0  # $2/hour per node
        if estimated_cost > self.config.max_cost_per_hour:
            optimal_nodes = max(self.config.min_nodes, int(self.config.max_cost_per_hour / 2.0))
        
        # Apply priority weighting
        priority_mult = self._get_priority_multiplier()
        if priority_mult > 1.0:
            # Premium users get more aggressive scaling
            optimal_nodes = min(optimal_nodes + 1, max_nodes)
        
        # Make decision
        if optimal_nodes > self.current_nodes and utilization > self.config.scale_up_threshold:
            cost_impact = self._estimate_cost_impact(self.current_nodes, optimal_nodes)
            time_impact = self._estimate_time_impact(metrics, self.current_nodes, optimal_nodes)
            
            decision = ScalingDecision(
                action="scale_up",
                target_nodes=optimal_nodes,
                reason=f"High utilization ({utilization:.1%}) - scaling to {optimal_nodes} nodes",
                estimated_cost_impact=cost_impact,
                estimated_time_impact=time_impact,
            )
            
        elif optimal_nodes < self.current_nodes and utilization < self.config.scale_down_threshold:
            cost_impact = self._estimate_cost_impact(self.current_nodes, optimal_nodes)
            time_impact = self._estimate_time_impact(metrics, self.current_nodes, optimal_nodes)
            
            decision = ScalingDecision(
                action="scale_down",
                target_nodes=optimal_nodes,
                reason=f"Low utilization ({utilization:.1%}) - scaling to {optimal_nodes} nodes",
                estimated_cost_impact=cost_impact,
                estimated_time_impact=time_impact,
            )
            
        else:
            decision = ScalingDecision(
                action="maintain",
                target_nodes=self.current_nodes,
                reason=f"Utilization within target range ({utilization:.1%})",
                estimated_cost_impact=0.0,
                estimated_time_impact=0.0,
            )
        
        self.scaling_history.append(decision)
        return decision
    
    def apply_scaling(self, decision: ScalingDecision, progress_callback: Callable | None = None) -> bool:
        """Apply scaling decision."""
        if decision.action == "maintain":
            return True
        
        if not self.premium_manager.has_feature("distributed_training"):
            if progress_callback:
                progress_callback("Distributed training not available in current tier", 0.0)
            return False
        
        # In a real implementation, this would interact with cloud APIs
        # For now, we simulate the scaling
        self.current_nodes = decision.target_nodes
        self.last_scale_time = time.time()
        
        if progress_callback:
            progress_callback(f"Scaled to {decision.target_nodes} nodes: {decision.reason}", 1.0)
        
        return True
    
    def get_scaling_report(self) -> dict[str, Any]:
        """Get comprehensive scaling report."""
        return {
            "current_nodes": self.current_nodes,
            "max_allowed_nodes": self._get_max_allowed_nodes(),
            "strategy": self.config.strategy.value,
            "scaling_history": [
                {
                    "action": d.action,
                    "target_nodes": d.target_nodes,
                    "reason": d.reason,
                    "cost_impact": d.estimated_cost_impact,
                    "time_impact": d.estimated_time_impact,
                }
                for d in self.scaling_history[-10:]
            ],
            "premium_tier": self.premium_manager.get_current_tier().value,
            "priority_multiplier": self._get_priority_multiplier(),
        }


class PremiumDistributedTrainer:
    """Enhanced distributed trainer with premium features."""
    
    def __init__(self, auto_scaler: AutoScaler | None = None):
        self.auto_scaler = auto_scaler or AutoScaler()
        self.premium_manager = get_premium_manager()
        self._distributed_trainer = None
    
    def setup_premium_distributed(self, num_nodes: int | None = None) -> bool:
        """Setup distributed training with premium features."""
        if not self.premium_manager.has_feature("distributed_training"):
            return False
        
        if num_nodes is None:
            num_nodes = self.auto_scaler.current_nodes
        
        max_nodes = self.premium_manager.get_limits().max_gpu_nodes
        num_nodes = min(num_nodes, max_nodes)
        
        # Import and setup distributed trainer
        try:
            from inferforge.nexara.distributed_training import DistributedConfig, DistributedTrainer
            
            config = DistributedConfig(
                world_size=num_nodes,
                use_fsdp=self.premium_manager.has_feature("distributed_training"),
                use_ddp=True,
                mixed_precision=self.premium_manager.has_feature("mixed_precision"),
            )
            
            self._distributed_trainer = DistributedTrainer(config)
            return True
            
        except Exception as e:
            print(f"Failed to setup distributed training: {e}")
            return False
    
    def train_with_auto_scaling(
        self,
        train_fn: Callable,
        metrics_callback: Callable,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Train with automatic scaling enabled."""
        if not self.premium_manager.has_feature("automatic_scaling"):
            # Fall back to regular training
            return train_fn()
        
        scaling_decisions = []
        total_cost_savings = 0.0
        total_time_savings = 0.0
        
        def wrapped_metrics_callback(metrics: ScalingMetrics):
            nonlocal total_cost_savings, total_time_savings
            # Check if scaling is needed
            decision = self.auto_scaler.should_scale(metrics)
            scaling_decisions.append(decision)
            
            if decision.action != "maintain":
                # Apply scaling
                if self.auto_scaler.apply_scaling(decision, progress_callback):
                    total_cost_savings += abs(decision.estimated_cost_impact)
                    total_time_savings += abs(decision.estimated_time_impact)
            
            # Call original callback
            return metrics_callback(metrics)
        
        # Run training
        result = train_fn(wrapped_metrics_callback)
        
        # Add scaling metrics to result
        result["scaling_metrics"] = {
            "decisions_made": len([d for d in scaling_decisions if d.action != "maintain"]),
            "total_cost_savings": total_cost_savings,
            "total_time_savings": total_time_savings,
            "final_nodes": self.auto_scaler.current_nodes,
            "scaling_report": self.auto_scaler.get_scaling_report(),
        }
        
        return result
