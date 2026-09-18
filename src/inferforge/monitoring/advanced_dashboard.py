"""Advanced monitoring dashboard with ML insights for premium users."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from inferforge.core.premium import get_premium_manager


class InsightType(Enum):
    """Types of ML insights."""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    RESOURCE = "resource"
    COST = "cost"
    ANOMALY = "anomaly"


@dataclass
class TrainingMetric:
    """Individual training metric."""
    timestamp: float
    step: int
    epoch: int
    loss: float
    learning_rate: float
    gradient_norm: float
    gpu_utilization: float
    memory_usage: float
    throughput: float
    batch_size: int
    sequence_length: int


@dataclass
class MLInsight:
    """ML-generated insight about training."""
    insight_type: InsightType
    severity: str  # info, warning, critical
    title: str
    description: str
    recommendation: str
    confidence: float
    metrics: dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AnomalyDetection:
    """Detected anomaly in training metrics."""
    anomaly_type: str
    severity: str
    detected_at: float
    metrics: dict[str, float]
    expected_range: tuple[float, float]
    actual_value: float
    context: str


class AdvancedMetricsCollector:
    """Collect and analyze advanced training metrics."""
    
    def __init__(self):
        self.premium_manager = get_premium_manager()
        self.metrics_history: list[TrainingMetric] = []
        self.insights: list[MLInsight] = []
        self.anomalies: list[AnomalyDetection] = []
        self.baseline_metrics: dict[str, float] = {}
        self._max_history_size = 10000
    
    def collect_metric(self, metric: TrainingMetric) -> None:
        """Collect a training metric."""
        self.metrics_history.append(metric)
        
        # Maintain history size
        if len(self.metrics_history) > self._max_history_size:
            self.metrics_history = self.metrics_history[-self._max_history_size:]
        
        # Detect anomalies if premium
        if self.premium_manager.has_feature("advanced_analytics"):
            self._detect_anomalies(metric)
        
        # Generate insights if premium
        if self.premium_manager.has_feature("ml_insights"):
            self._generate_insights()
    
    def _detect_anomalies(self, metric: TrainingMetric) -> None:
        """Detect anomalies in training metrics."""
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = self.metrics_history[-10:]
        
        # Calculate statistical baselines
        losses = [m.loss for m in recent_metrics]
        gpu_utils = [m.gpu_utilization for m in recent_metrics]
        memory_usages = [m.memory_usage for m in recent_metrics]
        
        # Check for loss spikes
        loss_mean = np.mean(losses) if NUMPY_AVAILABLE else sum(losses) / len(losses)
        loss_std = np.std(losses) if NUMPY_AVAILABLE else math.sqrt(sum((x - loss_mean) ** 2 for x in losses) / len(losses))
        
        if abs(metric.loss - loss_mean) > 3 * loss_std:
            self.anomalies.append(AnomalyDetection(
                anomaly_type="loss_spike",
                severity="warning" if abs(metric.loss - loss_mean) < 5 * loss_std else "critical",
                detected_at=metric.timestamp,
                metrics={"loss": metric.loss, "mean": loss_mean, "std": loss_std},
                expected_range=(loss_mean - 2 * loss_std, loss_mean + 2 * loss_std),
                actual_value=metric.loss,
                context=f"Loss deviation detected at step {metric.step}",
            ))
        
        # Check for GPU utilization anomalies
        gpu_mean = np.mean(gpu_utils) if NUMPY_AVAILABLE else sum(gpu_utils) / len(gpu_utils)
        if metric.gpu_utilization < gpu_mean * 0.5 and metric.gpu_utilization < 50:
            self.anomalies.append(AnomalyDetection(
                anomaly_type="low_gpu_utilization",
                severity="warning",
                detected_at=metric.timestamp,
                metrics={"gpu_utilization": metric.gpu_utilization, "mean": gpu_mean},
                expected_range=(gpu_mean * 0.8, gpu_mean * 1.2),
                actual_value=metric.gpu_utilization,
                context=f"GPU utilization dropped at step {metric.step}",
            ))
        
        # Check for memory leaks
        memory_trend = memory_usages[-1] - memory_usages[0]
        if memory_trend > 10:  # 10% increase
            self.anomalies.append(AnomalyDetection(
                anomaly_type="memory_leak",
                severity="warning",
                detected_at=metric.timestamp,
                metrics={"memory_usage": metric.memory_usage, "trend": memory_trend},
                expected_range=(memory_usages[0], memory_usages[0] + 5),
                actual_value=metric.memory_usage,
                context=f"Memory usage increasing at step {metric.step}",
            ))
    
    def _generate_insights(self) -> None:
        """Generate ML insights from training metrics."""
        if len(self.metrics_history) < 20:
            return
        
        recent_metrics = self.metrics_history[-20:]
        
        # Performance insights
        losses = [m.loss for m in recent_metrics]
        if len(losses) >= 5:
            loss_trend = losses[-1] - losses[0]
            if loss_trend > 0.1:
                self.insights.append(MLInsight(
                    insight_type=InsightType.PERFORMANCE,
                    severity="warning",
                    title="Loss Increasing",
                    description=f"Training loss has increased by {loss_trend:.3f} over the last 20 steps",
                    recommendation="Consider reducing learning rate or checking for data quality issues",
                    confidence=0.8,
                    metrics={"loss_trend": loss_trend, "current_loss": losses[-1]},
                ))
            elif loss_trend < -0.05:
                self.insights.append(MLInsight(
                    insight_type=InsightType.PERFORMANCE,
                    severity="info",
                    title="Loss Decreasing Well",
                    description=f"Training loss is decreasing steadily by {abs(loss_trend):.3f}",
                    recommendation="Current training progression looks good",
                    confidence=0.9,
                    metrics={"loss_trend": loss_trend, "current_loss": losses[-1]},
                ))
        
        # Efficiency insights
        gpu_utils = [m.gpu_utilization for m in recent_metrics]
        avg_gpu = np.mean(gpu_utils) if NUMPY_AVAILABLE else sum(gpu_utils) / len(gpu_utils)
        
        if avg_gpu < 60:
            self.insights.append(MLInsight(
                insight_type=InsightType.EFFICIENCY,
                severity="warning",
                title="Low GPU Utilization",
                description=f"Average GPU utilization is {avg_gpu:.1f}%, which is below optimal",
                recommendation="Consider increasing batch size or reducing gradient accumulation steps",
                confidence=0.85,
                metrics={"avg_gpu_utilization": avg_gpu},
            ))
        
        # Resource insights
        memory_usages = [m.memory_usage for m in recent_metrics]
        max_memory = max(memory_usages)
        
        if max_memory > 90:
            self.insights.append(MLInsight(
                insight_type=InsightType.RESOURCE,
                severity="critical",
                title="High Memory Usage",
                description=f"Memory usage peaked at {max_memory:.1f}%",
                recommendation="Consider gradient checkpointing or reducing batch size to prevent OOM errors",
                confidence=0.95,
                metrics={"max_memory_usage": max_memory},
            ))
        
        # Throughput insights
        throughputs = [m.throughput for m in recent_metrics if m.throughput > 0]
        if throughputs:
            avg_throughput = np.mean(throughputs) if NUMPY_AVAILABLE else sum(throughputs) / len(throughputs)
            self.insights.append(MLInsight(
                insight_type=InsightType.EFFICIENCY,
                severity="info",
                title="Training Throughput",
                description=f"Current training throughput is {avg_throughput:.1f} samples/second",
                recommendation="Monitor throughput trends to optimize training efficiency",
                confidence=0.7,
                metrics={"avg_throughput": avg_throughput},
            ))


class AdvancedDashboard:
    """Advanced monitoring dashboard with real-time insights."""
    
    def __init__(self):
        self.premium_manager = get_premium_manager()
        self.metrics_collector = AdvancedMetricsCollector()
        self.alerts: list[dict[str, Any]] = []
        self.dashboard_data: dict[str, Any] = {}
        self._subscribers: list[Callable] = []
    
    def subscribe_to_updates(self, callback: Callable) -> None:
        """Subscribe to dashboard updates."""
        self._subscribers.append(callback)
    
    def _notify_subscribers(self) -> None:
        """Notify all subscribers of updates."""
        for callback in self._subscribers:
            try:
                callback(self.get_dashboard_data())
            except Exception:
                pass
    
    def update_metrics(self, metric: TrainingMetric) -> None:
        """Update dashboard with new metrics."""
        self.metrics_collector.collect_metric(metric)
        self._update_dashboard_data()
        self._notify_subscribers()
    
    def _update_dashboard_data(self) -> None:
        """Update dashboard data structure."""
        if not self.metrics_collector.metrics_history:
            return
        
        recent_metrics = self.metrics_collector.metrics_history[-100:]
        
        # Calculate aggregates
        losses = [m.loss for m in recent_metrics]
        gpu_utils = [m.gpu_utilization for m in recent_metrics]
        memory_usages = [m.memory_usage for m in recent_metrics]
        throughputs = [m.throughput for m in recent_metrics if m.throughput > 0]
        
        self.dashboard_data = {
            "current_metrics": {
                "loss": recent_metrics[-1].loss if recent_metrics else 0.0,
                "learning_rate": recent_metrics[-1].learning_rate if recent_metrics else 0.0,
                "gradient_norm": recent_metrics[-1].gradient_norm if recent_metrics else 0.0,
                "gpu_utilization": recent_metrics[-1].gpu_utilization if recent_metrics else 0.0,
                "memory_usage": recent_metrics[-1].memory_usage if recent_metrics else 0.0,
                "throughput": recent_metrics[-1].throughput if recent_metrics else 0.0,
                "step": recent_metrics[-1].step if recent_metrics else 0,
                "epoch": recent_metrics[-1].epoch if recent_metrics else 0,
            },
            "aggregates": {
                "avg_loss": np.mean(losses) if NUMPY_AVAILABLE and losses else sum(losses) / len(losses) if losses else 0.0,
                "min_loss": min(losses) if losses else 0.0,
                "max_loss": max(losses) if losses else 0.0,
                "loss_std": np.std(losses) if NUMPY_AVAILABLE and losses else 0.0,
                "avg_gpu_utilization": np.mean(gpu_utils) if NUMPY_AVAILABLE and gpu_utils else sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0,
                "avg_memory_usage": np.mean(memory_usages) if NUMPY_AVAILABLE and memory_usages else sum(memory_usages) / len(memory_usages) if memory_usages else 0.0,
                "avg_throughput": np.mean(throughputs) if NUMPY_AVAILABLE and throughputs else sum(throughputs) / len(throughputs) if throughputs else 0.0,
            },
            "trends": {
                "loss_trend": losses[-1] - losses[0] if len(losses) >= 2 else 0.0,
                "gpu_trend": gpu_utils[-1] - gpu_utils[0] if len(gpu_utils) >= 2 else 0.0,
                "memory_trend": memory_usages[-1] - memory_usages[0] if len(memory_usages) >= 2 else 0.0,
            },
            "insights": [
                {
                    "type": insight.insight_type.value,
                    "severity": insight.severity,
                    "title": insight.title,
                    "description": insight.description,
                    "recommendation": insight.recommendation,
                    "confidence": insight.confidence,
                }
                for insight in self.metrics_collector.insights[-10:]
            ],
            "anomalies": [
                {
                    "type": anomaly.anomaly_type,
                    "severity": anomaly.severity,
                    "detected_at": anomaly.detected_at,
                    "context": anomaly.context,
                }
                for anomaly in self.metrics_collector.anomalies[-10:]
            ],
            "premium_features": {
                "advanced_analytics": self.premium_manager.has_feature("advanced_analytics"),
                "ml_insights": self.premium_manager.has_feature("ml_insights"),
                "real_time_metrics": self.premium_manager.has_feature("real_time_metrics"),
                "tensorboard_integration": self.premium_manager.has_feature("tensorboard_integration"),
            },
            "tier": self.premium_manager.get_current_tier().value,
        }
    
    def get_dashboard_data(self) -> dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data
    
    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_collector.metrics_history:
            return {"error": "No metrics available"}
        
        metrics = self.metrics_collector.metrics_history
        
        # Calculate performance metrics
        total_time = metrics[-1].timestamp - metrics[0].timestamp if len(metrics) > 1 else 0
        total_steps = metrics[-1].step - metrics[0].step if len(metrics) > 1 else 0
        steps_per_second = total_steps / total_time if total_time > 0 else 0
        
        losses = [m.loss for m in metrics]
        loss_improvement = losses[0] - losses[-1] if len(losses) > 1 else 0
        loss_improvement_percent = (loss_improvement / losses[0] * 100) if losses[0] > 0 else 0
        
        gpu_utils = [m.gpu_utilization for m in metrics]
        avg_gpu = np.mean(gpu_utils) if NUMPY_AVAILABLE and gpu_utils else sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
        
        throughputs = [m.throughput for m in metrics if m.throughput > 0]
        avg_throughput = np.mean(throughputs) if NUMPY_AVAILABLE and throughputs else sum(throughputs) / len(throughputs) if throughputs else 0
        
        return {
            "training_summary": {
                "total_time_hours": total_time / 3600.0,
                "total_steps": total_steps,
                "steps_per_second": steps_per_second,
                "final_loss": losses[-1] if losses else 0.0,
                "initial_loss": losses[0] if losses else 0.0,
                "loss_improvement": loss_improvement,
                "loss_improvement_percent": loss_improvement_percent,
            },
            "resource_efficiency": {
                "avg_gpu_utilization": avg_gpu,
                "avg_memory_usage": np.mean([m.memory_usage for m in metrics]) if NUMPY_AVAILABLE else sum(m.memory_usage for m in metrics) / len(metrics),
                "avg_throughput": avg_throughput,
            },
            "insights_summary": {
                "total_insights": len(self.metrics_collector.insights),
                "critical_insights": len([i for i in self.metrics_collector.insights if i.severity == "critical"]),
                "warning_insights": len([i for i in self.metrics_collector.insights if i.severity == "warning"]),
                "info_insights": len([i for i in self.metrics_collector.insights if i.severity == "info"]),
            },
            "anomaly_summary": {
                "total_anomalies": len(self.metrics_collector.anomalies),
                "critical_anomalies": len([a for a in self.metrics_collector.anomalies if a.severity == "critical"]),
                "warning_anomalies": len([a for a in self.metrics_collector.anomalies if a.severity == "warning"]),
            },
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on insights."""
        recommendations = []
        
        for insight in self.metrics_collector.insights[-5:]:
            if insight.severity in ["critical", "warning"]:
                recommendations.append(insight.recommendation)
        
        for anomaly in self.metrics_collector.anomalies[-5:]:
            if anomaly.severity == "critical":
                recommendations.append(f"Address {anomaly.anomaly_type}: {anomaly.context}")
        
        return list(set(recommendations))  # Remove duplicates
    
    def export_metrics(self, export_path: Path) -> None:
        """Export metrics to file for analysis."""
        export_data = {
            "metrics": [m.__dict__ for m in self.metrics_collector.metrics_history],
            "insights": [i.__dict__ for i in self.metrics_collector.insights],
            "anomalies": [a.__dict__ for a in self.metrics_collector.anomalies],
            "dashboard_data": self.dashboard_data,
            "exported_at": time.time(),
        }
        
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def setup_tensorboard_integration(self, log_dir: Path) -> bool:
        """Setup TensorBoard integration for premium users."""
        if not self.premium_manager.has_feature("tensorboard_integration"):
            return False
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(log_dir)
            return True
        except ImportError:
            return False
    
    def log_to_tensorboard(self, metric: TrainingMetric) -> None:
        """Log metrics to TensorBoard."""
        if not self.premium_manager.has_feature("tensorboard_integration"):
            return
        
        if hasattr(self, 'tensorboard_writer') and self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("Loss/train", metric.loss, metric.step)
            self.tensorboard_writer.add_scalar("Learning_Rate", metric.learning_rate, metric.step)
            self.tensorboard_writer.add_scalar("GPU_Utilization", metric.gpu_utilization, metric.step)
            self.tensorboard_writer.add_scalar("Memory_Usage", metric.memory_usage, metric.step)
            self.tensorboard_writer.add_scalar("Throughput", metric.throughput, metric.step)
            self.tensorboard_writer.add_scalar("Gradient_Norm", metric.gradient_norm, metric.step)


def get_advanced_dashboard() -> AdvancedDashboard:
    """Get global advanced dashboard instance."""
    if not hasattr(get_advanced_dashboard, "_instance"):
        get_advanced_dashboard._instance = AdvancedDashboard()
    return get_advanced_dashboard._instance
