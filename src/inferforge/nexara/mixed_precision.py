"""Mixed precision training engine for Nexara."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


def _has_non_finite(value: Any) -> bool:
    """Recursively detect inf/nan in tensors, numpy arrays, mappings, sequences, or scalars."""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_has_non_finite(v) for v in value.values())
    if hasattr(value, "grad") and getattr(value, "grad", None) is not None and not hasattr(value, "isfinite"):
        return _has_non_finite(value.grad)
    isfinite = getattr(value, "isfinite", None)
    if callable(isfinite):
        try:
            result = isfinite()
            if hasattr(result, "all"):
                return not bool(result.all())
        except Exception:
            pass
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return any(_has_non_finite(v) for v in value)
    return False


class PrecisionMode(Enum):
    """Precision modes for training."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    MIXED = "mixed"


@dataclass
class PrecisionConfig:
    """Configuration for mixed precision training."""
    mode: PrecisionMode = PrecisionMode.MIXED
    loss_scale: float = 2.0**16
    loss_scale_window: int = 1000
    min_loss_scale: float = 1.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000


class MixedPrecisionEngine:
    """Engine for mixed precision training."""
    
    def __init__(self, config: PrecisionConfig | None = None):
        self.config = config or PrecisionConfig()
        self.overflow_count = 0
        self.scale_growth_tracker = 0
    
    def get_precision_strategy(
        self,
        operation: str,
        importance: str = "medium",
    ) -> str:
        """Get precision strategy for specific operation."""
        # Operations that need high precision
        high_precision_ops = {
            "loss_calculation",
            "gradient_accumulation",
            "optimizer_step",
            "layer_norm",
            "softmax",
        }
        
        # Operations that can use lower precision
        low_precision_ops = {
            "forward_pass",
            "matmul",
            "conv",
            "relu",
            "attention",
        }
        
        if operation in high_precision_ops or importance == "high":
            return "fp32"
        elif operation in low_precision_ops:
            if self.config.mode == PrecisionMode.FP16:
                return "fp16"
            elif self.config.mode == PrecisionMode.BF16:
                return "bf16"
            else:
                return "mixed"
        else:
            return str(self.config.mode.value)
    
    def scale_loss(self, loss: float) -> float:
        """Scale loss for mixed precision training."""
        if self.config.mode in [PrecisionMode.FP16, PrecisionMode.MIXED]:
            return loss * self.config.loss_scale
        return loss
    
    def unscale_gradients(self, gradients: Any) -> Any:
        """Unscale gradients after backward pass."""
        # In real implementation, this would unscale torch tensors
        # Here we just track the scaling
        return gradients
    
    def check_overflow(self, gradients: Any) -> bool:
        """Check if gradients have overflowed."""
        has_overflow = _has_non_finite(gradients)
        
        if has_overflow:
            self.overflow_count += 1
            self._adjust_loss_scale(overflow=True)
        else:
            self.scale_growth_tracker += 1
            if self.scale_growth_tracker >= self.config.growth_interval:
                self._adjust_loss_scale(overflow=False)
                self.scale_growth_tracker = 0
        
        return has_overflow
    
    def _adjust_loss_scale(self, overflow: bool) -> None:
        """Adjust loss scale based on overflow."""
        if overflow:
            self.config.loss_scale *= self.config.backoff_factor
            self.config.loss_scale = max(
                self.config.loss_scale,
                self.config.min_loss_scale,
            )
        else:
            self.config.loss_scale *= self.config.growth_factor
    
    def should_skip_step(self) -> bool:
        """Check if optimizer step should be skipped due to overflow."""
        return self.overflow_count > 0
    
    def get_dtype_recommendation(self, hardware: str) -> PrecisionMode:
        """Recommend precision mode based on hardware."""
        if "A100" in hardware or "H100" in hardware:
            # Modern GPUs with good BF16 support
            return PrecisionMode.BF16
        elif "V100" in hardware or "T4" in hardware:
            # Older GPUs with FP16 support
            return PrecisionMode.FP16
        elif "RTX" in hardware:
            # Consumer GPUs
            return PrecisionMode.MIXED
        else:
            # CPU or unknown - use FP32
            return PrecisionMode.FP32
    
    def estimate_speedup(self) -> float:
        """Estimate training speedup from mixed precision."""
        speedup_map = {
            PrecisionMode.FP32: 1.0,
            PrecisionMode.FP16: 2.0,
            PrecisionMode.BF16: 1.8,
            PrecisionMode.INT8: 3.0,
            PrecisionMode.MIXED: 1.5,
        }
        return speedup_map.get(self.config.mode, 1.0)
    
    def estimate_memory_saving(self) -> float:
        """Estimate memory savings from mixed precision."""
        savings_map = {
            PrecisionMode.FP32: 0.0,
            PrecisionMode.FP16: 0.5,
            PrecisionMode.BF16: 0.5,
            PrecisionMode.INT8: 0.75,
            PrecisionMode.MIXED: 0.3,
        }
        return savings_map.get(self.config.mode, 0.0)
    
    def get_training_config(self) -> dict[str, Any]:
        """Get training configuration for mixed precision."""
        return {
            "mode": self.config.mode.value,
            "loss_scale": self.config.loss_scale,
            "estimated_speedup": self.estimate_speedup(),
            "estimated_memory_saving": self.estimate_memory_saving(),
            "overflow_count": self.overflow_count,
        }
