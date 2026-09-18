"""
Weight blending algorithms for model merging.

Implements multiple sophisticated merging strategies:
- SLERP (Spherical Linear Interpolation)
- TIES (Task Iteration and Ejection Sign)
- MoE (Mixture of Experts) routing
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch


class MergeStrategy(Enum):
    """Available merging strategies."""
    SLERP = "slerp"
    TIES = "ties"
    DARE_TIES = "dare_ties"
    DARE_LINEAR = "dare_linear"
    TASK_ARITHMETIC = "task_arithmetic"
    PASSTHROUGH = "passthrough"
    MOE = "moe"
    LINEAR = "linear"
    SIMPLE_AVERAGE = "simple_average"


@dataclass
class MergeConfig:
    """Configuration for weight merging."""
    strategy: MergeStrategy = MergeStrategy.TIES
    interpolation: float = 0.5
    weights: Optional[List[float]] = None
    ties_param_k: float = 0.2
    ties_lambda: float = 0.5
    dare_drop_rate: float = 0.2
    dare_rescale: bool = True
    moe_num_experts: int = 4
    moe_top_k: int = 2
    precision: str = "bfloat16"
    normalize: bool = False


class WeightBlender:
    """Advanced weight blending for model merging."""
    
    def __init__(self, config: Optional[MergeConfig] = None):
        """
        Initialize weight blender.
        
        Args:
            config: Merge configuration, uses defaults if None
        """
        self.config = config or MergeConfig()
        
    def slerp(self, val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """
        Spherical linear interpolation for precise geometric weight blending.
        """
        low_f = low.float()
        high_f = high.float()
        low_norm = low_f / (torch.norm(low_f) + 1e-8)
        high_norm = high_f / (torch.norm(high_f) + 1e-8)
        
        dot = torch.clamp(torch.dot(low_norm.flatten(), high_norm.flatten()), -1.0, 1.0)
        omega = torch.acos(dot)
        so = torch.sin(omega)
        
        if so < 1e-6:
            res = (1.0 - val) * low_f + val * high_f
        else:
            res = (torch.sin((1.0 - val) * omega) / so) * low_f + (torch.sin(val * omega) / so) * high_f
        return res.to(low.dtype)
    
    def ties_merge(
        self,
        weights: List[torch.Tensor],
        base_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        TIES merging: Trim, Elect Sign, Disjoint Merge.
        Fixed with float quantile calculation to avoid BFloat16/Half errors.
        """
        if not weights:
            raise ValueError("No weights provided for merging")
        
        if base_weights is not None:
            deltas = [w - base_weights for w in weights]
        else:
            deltas = weights
        
        k = max(1e-4, min(1.0, float(self.config.ties_param_k)))
        trimmed_deltas = []
        
        for delta in deltas:
            # Crucial fix: convert magnitude to float32 before torch.quantile
            magnitude = torch.abs(delta).float()
            if magnitude.numel() <= 1:
                trimmed_deltas.append(delta)
                continue
            
            # For massive tensors, sample to avoid excessive memory spike
            if magnitude.numel() > 500_000:
                step = max(1, magnitude.numel() // 100_000)
                sample = magnitude.reshape(-1)[::step]
                threshold = torch.quantile(sample, 1.0 - k).item()
            else:
                threshold = torch.quantile(magnitude, 1.0 - k).item()
            
            mask = magnitude >= threshold
            trimmed_delta = delta * mask.to(delta.dtype)
            trimmed_deltas.append(trimmed_delta)
        
        if len(trimmed_deltas) > 1:
            sign_sum = torch.zeros_like(trimmed_deltas[0], dtype=torch.float32)
            for d in trimmed_deltas:
                sign_sum += torch.sign(d.float())
            majority_sign = torch.sign(sign_sum)
            majority_sign[majority_sign == 0] = 1.0
            
            for i in range(len(trimmed_deltas)):
                trimmed_deltas[i] = torch.abs(trimmed_deltas[i]) * majority_sign.to(trimmed_deltas[i].dtype)
            
            first_sign = torch.sign(trimmed_deltas[0])
            agreement_mask = torch.ones_like(first_sign, dtype=torch.bool)
            acc = torch.zeros_like(trimmed_deltas[0], dtype=torch.float32)
            for d in trimmed_deltas:
                agreement_mask &= (torch.sign(d) == first_sign)
                acc += d.float()
            
            merged_delta = torch.zeros_like(trimmed_deltas[0])
            if agreement_mask.any():
                merged_delta[agreement_mask] = (acc[agreement_mask] / len(trimmed_deltas)).to(trimmed_deltas[0].dtype)
        else:
            merged_delta = trimmed_deltas[0]
        
        if base_weights is not None:
            result = base_weights + merged_delta
        else:
            result = merged_delta
        
        return result

    def dare_merge(
        self,
        weights: List[torch.Tensor],
        base_weights: Optional[torch.Tensor] = None,
        use_ties_consensus: bool = True
    ) -> torch.Tensor:
        """
        DARE (Drop And REscale) merging.
        Drops delta weights randomly with drop_rate p, and rescales surviving by 1/(1-p).
        """
        if not weights:
            raise ValueError("No weights provided for DARE merging")
        
        base = base_weights if base_weights is not None else weights[0]
        p = max(0.0, min(0.99, float(self.config.dare_drop_rate)))
        rescale = 1.0 / (1.0 - p) if (self.config.dare_rescale and p < 1.0) else 1.0
        
        pruned_deltas = []
        for w in weights:
            delta = (w - base).float()
            if p > 0.0:
                mask = (torch.rand_like(delta) > p).float()
                pruned = delta * mask * rescale
            else:
                pruned = delta
            pruned_deltas.append(pruned)
        
        if use_ties_consensus and len(pruned_deltas) > 1:
            sign_sum = torch.zeros_like(pruned_deltas[0], dtype=torch.float32)
            for d in pruned_deltas:
                sign_sum += torch.sign(d)
            majority_sign = torch.sign(sign_sum)
            majority_sign[majority_sign == 0] = 1.0
            
            merged_delta = torch.zeros_like(pruned_deltas[0])
            for d in pruned_deltas:
                merged_delta += torch.abs(d) * majority_sign
            merged_delta = merged_delta / len(pruned_deltas)
        else:
            weights_coeff = self._get_normalized_weights(len(pruned_deltas))
            merged_delta = sum(w * d for w, d in zip(weights_coeff, pruned_deltas))
        
        return (base.float() + merged_delta).to(base.dtype)

    def task_arithmetic_merge(
        self,
        weights: List[torch.Tensor],
        base_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Task Arithmetic: theta_merged = theta_base + sum(w_i * (theta_i - theta_base))
        """
        if not weights:
            raise ValueError("No weights provided for task arithmetic")
        
        base = base_weights if base_weights is not None else weights[0]
        model_weights = weights[1:] if base_weights is None and len(weights) > 1 else weights
        coeffs = self._get_normalized_weights(len(model_weights))
        
        acc = base.float().clone()
        for c, w in zip(coeffs, model_weights):
            delta = (w - base).float()
            acc = acc + c * delta
        return acc.to(base.dtype)

    def _get_normalized_weights(self, count: int) -> List[float]:
        if self.config.weights and len(self.config.weights) >= count:
            raw = [float(x) for x in self.config.weights[:count]]
            s = sum(raw)
            return [x / s for x in raw] if s > 0 else [1.0 / count] * count
        return [1.0 / count] * count

    def moe_routing(
        self,
        expert_weights: List[torch.Tensor],
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        num_experts = len(expert_weights)
        if num_experts == 1:
            return torch.matmul(input_tensor, expert_weights[0])
        input_norm = torch.norm(input_tensor)
        expert_idx = int(input_norm) % num_experts
        selected_expert = expert_weights[expert_idx]
        return torch.matmul(input_tensor, selected_expert)
    
    def linear_interpolate(
        self,
        weights: List[torch.Tensor],
        interpolation: Optional[float] = None
    ) -> torch.Tensor:
        if not weights:
            raise ValueError("No weights provided for interpolation")
        
        coeffs = self._get_normalized_weights(len(weights))
        acc = torch.zeros_like(weights[0].float())
        for c, w in zip(coeffs, weights):
            acc = acc + c * w.float()
        return acc.to(weights[0].dtype)
    
    def simple_average(self, weights: List[torch.Tensor]) -> torch.Tensor:
        if not weights:
            raise ValueError("No weights provided for averaging")
        if len(weights) == 1:
            return weights[0]
        acc = weights[0].float().clone()
        for w in weights[1:]:
            acc += w.float()
        return (acc / len(weights)).to(weights[0].dtype)
    
    def _execute_strategy(
        self,
        strategy: MergeStrategy,
        weights: List[torch.Tensor],
        base_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if strategy == MergeStrategy.SLERP:
            result = weights[0]
            for other in weights[1:]:
                result = self.slerp(self.config.interpolation, result, other)
        elif strategy == MergeStrategy.TIES:
            result = self.ties_merge(weights, base_weights)
        elif strategy == MergeStrategy.DARE_TIES:
            result = self.dare_merge(weights, base_weights, use_ties_consensus=True)
        elif strategy == MergeStrategy.DARE_LINEAR:
            result = self.dare_merge(weights, base_weights, use_ties_consensus=False)
        elif strategy == MergeStrategy.TASK_ARITHMETIC:
            result = self.task_arithmetic_merge(weights, base_weights)
        elif strategy == MergeStrategy.PASSTHROUGH:
            result = weights[0]
        elif strategy == MergeStrategy.MOE:
            result = self._moe_blend(weights)
        elif strategy == MergeStrategy.LINEAR:
            result = self.linear_interpolate(weights)
        elif strategy == MergeStrategy.SIMPLE_AVERAGE:
            result = self.simple_average(weights)
        else:
            result = self.simple_average(weights)
        return result

    def merge_weights(
        self,
        weights: List[torch.Tensor],
        base_weights: Optional[torch.Tensor] = None,
        strategy: Optional[MergeStrategy] = None
    ) -> torch.Tensor:
        if not weights:
            raise ValueError("No weights provided for merging")
        
        if strategy is None:
            strategy = self.config.strategy
        if isinstance(strategy, str):
            strategy = MergeStrategy(strategy.lower())
        
        target_dtype = self._get_target_dtype()
        weights = [self._ensure_float_tensor(w).to(target_dtype) for w in weights]
        if base_weights is not None:
            base_weights = self._ensure_float_tensor(base_weights).to(target_dtype)
        
        if self.config.normalize:
            weights = [self._normalize_tensor(w) for w in weights]
            if base_weights is not None:
                base_weights = self._normalize_tensor(base_weights)
        
        weights = self._match_group(weights)
        if base_weights is not None and base_weights.shape != weights[0].shape:
            base_weights = self._match_to(base_weights, weights[0].shape)

        try:
            return self._execute_strategy(strategy, weights, base_weights)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError):
                import logging
                logging.warning(f"CUDA Out-of-memory during {strategy}; falling back to CPU")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cpu_weights = [w.cpu() for w in weights]
                cpu_base = base_weights.cpu() if base_weights is not None else None
                result = self._execute_strategy(strategy, cpu_weights, cpu_base)
                return result.to(weights[0].device)
            import logging
            logging.warning(f"Merge strategy {strategy} failed ({exc}); falling back to simple_average")
            return self.simple_average(weights)
        except Exception as exc:
            import logging
            logging.warning(f"Merge strategy {strategy} failed ({exc}); falling back to simple_average")
            return self.simple_average(weights)

    def _ensure_float_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_floating_point():
            return tensor.float()
        return tensor

    def merge_state_dicts(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        base: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if not state_dicts:
            raise ValueError("No state dicts provided for merging")
        keys: set[str] = set()
        for state in state_dicts:
            keys.update(state.keys())
        merged: Dict[str, torch.Tensor] = {}
        for key in keys:
            tensors = [state[key] for state in state_dicts if key in state]
            if not tensors:
                continue
            if len(tensors) == 1:
                merged[key] = tensors[0]
                continue
            base_tensor = base.get(key) if base else tensors[0]
            merged[key] = self.merge_weights(tensors, base_weights=base_tensor)
        return merged

    def _match_group(self, weights: List[torch.Tensor]) -> List[torch.Tensor]:
        reference = weights[0]
        return [self._match_to(tensor, reference.shape) for tensor in weights]

    def _match_to(self, tensor: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        if tensor.shape == shape:
            return tensor
        from inferforge.merger.core.tensor_utils import TensorUtils

        return TensorUtils.match_shape(tensor, shape)

    def _moe_blend(self, weights: List[torch.Tensor]) -> torch.Tensor:
        norms = torch.stack([torch.norm(tensor.float()) + 1e-8 for tensor in weights])
        gates = norms / norms.sum()
        result = torch.zeros_like(weights[0])
        for gate, tensor in zip(gates, weights):
            result = result + tensor * gate.to(tensor.dtype)
        return result
    
    def _get_target_dtype(self) -> torch.dtype:
        """Get target dtype from config."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
        }
        return dtype_map.get(self.config.precision, torch.bfloat16)
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to prevent numerical issues."""
        norm = torch.norm(tensor)
        if norm < 1e-8:
            return tensor
        return tensor / norm
    
    def calculate_parameters(self, model_dict: Dict[str, torch.Tensor]) -> int:
        """
        Calculate total parameters in a model dictionary.
        
        Args:
            model_dict: Dictionary of parameter name -> tensor
        
        Returns:
            Total number of parameters
        """
        return sum(tensor.numel() for tensor in model_dict.values())
    
    def estimate_moe_parameters(
        self,
        base_params: int,
        num_experts: int,
        expert_param_ratio: float = 0.5
    ) -> int:
        """
        Estimate parameters for MoE model.
        
        Args:
            base_params: Parameters in base model
            num_experts: Number of experts
            expert_param_ratio: Ratio of parameters that become expert-specific
        
        Returns:
            Estimated total parameters
        """
        shared_params = base_params * (1.0 - expert_param_ratio)
        expert_params = base_params * expert_param_ratio * num_experts
        routing_params = base_params * 0.01                   
        
        return int(shared_params + expert_params + routing_params)
