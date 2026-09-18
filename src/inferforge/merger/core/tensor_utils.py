"""
Tensor utilities for efficient model merging.

Handles precision management, memory optimization, and disk-to-disk streaming
for merging large models without exhausting system resources.
"""

import gc
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from safetensors.torch import load_file, save_file


class TensorUtils:
    """Utilities for tensor operations during model merging."""
    
    @staticmethod
    def get_precision(dtype: str = "bfloat16") -> torch.dtype:
        """Get torch dtype from string identifier."""
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
        }
        return precision_map.get(dtype, torch.bfloat16)
    
    @staticmethod
    def clear_cache() -> None:
        """Clear GPU and CPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def load_tensor_safely(
        file_path: Union[str, Path],
        tensor_name: str,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Load a single tensor from safetensors file with memory safety."""
        try:
                                                   
            tensors = load_file(str(file_path), device="cpu")
            tensor = tensors[tensor_name]
            
            if dtype is not None:
                tensor = tensor.to(dtype)
            
                                                  
            del tensors
            TensorUtils.clear_cache()
            
            return tensor
        except Exception as e:
            raise RuntimeError(f"Failed to load tensor {tensor_name} from {file_path}: {e}")
    
    @staticmethod
    def save_tensor_safely(
        tensor: torch.Tensor,
        file_path: Union[str, Path],
        tensor_name: str
    ) -> None:
        """Save a single tensor to safetensors file with memory safety."""
        try:
            save_file({tensor_name: tensor}, str(file_path))
            TensorUtils.clear_cache()
        except Exception as e:
            raise RuntimeError(f"Failed to save tensor {tensor_name} to {file_path}: {e}")
    
    @staticmethod
    def estimate_tensor_memory(tensor: torch.Tensor) -> float:
        """Estimate memory usage of a tensor in GB."""
        return tensor.numel() * tensor.element_size() / (1024**3)
    
    @staticmethod
    def safe_reshape(
        tensor: torch.Tensor,
        new_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Safely reshape tensor, handling dimension mismatches."""
        try:
            return tensor.reshape(new_shape)
        except RuntimeError:
                                                               
            total_elements = tensor.numel()
            target_elements = np.prod(new_shape)
            
            if total_elements != target_elements:
                raise ValueError(
                    f"Cannot reshape tensor from {tensor.shape} to {new_shape}: "
                    f"element count mismatch ({total_elements} vs {target_elements})"
                )
            
            return tensor.reshape(new_shape)
    
    @staticmethod
    def interpolate_size(
        size1: int,
        size2: int,
        target_size: int,
        method: str = "linear"
    ) -> torch.Tensor:
        """Interpolate between two tensor sizes to reach target size."""
        if method == "linear":
                                                  
            if target_size >= max(size1, size2):
                                         
                larger = max(size1, size2)
                scale = target_size / larger
                return torch.linspace(0, larger - 1, target_size).long()
            else:
                            
                scale = target_size / max(size1, size2)
                return torch.linspace(0, max(size1, size2) - 1, target_size).long()
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    @staticmethod
    def match_shape(tensor: torch.Tensor, shape: Tuple[int, ...], is_norm_scale: bool = False) -> torch.Tensor:
        if tensor.shape == shape:
            return tensor
        if tensor.numel() == int(np.prod(shape)):
            return tensor.reshape(shape)
        # If 1D normalization or scale weight (mean close to 1.0), pad with 1.0 instead of 0.0
        fill_val = 1.0 if (is_norm_scale or (tensor.dim() == 1 and len(shape) == 1 and abs(tensor.float().mean().item() - 1.0) < 0.35)) else 0.0
        result = torch.full(shape, fill_val, dtype=tensor.dtype, device=tensor.device)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(shape, tensor.shape))
        if len(slices) == tensor.dim() and len(slices) == len(shape):
            result[slices] = tensor[slices]
            return result
        flat_src = tensor.reshape(-1)
        flat_dst = result.reshape(-1)
        count = min(flat_src.numel(), flat_dst.numel())
        flat_dst[:count] = flat_src[:count]
        return flat_dst.reshape(shape)

    @staticmethod
    def normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize tensor to unit norm."""
        norm = torch.norm(tensor)
        if norm < eps:
            return tensor
        return tensor / norm
    
    @staticmethod
    def calculate_cosine_similarity(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor
    ) -> float:
        """Calculate cosine similarity between two tensors."""
        tensor1_flat = tensor1.flatten()
        tensor2_flat = tensor2.flatten()
        
                                                         
        min_size = min(tensor1_flat.numel(), tensor2_flat.numel())
        tensor1_flat = tensor1_flat[:min_size]
        tensor2_flat = tensor2_flat[:min_size]
        
        similarity = torch.nn.functional.cosine_similarity(
            tensor1_flat.unsqueeze(0),
            tensor2_flat.unsqueeze(0)
        )
        return similarity.item()
