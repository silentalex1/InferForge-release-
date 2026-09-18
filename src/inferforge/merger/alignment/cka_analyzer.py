"""
Dynamic SVD (Singular Value Decomposition) for Dimension Mismatch.

If models have different hidden dimensions (e.g., 4096 vs 5120), use SVD to
compress or expand matrix spaces into matching target dimensions while preserving
the most important mathematical "singular vectors".
"""

from typing import Optional, Tuple

import torch


class DynamicSVD:
    """Dynamic SVD for handling dimension mismatches during merging."""
    
    @staticmethod
    def compress_via_svd(
        tensor: torch.Tensor,
        target_dim: int,
        retain_ratio: float = 0.95
    ) -> torch.Tensor:
        """
        Compress tensor to target dimension using SVD.
        
        Args:
            tensor: Input tensor [m, n]
            target_dim: Target dimension
            retain_ratio: Ratio of singular values to retain
        
        Returns:
            Compressed tensor
        """
        orig_device = tensor.device
        orig_dtype = tensor.dtype
        t_f = tensor.detach().float()
        
        was_1d = False
        if t_f.ndim == 1:
            was_1d = True
            t_f = t_f.unsqueeze(1)
            
        m, n = t_f.shape
        if n == target_dim:
            return tensor
            
        try:
            # If matrix is large, use lowrank SVD or fast projection
            if m > 2048 or n > 2048:
                q = min(target_dim, 64, min(m, n))
                U, s, V = torch.svd_lowrank(t_f, q=q)
                compressed = U @ torch.diag(s) @ V.T
            else:
                U, s, Vh = torch.linalg.svd(t_f, full_matrices=False)
                total_variance = torch.sum(s ** 2)
                cumulative_variance = torch.cumsum(s ** 2, dim=0) / max(total_variance.item(), 1e-8)
                comp_mask = cumulative_variance >= retain_ratio
                n_components = int(torch.argmax(comp_mask.int()).item()) + 1 if comp_mask.any() else s.numel()
                n_components = max(1, min(n_components, target_dim, s.numel()))
                
                U_reduced = U[:, :n_components]
                s_reduced = s[:n_components]
                Vh_reduced = Vh[:n_components, :]
                compressed = U_reduced @ torch.diag(s_reduced) @ Vh_reduced
        except Exception:
            # Graceful fallback to slice or linear interpolation
            if n > target_dim:
                compressed = t_f[:, :target_dim]
            else:
                pad = torch.zeros((m, target_dim - n), dtype=t_f.dtype, device=t_f.device)
                compressed = torch.cat([t_f, pad], dim=1)
                
        if compressed.shape[1] != target_dim:
            if compressed.shape[1] < target_dim:
                padding = target_dim - compressed.shape[1]
                pad = torch.zeros((compressed.shape[0], padding), dtype=compressed.dtype, device=compressed.device)
                compressed = torch.cat([compressed, pad], dim=1)
            else:
                compressed = compressed[:, :target_dim]
                
        if was_1d:
            compressed = compressed.squeeze(1)
            
        return compressed.to(orig_dtype).to(orig_device)
    
    @staticmethod
    def expand_via_svd(
        tensor: torch.Tensor,
        target_dim: int,
        expansion_method: str = "zero_pad"
    ) -> torch.Tensor:
        """
        Expand tensor to target dimension.
        
        Args:
            tensor: Input tensor [m, n]
            target_dim: Target dimension
            expansion_method: Method for expansion ("zero_pad", "interpolate", "repeat")
        
        Returns:
            Expanded tensor
        """
        current_dim = tensor.shape[-1]
        
        if current_dim >= target_dim:
            return tensor
        
        if expansion_method == "zero_pad":
            padding_size = target_dim - current_dim
            return torch.nn.functional.pad(tensor, (0, padding_size))
        
        elif expansion_method == "interpolate":
                                  
            scale_factor = target_dim / current_dim
            if tensor.dim() == 1:
                indices = torch.linspace(0, current_dim - 1, target_dim).long()
                return tensor[indices]
            else:
                                                                      
                original = tensor
                expanded = torch.zeros(*tensor.shape[:-1], target_dim, 
                                      dtype=tensor.dtype, device=tensor.device)
                for i in range(original.shape[0]):
                    indices = torch.linspace(0, current_dim - 1, target_dim).long()
                    expanded[i] = original[i][indices]
                return expanded
        
        elif expansion_method == "repeat":
                                 
            repeated = tensor.repeat(1, (target_dim + current_dim - 1) // current_dim)
            return repeated[:, :target_dim] if tensor.dim() > 1 else repeated[:target_dim]
        
        else:
                                     
            return DynamicSVD.expand_via_svd(tensor, target_dim, "zero_pad")
    
    @staticmethod
    def align_dimensions_svd(
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        target_dim: Optional[int] = None,
        method: str = "auto"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align two tensors to the same dimension using SVD.
        
        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            target_dim: Target dimension (auto-detected if None)
            method: Alignment method ("auto", "compress", "expand", "both")
        
        Returns:
            Tuple of (aligned_tensor_a, aligned_tensor_b)
        """
        dim_a = tensor_a.shape[-1]
        dim_b = tensor_b.shape[-1]
        
        if target_dim is None:
            target_dim = max(dim_a, dim_b)
        
        if method == "auto":
                                              
            if dim_a == dim_b:
                return tensor_a, tensor_b
            elif dim_a > target_dim and dim_b > target_dim:
                method = "compress"
            elif dim_a < target_dim and dim_b < target_dim:
                method = "expand"
            else:
                method = "both"
        
        if method == "compress":
                                               
            aligned_a = DynamicSVD.compress_via_svd(tensor_a, target_dim)
            aligned_b = DynamicSVD.compress_via_svd(tensor_b, target_dim)
        
        elif method == "expand":
                                             
            aligned_a = DynamicSVD.expand_via_svd(tensor_a, target_dim)
            aligned_b = DynamicSVD.expand_via_svd(tensor_b, target_dim)
        
        elif method == "both":
                                             
            if dim_a > target_dim:
                aligned_a = DynamicSVD.compress_via_svd(tensor_a, target_dim)
            else:
                aligned_a = DynamicSVD.expand_via_svd(tensor_a, target_dim)
            
            if dim_b > target_dim:
                aligned_b = DynamicSVD.compress_via_svd(tensor_b, target_dim)
            else:
                aligned_b = DynamicSVD.expand_via_svd(tensor_b, target_dim)
        
        else:
                                                
            aligned_a = DynamicSVD._simple_align(tensor_a, target_dim)
            aligned_b = DynamicSVD._simple_align(tensor_b, target_dim)
        
        return aligned_a, aligned_b
    
    @staticmethod
    def _simple_align(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Simple alignment using padding/truncation."""
        current_dim = tensor.shape[-1]
        
        if current_dim == target_dim:
            return tensor
        elif current_dim < target_dim:
            padding_size = target_dim - current_dim
            if tensor.dim() == 1:
                return torch.nn.functional.pad(tensor, (0, padding_size))
            else:
                padding = [0] * (2 * (tensor.dim() - 1)) + [0, padding_size]
                return torch.nn.functional.pad(tensor, padding)
        else:
            return tensor[..., :target_dim]
    
    @staticmethod
    def calculate_optimal_target_dim(
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        variance_threshold: float = 0.95
    ) -> int:
        """
        Calculate optimal target dimension based on variance preservation.
        
        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            variance_threshold: Minimum variance to preserve
        
        Returns:
            Optimal target dimension
        """
        dim_a = tensor_a.shape[-1]
        dim_b = tensor_b.shape[-1]
        
                                      
        max_dim = max(dim_a, dim_b)
        
                                                                       
        def variance_at_dim(tensor, target_dim):
            if tensor.shape[-1] <= target_dim:
                return 1.0
            compressed = DynamicSVD.compress_via_svd(tensor, target_dim, retain_ratio=variance_threshold)
            original_variance = torch.var(tensor).item()
            compressed_variance = torch.var(compressed).item()
            return compressed_variance / (original_variance + 1e-8)
        
                                                                         
        for target_dim in range(min(dim_a, dim_b), max_dim + 1):
            var_a = variance_at_dim(tensor_a, target_dim)
            var_b = variance_at_dim(tensor_b, target_dim)
            
            if var_a >= variance_threshold and var_b >= variance_threshold:
                return target_dim
        
                                                    
        return max_dim
