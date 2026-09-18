"""
Procrustes Alignment for Architecture-Agnostic Model Merging.

Uses orthogonal Procrustes rotation matrices to mathematically align the 
vector spaces of different models before combining them. This allows merging
models with different architectures by rotating their weight matrices to 
"speak the same geometric language."
"""

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment


class ProcrustesAligner:
    """Aligns different model architectures using Procrustes analysis."""
    
    @staticmethod
    def orthogonal_procrustes(
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute optimal orthogonal transformation to align source to target.
        
        Args:
            source: Source weight matrix [d1, d2]
            target: Target weight matrix [d1, d2]
        
        Returns:
            Tuple of (rotation_matrix, aligned_source)
        """
        source_f = source.detach().float().cpu()
        target_f = target.detach().float().cpu()
        
        if source_f.ndim == 1:
            source_f = source_f.reshape(-1, 1)
        if target_f.ndim == 1:
            target_f = target_f.reshape(-1, 1)
        
        source_np = source_f.numpy()
        target_np = target_f.numpy()
        
        try:
            R, _ = orthogonal_procrustes(source_np, target_np)
            aligned_source_np = source_np @ R
            rotation_matrix = torch.from_numpy(R).to(source.dtype).to(source.device)
            aligned_source = torch.from_numpy(aligned_source_np).to(source.dtype).to(source.device)
            return rotation_matrix, aligned_source
        except Exception:
            return torch.eye(source.shape[-1], dtype=source.dtype, device=source.device), source
    
    @staticmethod
    def central_kernel_alignment(
        matrix_a: torch.Tensor,
        matrix_b: torch.Tensor,
        layer_name: str = ""
    ) -> float:
        """
        Compute Central Kernel Alignment (CKA) similarity between two weight matrices.
        
        CKA measures the similarity of neural network representations and is
        architecture-agnostic, making it perfect for matching layers across
        different model architectures.
        
        Args:
            matrix_a: First weight matrix
            matrix_b: Second weight matrix
            layer_name: Optional layer identifier for logging
        
        Returns:
            CKA similarity score (0.0 to 1.0)
        """
        a_f = matrix_a.detach().float()
        b_f = matrix_b.detach().float()
        
        if a_f.ndim == 1:
            a_f = a_f.unsqueeze(1)
        if b_f.ndim == 1:
            b_f = b_f.unsqueeze(1)
            
        # Subsample rows to avoid high memory usage
        max_samples = 1024
        if a_f.shape[0] > max_samples:
            step = a_f.shape[0] // max_samples
            a_f = a_f[::step][:max_samples]
        if b_f.shape[0] > max_samples:
            step = b_f.shape[0] // max_samples
            b_f = b_f[::step][:max_samples]
            
        min_rows = min(a_f.shape[0], b_f.shape[0])
        a_f = a_f[:min_rows]
        b_f = b_f[:min_rows]
        
        # Center features
        a_c = a_f - a_f.mean(dim=0, keepdim=True)
        b_c = b_f - b_f.mean(dim=0, keepdim=True)
        
        # Fast Linear CKA: ||B_c^T A_c||_F^2 / (||A_c^T A_c||_F * ||B_c^T B_c||_F)
        cross = torch.matmul(b_c.T, a_c)
        sim_ab = torch.sum(cross ** 2)
        
        self_a = torch.matmul(a_c.T, a_c)
        sim_aa = torch.sum(self_a ** 2)
        
        self_b = torch.matmul(b_c.T, b_c)
        sim_bb = torch.sum(self_b ** 2)
        
        denom = torch.sqrt(sim_aa * sim_bb)
        if denom < 1e-8:
            return 0.0
            
        cka = (sim_ab / denom).item()
        return float(max(0.0, min(1.0, cka)))
    
    @staticmethod
    def Hungarian_weight_matching(
        weights_a: torch.Tensor,
        weights_b: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Match neurons between layers using Hungarian algorithm.
        
        Neutral networks can learn the same concepts but store them in different
        neuron layouts. This algorithm reorders neurons to align semantically.
        
        Args:
            weights_a: First layer weights [out_features, in_features]
            weights_b: Second layer weights [out_features, in_features]
        
        Returns:
            Tuple of (reordered_weights_b, permutation_indices)
        """
                                                   
                                                      
        weights_a_norm = torch.nn.functional.normalize(weights_a, dim=1)
        weights_b_norm = torch.nn.functional.normalize(weights_b, dim=1)
        
                                   
        min_features = min(weights_a.shape[0], weights_b.shape[0])
        weights_a_norm = weights_a_norm[:min_features]
        weights_b_norm = weights_b_norm[:min_features]
        
                                          
        similarity_matrix = torch.matmul(weights_a_norm, weights_b_norm.T)
        
                                                                       
        cost_matrix = -similarity_matrix.detach().float().cpu().numpy()
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        reordered_weights_b = weights_b.clone()
        reordered_weights_b[:len(col_ind)] = weights_b[col_ind]
        
        return reordered_weights_b, col_ind
    
    @staticmethod
    def dynamic_layer_mapping(
        layers_a: dict,
        layers_b: dict,
        cka_threshold: float = 0.7
    ) -> dict:
        """
        Map layers between different architectures using CKA similarity.
        
        If Model A has 80 layers and Model B has 128 layers, this function
        calculates semantic similarities and maps only layers that serve
        equivalent structural purposes.
        
        Args:
            layers_a: Dictionary of layer_name -> weight_tensor for model A
            layers_b: Dictionary of layer_name -> weight_tensor for model B
            cka_threshold: Minimum CKA score to consider layers as matching
        
        Returns:
            Dictionary mapping layer_a_name -> layer_b_name (or None if no match)
        """
        layer_mapping = {}
        
        for layer_a_name, weights_a in layers_a.items():
            best_match = None
            best_score = 0.0
            
            for layer_b_name, weights_b in layers_b.items():
                try:
                                              
                    cka_score = ProcrustesAligner.central_kernel_alignment(
                        weights_a, weights_b, f"{layer_a_name} vs {layer_b_name}"
                    )
                    
                    if cka_score > best_score and cka_score > cka_threshold:
                        best_score = cka_score
                        best_match = layer_b_name
                except Exception as e:
                                             
                    continue
            
            layer_mapping[layer_a_name] = best_match
        
        return layer_mapping
    
    @staticmethod
    def align_mismatched_dimensions(
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        target_dim: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align tensors with different dimensions using padding/truncation.
        
        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            target_dim: Target dimension (auto-detected if None)
        
        Returns:
            Tuple of (aligned_tensor_a, aligned_tensor_b)
        """
        if target_dim is None:
            target_dim = max(tensor_a.shape[-1], tensor_b.shape[-1])
        
                                                     
        def align_tensor(tensor, target_dim):
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
        
        aligned_a = align_tensor(tensor_a, target_dim)
        aligned_b = align_tensor(tensor_b, target_dim)
        
        return aligned_a, aligned_b
