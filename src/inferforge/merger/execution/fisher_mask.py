"""
Fisher Information Matrix for Activation-Guided Merging.

Calculates weight importance to protect critical features during merging.
If a specific weight matrix is critical to a model's capabilities, mask it
so other models' weights do not overwrite or dilute it.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class FisherResult:
    """Result of Fisher Information computation."""
    fisher_matrix: torch.Tensor
    importance_scores: torch.Tensor
    critical_indices: List[int]
    threshold: float


class FisherImportanceMask:
    """Calculates and applies Fisher Information-based importance masking."""
    
    def __init__(self, importance_threshold: float = 0.7):
        """
        Initialize Fisher importance mask.
        
        Args:
            importance_threshold: Threshold for considering weights as critical
        """
        self.importance_threshold = importance_threshold
        self.cached_fisher: Dict[str, FisherResult] = {}
    
    def compute_fisher_information(
        self,
        model: torch.nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        num_samples: int = 100
    ) -> Dict[str, FisherResult]:
        """
        Compute Fisher Information Matrix for a model.
        
        Fisher Information measures how sensitive the model's loss is to
        changes in each parameter, indicating parameter importance.
        
        Args:
            model: PyTorch model
            dataloader: Optional dataloader for calibration data
            num_samples: Number of samples to use if no dataloader provided
        
        Returns:
            Dictionary mapping parameter names to Fisher results
        """
        model.eval()
        fisher_dict = {}
        
                                        
        for name, param in model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param)
        
                                                                     
        if dataloader is None:
            self._compute_fisher_random(model, fisher_dict, num_samples)
        else:
            self._compute_fisher_data(model, fisher_dict, dataloader)
        
                                
        results = {}
        for name, fisher in fisher_dict.items():
            importance_scores = torch.sqrt(fisher)
            threshold = torch.quantile(importance_scores, self.importance_threshold)
            critical_indices = torch.where(importance_scores > threshold)[0].tolist()
            
            results[name] = FisherResult(
                fisher_matrix=fisher,
                importance_scores=importance_scores,
                critical_indices=critical_indices,
                threshold=threshold.item()
            )
        
        self.cached_fisher = results
        return results
    
    def _compute_fisher_random(
        self,
        model: torch.nn.Module,
        fisher_dict: Dict[str, torch.Tensor],
        num_samples: int
    ) -> None:
        """Compute Fisher using random forward passes."""
        for _ in range(num_samples):
                                 
            dummy_input = torch.randn(1, 512)                              
            
                          
            try:
                output = model(dummy_input)
                loss = output.sum()                               
                
                                                
                loss.backward()
                
                                               
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher_dict[name] += param.grad.pow(2)
                        param.grad.zero_()
            except Exception:
                                                 
                continue
    
    def _compute_fisher_data(
        self,
        model: torch.nn.Module,
        fisher_dict: Dict[str, torch.Tensor],
        dataloader: torch.utils.data.DataLoader
    ) -> None:
        """Compute Fisher using real calibration data."""
        for batch in dataloader:
                          
            try:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                output = model(inputs)
                loss = output.sum()
                
                               
                loss.backward()
                
                                   
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher_dict[name] += param.grad.pow(2)
                        param.grad.zero_()
            except Exception:
                continue
    
    def create_importance_mask(
        self,
        fisher_result: FisherResult,
        protect_ratio: float = 0.8
    ) -> torch.Tensor:
        """
        Create binary mask for protecting important weights.
        
        Args:
            fisher_result: Fisher computation result
            protect_ratio: Ratio of top weights to protect
        
        Returns:
            Binary mask (1 = protect, 0 = allow modification)
        """
        importance = fisher_result.importance_scores.float()
        threshold = torch.quantile(importance, protect_ratio).item()
        mask = (importance > threshold).to(fisher_result.importance_scores.dtype)
        return mask
    
    def adaptive_merge(
        self,
        weights_a: torch.Tensor,
        weights_b: torch.Tensor,
        fisher_a: Optional[FisherResult] = None,
        fisher_b: Optional[FisherResult] = None,
        merge_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Perform adaptive merge using Fisher importance masking.
        
        Args:
            weights_a: First model weights
            weights_b: Second model weights
            fisher_a: Fisher result for model A
            fisher_b: Fisher result for model B
            merge_ratio: Base merge ratio
        
        Returns:
            Merged weights with importance-aware blending
        """
        if weights_a.shape != weights_b.shape:
            min_dims = tuple(min(s_a, s_b) for s_a, s_b in zip(weights_a.shape, weights_b.shape))
            slices = tuple(slice(0, d) for d in min_dims)
            weights_a = weights_a[slices]
            weights_b = weights_b[slices]
        
                                 
        if fisher_a is not None:
            mask_a = self.create_importance_mask(fisher_a, protect_ratio=0.9)
        else:
            mask_a = torch.zeros_like(weights_a)
        
        if fisher_b is not None:
            mask_b = self.create_importance_mask(fisher_b, protect_ratio=0.9)
        else:
            mask_b = torch.zeros_like(weights_b)
        
                                      
                                                   
        protected_a = mask_a * weights_a
        protected_b = mask_b * weights_b
        
                                          
        mergeable_a = (1 - mask_a) * weights_a
        mergeable_b = (1 - mask_b) * weights_b
        
        merged_mergeable = (1 - merge_ratio) * mergeable_a + merge_ratio * mergeable_b
        
                                              
        final_weights = protected_a + protected_b + merged_mergeable
        
        return final_weights
    
    def activation_calibration(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        calibration_prompts: List[str],
        tokenizer: Optional[object] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform activation calibration to identify critical neurons.
        
        Pass calibration prompts through source models and track which
        neurons activate most intensely for specific tasks.
        
        Args:
            model_a: First model
            model_b: Second model
            calibration_prompts: List of prompts for calibration
            tokenizer: Optional tokenizer for text inputs
        
        Returns:
            Dictionary of activation statistics
        """
        activation_stats = {"model_a": {}, "model_b": {}}
        
                                     
        def get_activation_hook(name, stats_dict):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation = output.detach()
                    stats_dict[name] = {
                        "mean": activation.mean().item(),
                        "std": activation.std().item(),
                        "max": activation.max().item(),
                        "min": activation.min().item(),
                        "shape": activation.shape
                    }
            return hook
        
                                        
        hooks_a = []
        hooks_b = []
        
        for name, module in model_a.named_modules():
            if len(list(module.children())) == 0:                     
                hook = module.register_forward_hook(get_activation_hook(name, activation_stats["model_a"]))
                hooks_a.append(hook)
        
        for name, module in model_b.named_modules():
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(get_activation_hook(name, activation_stats["model_b"]))
                hooks_b.append(hook)
        
                                 
        for prompt in calibration_prompts:
            try:
                if tokenizer is not None:
                    inputs = tokenizer(prompt, return_tensors="pt")
                    _ = model_a(**inputs)
                    _ = model_b(**inputs)
                else:
                                     
                    dummy = torch.randn(1, 512)
                    _ = model_a(dummy)
                    _ = model_b(dummy)
            except Exception:
                continue
        
                      
        for hook in hooks_a:
            hook.remove()
        for hook in hooks_b:
            hook.remove()
        
        return activation_stats
