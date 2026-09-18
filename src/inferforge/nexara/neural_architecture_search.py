"""Neural Architecture Search for automated model architecture optimization."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError:
    raise ImportError("NAS requires: pip install torch transformers")

@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture search."""
    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    attention_type: str = "standard"                           
    use_rotary_embeddings: bool = True
    use_alibi: bool = False
    max_position_embeddings: int = 2048
    vocab_size: int = 50257
    layer_norm_epsilon: float = 1e-5
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_cache: bool = True
    
                       
    use_moe: bool = False                      
    num_experts: int = 8
    expert_factor: int = 2
    use_switch_transformer: bool = False
    
                
    use_gradient_checkpointing: bool = False
    use_recompute: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "attention_type": self.attention_type,
            "use_rotary_embeddings": self.use_rotary_embeddings,
            "use_alibi": self.use_alibi,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "use_cache": self.use_cache,
            "use_moe": self.use_moe,
            "num_experts": self.num_experts,
            "expert_factor": self.expert_factor,
            "use_switch_transformer": self.use_switch_transformer,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_recompute": self.use_recompute,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ArchitectureConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

class NeuralArchitectureSearch:
    """Neural Architecture Search for automated model optimization."""
    
    def __init__(
        self,
        search_space: dict[str, Any] | None = None,
        max_iterations: int = 50,
        evaluation_budget: int = 10,
        target_metric: str = "perplexity",
        optimization_target: str = "minimize"                        
    ):
        self.search_space = search_space or self._default_search_space()
        self.max_iterations = max_iterations
        self.evaluation_budget = evaluation_budget
        self.target_metric = target_metric
        self.optimization_target = optimization_target
        self.best_architecture: ArchitectureConfig | None = None
        self.best_score: float | None = None
        self.search_history: list[dict[str, Any]] = []
    
    def _default_search_space(self) -> dict[str, Any]:
        """Default search space for architecture search."""
        return {
            "num_layers": [6, 8, 12, 16, 24, 32],
            "hidden_size": [256, 512, 768, 1024, 1280, 1536, 2048],
            "num_attention_heads": [4, 8, 12, 16, 20, 24, 32],
            "intermediate_size": [1024, 2048, 3072, 4096, 5120, 6144],
            "hidden_act": ["gelu", "relu", "swish", "silu"],
            "attention_type": ["standard", "flash"],
            "use_rotary_embeddings": [True, False],
            "use_alibi": [True, False],
            "max_position_embeddings": [1024, 2048, 4096, 8192],
            "dropout": [0.0, 0.1, 0.2],
            "attention_dropout": [0.0, 0.1],
            "use_moe": [True, False],
            "num_experts": [4, 8, 16],
            "use_gradient_checkpointing": [True, False],
        }
    
    def sample_architecture(self) -> ArchitectureConfig:
        """Sample a random architecture from search space."""
        config_dict = {}
        for param, values in self.search_space.items():
            config_dict[param] = random.choice(values)
        
                                      
        if "hidden_size" in config_dict and "num_attention_heads" in config_dict:
            hidden_size = config_dict["hidden_size"]
            num_heads = config_dict["num_attention_heads"]
                                                          
            if hidden_size % num_heads != 0:
                config_dict["hidden_size"] = (hidden_size // num_heads) * num_heads
        
        if "hidden_size" in config_dict and "intermediate_size" not in config_dict:
            config_dict["intermediate_size"] = config_dict["hidden_size"] * 4
        
        return ArchitectureConfig.from_dict(config_dict)
    
    def estimate_complexity(self, config: ArchitectureConfig) -> dict[str, float]:
        """Estimate computational complexity of architecture."""
                                          
        params_per_layer = (
            12 * config.hidden_size * config.hidden_size +             
            2 * config.hidden_size * config.intermediate_size       
        )
        total_params = params_per_layer * config.num_layers
        
                                       
        seq_length = config.max_position_embeddings
        attention_flops = 2 * seq_length * seq_length * config.hidden_size * config.num_layers
        ffn_flops = 2 * seq_length * config.intermediate_size * config.hidden_size * config.num_layers
        total_flops = attention_flops + ffn_flops
        
                                        
        memory_mb = (total_params * 4) / (1024 * 1024)                  
        
        return {
            "parameters": total_params,
            "flops": total_flops,
            "memory_mb": memory_mb,
            "complexity_score": total_params / 1e6,                          
        }
    
    def evaluate_architecture(
        self,
        config: ArchitectureConfig,
        evaluation_fn: Callable[[ArchitectureConfig], dict[str, float]] | None = None
    ) -> dict[str, float]:
        """
        Evaluate architecture using provided evaluation function or proxy metrics.
        
        Args:
            config: Architecture configuration to evaluate
            evaluation_fn: Custom evaluation function that returns metrics
        
        Returns:
            Dictionary of evaluation metrics
        """
        if evaluation_fn:
            return evaluation_fn(config)
        
                                                             
        complexity = self.estimate_complexity(config)
        
                                   
        efficiency_score = 1.0 / (complexity["complexity_score"] * 0.1 + 1.0)
        
                                         
        quality_score = 1.0
        
                                        
        if 8 <= config.num_layers <= 24:
            quality_score *= 1.2
        elif config.num_layers > 32:
            quality_score *= 0.8
        
                                      
        if 512 <= config.hidden_size <= 1536:
            quality_score *= 1.2
        
                                            
        if config.use_rotary_embeddings:
            quality_score *= 1.1
        
                                    
        if config.dropout > 0.15:
            quality_score *= 0.9
        
                                                    
        if config.use_moe and config.hidden_size >= 1024:
            quality_score *= 1.15
        
        metrics = {
            "complexity_score": complexity["complexity_score"],
            "efficiency_score": efficiency_score,
            "quality_score": quality_score,
            "overall_score": quality_score * efficiency_score,
            "parameters": complexity["parameters"],
            "flops": complexity["flops"],
            "memory_mb": complexity["memory_mb"],
        }
        
        return metrics
    
    def search(
        self,
        evaluation_fn: Callable[[ArchitectureConfig], dict[str, float]] | None = None,
        seed: int = 42
    ) -> ArchitectureConfig:
        """
        Perform neural architecture search.
        
        Args:
            evaluation_fn: Custom evaluation function
            seed: Random seed for reproducibility
        
        Returns:
            Best architecture found
        """
        random.seed(seed)
        
        print("Starting Neural Architecture Search...")
        print(f"Search space size: {self._estimate_search_space_size()}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Evaluation budget: {self.evaluation_budget}")
        
        for iteration in range(self.max_iterations):
                                 
            config = self.sample_architecture()
            
                      
            metrics = self.evaluate_architecture(config, evaluation_fn)
            score = metrics[self.target_metric]
            
                         
            if self.best_score is None:
                is_better = True
            elif self.optimization_target == "minimize":
                is_better = score < self.best_score
            else:
                is_better = score > self.best_score
            
            if is_better:
                self.best_score = score
                self.best_architecture = config
                print(f"Iteration {iteration}: New best! {self.target_metric}={score:.4f}")
            
                            
            self.search_history.append({
                "iteration": iteration,
                "config": config.to_dict(),
                "metrics": metrics,
                "is_best": is_better
            })
            
                                                          
            if iteration >= self.evaluation_budget:
                print(f"Evaluation budget reached at iteration {iteration}")
                break
        
        print(f"Search complete. Best {self.target_metric}: {self.best_score:.4f}")
        return self.best_architecture
    
    def _estimate_search_space_size(self) -> int:
        """Estimate total search space size."""
        size = 1
        for values in self.search_space.values():
            size *= len(values)
        return size
    
    def genetic_search(
        self,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        evaluation_fn: Callable[[ArchitectureConfig], dict[str, float]] | None = None
    ) -> ArchitectureConfig:
        """
        Perform genetic algorithm-based architecture search.
        
        Args:
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            evaluation_fn: Custom evaluation function
        
        Returns:
            Best architecture found
        """
        print("Starting Genetic Architecture Search...")
        print(f"Population size: {population_size}")
        print(f"Generations: {generations}")
        
                               
        population = [self.sample_architecture() for _ in range(population_size)]
        
        for generation in range(generations):
                                 
            scored_population = []
            for config in population:
                metrics = self.evaluate_architecture(config, evaluation_fn)
                score = metrics[self.target_metric]
                scored_population.append((config, score))
            
                           
            if self.optimization_target == "minimize":
                scored_population.sort(key=lambda x: x[1])
            else:
                scored_population.sort(key=lambda x: x[1], reverse=True)
            
                         
            best_config, best_score = scored_population[0]
            if self.best_score is None or (
                (self.optimization_target == "minimize" and best_score < self.best_score) or
                (self.optimization_target == "maximize" and best_score > self.best_score)
            ):
                self.best_score = best_score
                self.best_architecture = best_config
                print(f"Generation {generation}: New best! {self.target_metric}={best_score:.4f}")
            
                                      
            top_half = [config for config, score in scored_population[:population_size // 2]]
            
                                    
            new_population = top_half.copy()
            
            while len(new_population) < population_size:
                if random.random() < crossover_rate:
                               
                    parent1, parent2 = random.sample(top_half, 2)
                    child = self._crossover(parent1, parent2)
                else:
                    child = random.choice(top_half)
                
                          
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        print(f"Genetic search complete. Best {self.target_metric}: {self.best_score:.4f}")
        return self.best_architecture
    
    def _crossover(self, parent1: ArchitectureConfig, parent2: ArchitectureConfig) -> ArchitectureConfig:
        """Crossover two parent architectures."""
        child_dict = parent1.to_dict()
        parent2_dict = parent2.to_dict()
        
        for key in child_dict:
            if random.random() < 0.5:
                child_dict[key] = parent2_dict[key]
        
        return ArchitectureConfig.from_dict(child_dict)
    
    def _mutate(self, config: ArchitectureConfig) -> ArchitectureConfig:
        """Mutate an architecture."""
        config_dict = config.to_dict()
        
                                               
        param_to_mutate = random.choice(list(self.search_space.keys()))
        
        if param_to_mutate in config_dict:
                                                           
            current_value = config_dict[param_to_mutate]
            possible_values = self.search_space[param_to_mutate]
            if len(possible_values) > 1:
                new_values = [v for v in possible_values if v != current_value]
                if new_values:
                    config_dict[param_to_mutate] = random.choice(new_values)
        
        return ArchitectureConfig.from_dict(config_dict)
    
    def save_search_results(self, output_path: Path) -> None:
        """Save search results to file."""
        results = {
            "best_architecture": self.best_architecture.to_dict() if self.best_architecture else None,
            "best_score": self.best_score,
            "search_history": self.search_history,
            "search_space": self.search_space,
        }
        
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Search results saved to {output_path}")
    
    def load_search_results(self, input_path: Path) -> None:
        """Load search results from file."""
        with input_path.open("r") as f:
            results = json.load(f)
        
        if results["best_architecture"]:
            self.best_architecture = ArchitectureConfig.from_dict(results["best_architecture"])
        self.best_score = results["best_score"]
        self.search_history = results["search_history"]
        self.search_space = results["search_space"]
        
        print(f"Search results loaded from {input_path}")

def create_transformer_config(arch_config: ArchitectureConfig) -> dict[str, Any]:
    """Convert architecture config to Hugging Face config format."""
    return {
        "architectures": ["GPTNeoXForCausalLM"],
        "attention_type": arch_config.attention_type,
        "hidden_size": arch_config.hidden_size,
        "num_attention_heads": arch_config.num_attention_heads,
        "num_hidden_layers": arch_config.num_layers,
        "intermediate_size": arch_config.intermediate_size,
        "hidden_act": arch_config.hidden_act,
        "rotary_emb": arch_config.use_rotary_embeddings,
        "rotary_pct": 0.25,
        "use_cache": arch_config.use_cache,
        "vocab_size": arch_config.vocab_size,
        "max_position_embeddings": arch_config.max_position_embeddings,
        "layer_norm_epsilon": arch_config.layer_norm_epsilon,
        "hidden_dropout": arch_config.dropout,
        "attention_dropout": arch_config.attention_dropout,
    }