"""AutoML hyperparameter optimization for automated training configuration."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

@dataclass
class HyperparameterConfig:
    """Training hyperparameter configuration."""
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
                         
    use_fp16: bool = True
    use_bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
                     
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "optimizer": self.optimizer,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "use_fp16": self.use_fp16,
            "use_bf16": self.use_bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataloader_num_workers": self.dataloader_num_workers,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "HyperparameterConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

class HyperparameterOptimizer:
    """AutoML hyperparameter optimization system."""
    
    def __init__(
        self,
        search_space: dict[str, Any] | None = None,
        n_trials: int = 50,
        timeout: int | None = None,
        target_metric: str = "eval_loss",
        optimization_target: str = "minimize"
    ):
        self.search_space = search_space or self._default_search_space()
        self.n_trials = n_trials
        self.timeout = timeout
        self.target_metric = target_metric
        self.optimization_target = optimization_target
        self.best_config: HyperparameterConfig | None = None
        self.best_score: float | None = None
        self.study = None
        self.optimization_history: list[dict[str, Any]] = []
    
    def _default_search_space(self) -> dict[str, Any]:
        """Default hyperparameter search space."""
        return {
            "learning_rate": [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
            "batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4, 8],
            "num_train_epochs": [1, 2, 3, 5, 10],
            "warmup_steps": [0, 50, 100, 200, 500],
            "weight_decay": [0.0, 0.01, 0.05, 0.1],
            "max_grad_norm": [0.5, 1.0, 2.0, 5.0],
            "lr_scheduler_type": ["linear", "cosine", "cosine_with_restarts", "polynomial"],
            "optimizer": ["adamw", "adam", "sgd"],
            "beta1": [0.85, 0.9, 0.95],
            "beta2": [0.95, 0.98, 0.999],
            "epsilon": [1e-8, 1e-7, 1e-6],
            "use_fp16": [True, False],
            "gradient_checkpointing": [True, False],
            "lora_r": [4, 8, 16, 32],
            "lora_alpha": [8, 16, 32, 64],
            "lora_dropout": [0.0, 0.05, 0.1],
        }
    
    def sample_hyperparameters(self) -> HyperparameterConfig:
        """Sample random hyperparameters from search space."""
        config_dict = {}
        for param, values in self.search_space.items():
            config_dict[param] = random.choice(values)
        
        return HyperparameterConfig.from_dict(config_dict)
    
    def grid_search(
        self,
        evaluation_fn: Callable[[HyperparameterConfig], dict[str, float]],
        max_combinations: int = 100
    ) -> HyperparameterConfig:
        """
        Perform grid search over hyperparameter space.
        
        Args:
            evaluation_fn: Function to evaluate hyperparameter configuration
            max_combinations: Maximum number of combinations to try
        
        Returns:
            Best hyperparameter configuration
        """
        console = Console()
        console.print("[bold dark_orange]◈[/] Starting Grid Search")
        console.print(f"Max combinations: {max_combinations}\n")
        
        best_score = float('inf') if self.optimization_target == "minimize" else float('-inf')
        best_config = None
        
        for i in range(min(max_combinations, self._estimate_search_space_size())):
            config = self.sample_hyperparameters()
            metrics = evaluation_fn(config)
            score = metrics[self.target_metric]
            
            is_better = (
                (self.optimization_target == "minimize" and score < best_score) or
                (self.optimization_target == "maximize" and score > best_score)
            )
            
            if is_better:
                best_score = score
                best_config = config
                console.print(f"Trial {i}: New best! {self.target_metric}={score:.4f}")
            
            self.optimization_history.append({
                "trial": i,
                "config": config.to_dict(),
                "metrics": metrics,
                "is_best": is_better
            })
        
        self.best_config = best_config
        self.best_score = best_score
        
        console.print("\n[green]✓[/] Grid search complete")
        console.print(f"Best {self.target_metric}: {best_score:.4f}")
        
        return best_config
    
    def random_search(
        self,
        evaluation_fn: Callable[[HyperparameterConfig], dict[str, float]],
        n_trials: int = 50
    ) -> HyperparameterConfig:
        """
        Perform random search over hyperparameter space.
        
        Args:
            evaluation_fn: Function to evaluate hyperparameter configuration
            n_trials: Number of random trials
        
        Returns:
            Best hyperparameter configuration
        """
        console = Console()
        console.print("[bold dark_orange]◈[/] Starting Random Search")
        console.print(f"Trials: {n_trials}\n")
        
        best_score = float('inf') if self.optimization_target == "minimize" else float('-inf')
        best_config = None
        
        for i in range(n_trials):
            config = self.sample_hyperparameters()
            metrics = evaluation_fn(config)
            score = metrics[self.target_metric]
            
            is_better = (
                (self.optimization_target == "minimize" and score < best_score) or
                (self.optimization_target == "maximize" and score > best_score)
            )
            
            if is_better:
                best_score = score
                best_config = config
                console.print(f"Trial {i}: New best! {self.target_metric}={score:.4f}")
            
            self.optimization_history.append({
                "trial": i,
                "config": config.to_dict(),
                "metrics": metrics,
                "is_best": is_better
            })
        
        self.best_config = best_config
        self.best_score = best_score
        
        console.print("\n[green]✓[/] Random search complete")
        console.print(f"Best {self.target_metric}: {best_score:.4f}")
        
        return best_config
    
    def bayesian_optimization(
        self,
        evaluation_fn: Callable[[HyperparameterConfig], dict[str, float]],
        n_trials: int = 50,
        timeout: int | None = None
    ) -> HyperparameterConfig:
        """
        Perform Bayesian optimization using Optuna.
        
        Args:
            evaluation_fn: Function to evaluate hyperparameter configuration
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        
        Returns:
            Best hyperparameter configuration
        """
        if not OPTUNA_AVAILABLE:
            console = Console()
            console.print("[yellow]Optuna not available, falling back to random search[/]")
            return self.random_search(evaluation_fn, n_trials)
        
        console = Console()
        console.print("[bold dark_orange]◈[/] Starting Bayesian Optimization")
        console.print(f"Trials: {n_trials}")
        if timeout:
            console.print(f"Timeout: {timeout}s")
        console.print()
        
                             
        direction = "minimize" if self.optimization_target == "minimize" else "maximize"
        self.study = optuna.create_study(direction=direction)
        
        def objective(trial):
                                     
            config_dict = {}
            
            for param, values in self.search_space.items():
                if isinstance(values, list) and len(values) > 0:
                    if isinstance(values[0], bool):
                        config_dict[param] = trial.suggest_categorical(param, values)
                    elif isinstance(values[0], (int, float)):
                        if len(values) <= 10:                                   
                            config_dict[param] = trial.suggest_categorical(param, values)
                        else:                    
                            config_dict[param] = trial.suggest_float(param, min(values), max(values))
                    else:           
                        config_dict[param] = trial.suggest_categorical(param, values)
            
            config = HyperparameterConfig.from_dict(config_dict)
            metrics = evaluation_fn(config)
            score = metrics[self.target_metric]
            
                               
            self.optimization_history.append({
                "trial": trial.number,
                "config": config.to_dict(),
                "metrics": metrics,
                "is_best": False
            })
            
            return score
        
                          
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
                                
        best_params = self.study.best_params
        best_config = HyperparameterConfig.from_dict(best_params)
        self.best_config = best_config
        self.best_score = self.study.best_value
        
        console.print("\n[green]✓[/] Bayesian optimization complete")
        console.print(f"Best {self.target_metric}: {self.best_score:.4f}")
        console.print(f"Best trial: {self.study.best_trial.number}")
        
        return best_config
    
    def multi_objective_optimization(
        self,
        evaluation_fn: Callable[[HyperparameterConfig], dict[str, float]],
        n_trials: int = 50,
        objectives: list[str] | None = None
    ) -> list[HyperparameterConfig]:
        """
        Perform multi-objective optimization (Pareto front).
        
        Args:
            evaluation_fn: Function to evaluate hyperparameter configuration
            n_trials: Number of optimization trials
            objectives: List of objective metrics to optimize
        
        Returns:
            List of Pareto-optimal configurations
        """
        if objectives is None:
            objectives = ["eval_loss", "training_time"]
        if not OPTUNA_AVAILABLE:
            console = Console()
            console.print("[yellow]Optuna not available, falling back to single-objective[/]")
            return [self.random_search(evaluation_fn, n_trials)]
        
        console = Console()
        console.print("[bold dark_orange]◈[/] Starting Multi-Objective Optimization")
        console.print(f"Objectives: {objectives}")
        console.print(f"Trials: {n_trials}\n")
        
                                      
        self.study = optuna.create_study(directions=["minimize"] * len(objectives))
        
        def objective(trial):
            config_dict = {}
            for param, values in self.search_space.items():
                if isinstance(values, list) and len(values) > 0:
                    if isinstance(values[0], bool):
                        config_dict[param] = trial.suggest_categorical(param, values)
                    elif isinstance(values[0], (int, float)):
                        if len(values) <= 10:
                            config_dict[param] = trial.suggest_categorical(param, values)
                        else:
                            config_dict[param] = trial.suggest_float(param, min(values), max(values))
                    else:
                        config_dict[param] = trial.suggest_categorical(param, values)
            
            config = HyperparameterConfig.from_dict(config_dict)
            metrics = evaluation_fn(config)
            
            return [metrics[obj] for obj in objectives]
        
        self.study.optimize(objective, n_trials=n_trials)
        
                          
        pareto_trials = self.study.best_trials
        pareto_configs = []
        
        for trial in pareto_trials:
            config = HyperparameterConfig.from_dict(trial.params)
            pareto_configs.append(config)
        
        console.print("\n[green]✓[/] Multi-objective optimization complete")
        console.print(f"Pareto-optimal configurations: {len(pareto_configs)}")
        
        return pareto_configs
    
    def _estimate_search_space_size(self) -> int:
        """Estimate total search space size."""
        size = 1
        for values in self.search_space.values():
            size *= len(values)
        return size
    
    def save_results(self, output_path: Path) -> None:
        """Save optimization results to file."""
        results = {
            "best_config": self.best_config.to_dict() if self.best_config else None,
            "best_score": self.best_score,
            "optimization_history": self.optimization_history,
            "search_space": self.search_space,
            "target_metric": self.target_metric,
            "optimization_target": self.optimization_target,
        }
        
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Optimization results saved to {output_path}")
    
    def load_results(self, input_path: Path) -> None:
        """Load optimization results from file."""
        with input_path.open("r") as f:
            results = json.load(f)
        
        if results["best_config"]:
            self.best_config = HyperparameterConfig.from_dict(results["best_config"])
        self.best_score = results["best_score"]
        self.optimization_history = results["optimization_history"]
        self.search_space = results["search_space"]
        self.target_metric = results["target_metric"]
        self.optimization_target = results["optimization_target"]
        
        print(f"Optimization results loaded from {input_path}")

def create_training_hyperparameters(arch_config: HyperparameterConfig) -> dict[str, Any]:
    """Convert hyperparameter config to training arguments format."""
    return {
        "learning_rate": arch_config.learning_rate,
        "per_device_train_batch_size": arch_config.batch_size,
        "gradient_accumulation_steps": arch_config.gradient_accumulation_steps,
        "num_train_epochs": arch_config.num_train_epochs,
        "warmup_steps": arch_config.warmup_steps,
        "weight_decay": arch_config.weight_decay,
        "max_grad_norm": arch_config.max_grad_norm,
        "lr_scheduler_type": arch_config.lr_scheduler_type,
        "optim": arch_config.optimizer,
        "adam_beta1": arch_config.beta1,
        "adam_beta2": arch_config.beta2,
        "adam_epsilon": arch_config.epsilon,
        "fp16": arch_config.use_fp16,
        "bf16": arch_config.use_bf16,
        "gradient_checkpointing": arch_config.gradient_checkpointing,
        "dataloader_num_workers": arch_config.dataloader_num_workers,
        "logging_steps": arch_config.logging_steps,
        "save_steps": arch_config.save_steps,
        "eval_steps": arch_config.eval_steps,
    }