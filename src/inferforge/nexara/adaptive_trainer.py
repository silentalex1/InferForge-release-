"""Adaptive training engine with dynamic learning rate and curriculum."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingState:
    """Current training state with metrics."""
    epoch: int = 0
    step: int = 0
    learning_rate: float = 2e-5
    loss: float = 0.0
    gradient_norm: float = 0.0
    examples_seen: int = 0
    best_loss: float = float("inf")
    plateau_count: int = 0
    warmup_complete: bool = False


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive training."""
    min_lr: float = 1e-6
    max_lr: float = 1e-3
    warmup_steps: int = 100
    patience: int = 5
    lr_decay_factor: float = 0.5
    gradient_clip_threshold: float = 1.0
    difficulty_increase_threshold: float = 0.95
    difficulty_decrease_threshold: float = 0.70


class AdaptiveTrainingEngine:
    """Engine for adaptive training with dynamic adjustments."""
    
    def __init__(self, config: AdaptiveConfig | None = None):
        self.config = config or AdaptiveConfig()
        self.state = TrainingState()
        self.loss_history: list[float] = []
        self.lr_history: list[float] = []
    
    def update_learning_rate(
        self,
        current_loss: float,
        gradient_norm: float,
    ) -> float:
        """Dynamically adjust learning rate based on training metrics."""
        self.state.loss = current_loss
        self.state.gradient_norm = gradient_norm
        self.state.step += 1
        
        # Warmup phase
        if not self.state.warmup_complete:
            if self.state.step < self.config.warmup_steps:
                self.state.learning_rate = self.config.max_lr * (
                    self.state.step / self.config.warmup_steps
                )
                self.lr_history.append(self.state.learning_rate)
                return self.state.learning_rate
            else:
                self.state.warmup_complete = True
        
        # Check for plateau
        self.loss_history.append(current_loss)
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            if current_loss < self.state.best_loss:
                self.state.best_loss = current_loss
                self.state.plateau_count = 0
            else:
                # Check if we're stuck
                if max(recent_losses) - min(recent_losses) < 0.01:
                    self.state.plateau_count += 1
        
        # Reduce learning rate on plateau
        if self.state.plateau_count >= self.config.patience:
            self.state.learning_rate *= self.config.lr_decay_factor
            self.state.learning_rate = max(
                self.state.learning_rate,
                self.config.min_lr,
            )
            self.state.plateau_count = 0
        
        # Gradient-based adjustment
        if gradient_norm > self.config.gradient_clip_threshold * 2:
            # Gradients too large, reduce LR
            self.state.learning_rate *= 0.9
        elif gradient_norm < self.config.gradient_clip_threshold * 0.1:
            # Gradients too small, increase LR slightly
            self.state.learning_rate *= 1.05
            self.state.learning_rate = min(
                self.state.learning_rate,
                self.config.max_lr,
            )
        
        self.lr_history.append(self.state.learning_rate)
        return self.state.learning_rate
    
    def should_clip_gradients(self) -> bool:
        """Check if gradients should be clipped."""
        return self.state.gradient_norm > self.config.gradient_clip_threshold
    
    def get_curriculum_difficulty(self, accuracy: float) -> str:
        """Determine curriculum difficulty level based on accuracy."""
        if accuracy >= self.config.difficulty_increase_threshold:
            return "hard"
        elif accuracy <= self.config.difficulty_decrease_threshold:
            return "easy"
        else:
            return "medium"
    
    def compute_loss_weight(self, example_difficulty: str) -> float:
        """Compute loss weight based on example difficulty and current performance."""
        difficulty_weights = {
            "easy": 0.5,
            "medium": 1.0,
            "hard": 1.5,
        }
        return difficulty_weights.get(example_difficulty, 1.0)
    
    def get_batch_size_recommendation(
        self,
        available_memory_gb: float,
        model_size_gb: float,
    ) -> int:
        """Recommend batch size based on available memory."""
        # Reserve 2GB for system
        usable_memory = available_memory_gb - 2.0
        
        # Estimate memory per example (very rough)
        memory_per_example = model_size_gb * 0.1
        
        recommended = int(usable_memory / memory_per_example)
        recommended = max(1, min(recommended, 32))  # Clamp to [1, 32]
        
        return recommended
    
    def should_save_checkpoint(self) -> bool:
        """Determine if checkpoint should be saved."""
        # Save on best loss or every N steps
        if self.state.loss <= self.state.best_loss:
            return True
        if self.state.step % 500 == 0:
            return True
        return False
    
    def save_checkpoint(self, path: Path) -> None:
        """Save training state to checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "state": {
                "epoch": self.state.epoch,
                "step": self.state.step,
                "learning_rate": self.state.learning_rate,
                "loss": self.state.loss,
                "best_loss": self.state.best_loss,
                "plateau_count": self.state.plateau_count,
                "examples_seen": self.state.examples_seen,
            },
            "history": {
                "loss": self.loss_history[-100:],  # Last 100
                "lr": self.lr_history[-100:],
            },
            "timestamp": time.time(),
        }
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self, path: Path) -> None:
        """Load training state from checkpoint."""
        if not path.exists():
            return
        
        with path.open("r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        
        state_data = checkpoint.get("state", {})
        self.state.epoch = state_data.get("epoch", 0)
        self.state.step = state_data.get("step", 0)
        self.state.learning_rate = state_data.get("learning_rate", 2e-5)
        self.state.loss = state_data.get("loss", 0.0)
        self.state.best_loss = state_data.get("best_loss", float("inf"))
        self.state.plateau_count = state_data.get("plateau_count", 0)
        self.state.examples_seen = state_data.get("examples_seen", 0)
        
        history = checkpoint.get("history", {})
        self.loss_history = history.get("loss", [])
        self.lr_history = history.get("lr", [])
    
    def get_training_metrics(self) -> dict[str, Any]:
        """Get current training metrics."""
        return {
            "epoch": self.state.epoch,
            "step": self.state.step,
            "learning_rate": self.state.learning_rate,
            "current_loss": self.state.loss,
            "best_loss": self.state.best_loss,
            "gradient_norm": self.state.gradient_norm,
            "examples_seen": self.state.examples_seen,
            "plateau_count": self.state.plateau_count,
            "loss_trend": self._calculate_trend(),
        }
    
    def _calculate_trend(self) -> str:
        """Calculate loss trend from recent history."""
        if len(self.loss_history) < 5:
            return "insufficient_data"
        
        recent = self.loss_history[-5:]
        if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
            return "improving"
        elif all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            return "degrading"
        else:
            return "stable"
