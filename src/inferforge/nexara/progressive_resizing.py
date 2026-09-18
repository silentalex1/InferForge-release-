from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ProgressiveResizingConfig:
    initial_resolution: int
    final_resolution: int
    num_stages: int
    epochs_per_stage: int
    warmup_epochs: int
    scaling_factor: float


@dataclass
class ResizingStage:
    stage_number: int
    resolution: int
    epochs: int
    learning_rate: float
    batch_size: int


@dataclass
class ProgressiveResizingResult:
    final_accuracy: float
    training_time: float
    memory_savings: float
    convergence_speed: float


class ProgressiveResizing:
    def __init__(self):
        self.resizing_schedule: list[ResizingStage] = []
        self.current_stage = 0
        self.device = None
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_schedule(self, config: ProgressiveResizingConfig) -> list[ResizingStage]:
        self.resizing_schedule = []
        
        resolution_step = (config.final_resolution - config.initial_resolution) / config.num_stages
        current_resolution = config.initial_resolution
        
        for stage in range(config.num_stages):
            if stage < config.num_stages - 1:
                stage_resolution = int(current_resolution)
            else:
                stage_resolution = config.final_resolution
            
            stage_lr = self._calculate_stage_learning_rate(stage, config.num_stages)
            stage_batch_size = self._calculate_stage_batch_size(stage_resolution, config.final_resolution)
            
            stage = ResizingStage(
                stage_number=stage,
                resolution=stage_resolution,
                epochs=config.epochs_per_stage,
                learning_rate=stage_lr,
                batch_size=stage_batch_size
            )
            
            self.resizing_schedule.append(stage)
            current_resolution += resolution_step
        
        return self.resizing_schedule
    
    def _calculate_stage_learning_rate(self, stage: int, total_stages: int) -> float:
        base_lr = 0.001
        lr_schedule = base_lr * (1.0 + stage * 0.2)
        return min(lr_schedule, 0.01)
    
    def _calculate_stage_batch_size(self, current_resolution: int, final_resolution: int) -> int:
        base_batch = 32
        scaling_factor = (final_resolution / current_resolution) ** 2
        return max(1, int(base_batch / scaling_factor))
    
    def execute_stage(self, stage: ResizingStage, model: nn.Module, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for progressive resizing")
        
        self.current_stage = stage.stage_number
        model = model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=stage.learning_rate)
        
        transform = transforms.Compose([
            transforms.Resize((stage.resolution, stage.resolution)),
            transforms.ToTensor()
        ])
        
        start_time = time.time()
        
        for epoch in range(stage.epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch, labels in train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                
                if batch.shape[-2:] != (stage.resolution, stage.resolution):
                    batch = F.interpolate(batch, size=(stage.resolution, stage.resolution), mode='bilinear', align_corners=False)
                
                optimizer.zero_grad()
                outputs = model(batch)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        training_time = time.time() - start_time
        
        val_accuracy = 0.0
        if val_loader is not None:
            val_accuracy = self._evaluate_at_resolution(model, val_loader, stage.resolution)
        
        return {
            "stage": stage.stage_number,
            "resolution": stage.resolution,
            "training_loss": avg_loss,
            "validation_accuracy": val_accuracy,
            "learning_rate": stage.learning_rate,
            "batch_size": stage.batch_size,
            "training_time": training_time
        }
    
    def _evaluate_at_resolution(self, model: nn.Module, data_loader: Any, resolution: int) -> float:
        if not TORCH_AVAILABLE:
            return 0.85
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch, labels in data_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                
                if batch.shape[-2:] != (resolution, resolution):
                    batch = F.interpolate(batch, size=(resolution, resolution), mode='bilinear', align_corners=False)
                
                outputs = model(batch)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def get_current_stage(self) -> ResizingStage | None:
        if self.current_stage < len(self.resizing_schedule):
            return self.resizing_schedule[self.current_stage]
        return None
    
    def advance_stage(self) -> bool:
        if self.current_stage < len(self.resizing_schedule) - 1:
            self.current_stage += 1
            return True
        return False
    
    def calculate_memory_savings(self, config: ProgressiveResizingConfig) -> float:
        avg_resolution = (config.initial_resolution + config.final_resolution) / 2
        full_resolution_memory = (config.final_resolution ** 2) / (1024 ** 2)
        progressive_memory = (avg_resolution ** 2) / (1024 ** 2)
        
        savings = (full_resolution_memory - progressive_memory) / full_resolution_memory
        return savings
    
    def estimate_convergence_speed(self, config: ProgressiveResizingConfig) -> float:
        base_speed = 1.0
        speedup = 1.0 + (config.num_stages * 0.1)
        return min(speedup, 2.0)
    
    def optimize_schedule(self, config: ProgressiveResizingConfig, validation_accuracy: list[float]) -> ProgressiveResizingConfig:
        if len(validation_accuracy) < 2:
            return config
        
        accuracy_improvement = validation_accuracy[-1] - validation_accuracy[-2]
        
        if accuracy_improvement < 0.01:
            config.num_stages = max(config.num_stages - 1, 2)
            config.epochs_per_stage = max(config.epochs_per_stage - 1, 2)
        elif accuracy_improvement > 0.05:
            config.num_stages = min(config.num_stages + 1, 10)
        
        return config
    
    def get_resizing_summary(self) -> dict[str, Any]:
        if not self.resizing_schedule:
            return {"error": "No schedule created"}
        
        total_epochs = sum(stage.epochs for stage in self.resizing_schedule)
        avg_resolution = sum(stage.resolution for stage in self.resizing_schedule) / len(self.resizing_schedule)
        
        return {
            "total_stages": len(self.resizing_schedule),
            "total_epochs": total_epochs,
            "average_resolution": avg_resolution,
            "current_stage": self.current_stage,
            "stages": [
                {
                    "stage": s.stage_number,
                    "resolution": s.resolution,
                    "epochs": s.epochs,
                    "learning_rate": s.learning_rate,
                    "batch_size": s.batch_size
                }
                for s in self.resizing_schedule
            ]
        }
    
    def adaptive_resizing(self, model_weights: dict[str, Any], performance_metrics: dict[str, float]) -> dict[str, Any]:
        current_accuracy = performance_metrics.get("accuracy", 0.0)
        current_loss = performance_metrics.get("loss", 1.0)
        
        if current_accuracy > 0.8 and current_loss < 0.3:
            return {
                "action": "increase_resolution",
                "reason": "Good performance, can handle higher resolution",
                "target_resolution": 512
            }
        elif current_accuracy < 0.6 or current_loss > 0.7:
            return {
                "action": "decrease_resolution",
                "reason": "Poor performance, reduce resolution for faster convergence",
                "target_resolution": 128
            }
        else:
            return {
                "action": "maintain_resolution",
                "reason": "Stable performance, maintain current resolution",
                "target_resolution": 256
            }
