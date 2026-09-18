"""Hypernetwork training for adaptive model conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class HypernetworkConfig:
    input_dim: int = 512
    hidden_dim: int = 256
    output_dim: int = 512
    num_layers: int = 3
    condition_dim: int = 128
    use_layer_norm: bool = True
    use_residual: bool = True
    activation: str = "gelu"


class Hypernetwork(nn.Module):
    def __init__(self, config: HypernetworkConfig):
        super().__init__()
        self.config = config
        
        self.condition_embedding = nn.Linear(config.condition_dim, config.hidden_dim)
        
        layers = []
        input_dim = config.input_dim + config.hidden_dim
        
        for i in range(config.num_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(config.hidden_dim))
            
            if config.activation == "gelu":
                layers.append(nn.GELU())
            elif config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "swish":
                layers.append(nn.SiLU())
            
            if config.use_residual and i < config.num_layers - 1:
                layers.append(nn.Dropout(0.1))
            
            input_dim = config.hidden_dim
        
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, base_features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        condition_emb = self.condition_embedding(condition)
        condition_emb = condition_emb.unsqueeze(1).expand(-1, base_features.size(1), -1)
        
        combined = torch.cat([base_features, condition_emb], dim=-1)
        adaptation = self.network(combined)
        
        return base_features + adaptation


class HypernetworkTrainer:
    def __init__(self, hypernetwork: Hypernetwork, base_model: nn.Module, config: HypernetworkConfig):
        self.hypernetwork = hypernetwork
        self.base_model = base_model
        self.config = config
        self.device = None
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.hypernetwork = self.hypernetwork.to(self.device)
            self.base_model = self.base_model.to(self.device)
    
    def train_step(self, batch: Any, condition: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.hypernetwork.train()
        self.base_model.eval()
        
        batch = batch.to(self.device)
        condition = condition.to(self.device)
        targets = targets.to(self.device)
        
        with torch.no_grad():
            base_features = self.base_model(batch)
        
        adapted_features = self.hypernetwork(base_features, condition)
        
        loss = F.mse_loss(adapted_features, targets)
        
        return loss.item(), loss
    
    def train_epoch(self, train_loader: Any, condition_loader: Any, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        total_loss = 0.0
        num_batches = 0
        
        for (batch, targets), condition in zip(train_loader, condition_loader):
            loss, loss_tensor = self.train_step(batch, condition, targets)
            
            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()
            
            total_loss += loss
            num_batches += 1
        
        return {
            "avg_loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "num_batches": num_batches
        }
    
    def evaluate(self, val_loader: Any, condition_loader: Any) -> dict[str, float]:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.hypernetwork.eval()
        self.base_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for (batch, targets), condition in zip(val_loader, condition_loader):
                batch = batch.to(self.device)
                condition = condition.to(self.device)
                targets = targets.to(self.device)
                
                base_features = self.base_model(batch)
                adapted_features = self.hypernetwork(base_features, condition)
                
                loss = F.mse_loss(adapted_features, targets)
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "avg_loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "num_batches": num_batches
        }
