"""Advanced training engine with state-of-the-art optimization techniques."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.amp import GradScaler, autocast
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        CosineAnnealingWarmRestarts,
        LinearLR,
        OneCycleLR,
        SequentialLR,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        from torch.distributed import destroy_process_group, init_process_group
    except ImportError:  # distributed unavailable (e.g. some Windows builds)
        def init_process_group(*a: Any, **k: Any) -> None: ...
        def destroy_process_group(*a: Any, **k: Any) -> None: ...

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # optional — tensorboard not installed


@dataclass
class AdvancedTrainingConfig:
    optimizer_type: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    scheduler_type: str = "cosine_warmup"
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    label_smoothing: float = 0.0
    dropout_rate: float = 0.1
    use_mixup: bool = False
    use_cutmix: bool = False
    focal_loss_gamma: float = 0.0
    contrastive_temperature: float = 0.07
    use_early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    use_distributed: bool = False
    use_fsdp: bool = False
    log_tensorboard: bool = True
    log_wandb: bool = False
    wandb_project: str = "nexara"


@dataclass
class TrainingMetrics:
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    gradient_norm: float
    throughput: float
    memory_usage: float


class AdvancedOptimizer:
    def __init__(self):
        self.optimizers = {}
    
    def get_optimizer(self, model: nn.Module, config: AdvancedTrainingConfig) -> optim.Optimizer:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        optimizer_type = config.optimizer_type.lower()
        
        if optimizer_type == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == "adam":
            return optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=config.lr,
                momentum=0.9,
                weight_decay=config.weight_decay,
                nesterov=True
            )
        elif optimizer_type == "nadam":
            return optim.NAdam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
        elif optimizer_type == "radam":
            return optim.RAdam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
        elif optimizer_type == "lion":
            return self._lion_optimizer(model, config)
        elif optimizer_type == "adafactor":
            return self._adafactor_optimizer(model, config)
        elif optimizer_type == "sophia":
            return self._sophia_optimizer(model, config)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _lion_optimizer(self, model: nn.Module, config: AdvancedTrainingConfig) -> optim.Optimizer:
        class Lion(optim.Optimizer):
            def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
                defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
                super().__init__(params, defaults)
            
            @torch.no_grad()
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        
                        grad = p.grad
                        state = self.state[p]
                        
                        if len(state) == 0:
                            state['exp_avg'] = torch.zeros_like(p)
                        
                        exp_avg = state['exp_avg']
                        beta1, beta2 = group['betas']
                        
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        
                        update = exp_avg.clone()
                        update.sign_().mul_(group['lr'])
                        
                        if group['weight_decay'] != 0:
                            p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                        
                        p.add_(update)
                        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                return loss
        
        return Lion(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay
        )
    
    def _adafactor_optimizer(self, model: nn.Module, config: AdvancedTrainingConfig) -> optim.Optimizer:
        try:
            from transformers import Adafactor
            return Adafactor(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                relative_step=False
            )
        except ImportError:
            return optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    def _sophia_optimizer(self, model: nn.Module, config: AdvancedTrainingConfig) -> optim.Optimizer:
        class Sophia(optim.Optimizer):
            def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), weight_decay=0.01, rho=0.05):
                defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, rho=rho)
                super().__init__(params, defaults)
            
            @torch.no_grad()
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        
                        grad = p.grad
                        state = self.state[p]
                        
                        if len(state) == 0:
                            state['exp_avg'] = torch.zeros_like(p)
                            state['exp_avg_sq'] = torch.zeros_like(p)
                        
                        exp_avg = state['exp_avg']
                        exp_avg_sq = state['exp_avg_sq']
                        beta1, beta2 = group['betas']
                        
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        
                        hessian = exp_avg_sq.sqrt().add_(group['rho'])
                        
                        update = exp_avg / hessian
                        update.mul_(group['lr'])
                        
                        if group['weight_decay'] != 0:
                            p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                        
                        p.add_(-update)
                
                return loss
        
        return Sophia(
            model.parameters(),
            lr=config.lr,
            betas=(0.965, 0.99),
            weight_decay=config.weight_decay,
            rho=0.05
        )


class AdvancedScheduler:
    def get_scheduler(self, optimizer: optim.Optimizer, config: AdvancedTrainingConfig) -> Any:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        scheduler_type = config.scheduler_type.lower()
        
        if scheduler_type == "cosine_warmup":
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=config.warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=config.max_steps - config.warmup_steps)
            return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps])
        
        elif scheduler_type == "cosine_restarts":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.max_steps // 4,
                T_mult=2,
                eta_min=1e-6
            )
        
        elif scheduler_type == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=config.lr * 10,
                total_steps=config.max_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        
        elif scheduler_type == "linear_warmup":
            return LinearLR(optimizer, start_factor=0.1, total_iters=config.warmup_steps)
        
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.max_steps // 3,
                gamma=0.1
            )
        
        else:
            return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


class AdvancedLossFunctions:
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                    model: nn.Module | None = None) -> torch.Tensor:
        if self.config.focal_loss_gamma > 0:
            return self._focal_loss(outputs, targets)
        elif self.config.contrastive_temperature > 0:
            return self._contrastive_loss(outputs, targets)
        else:
            return self.cross_entropy(outputs, targets)
    
    def _focal_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.config.focal_loss_gamma * ce_loss
        return focal_loss.mean()
    
    def _contrastive_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = F.normalize(outputs, dim=1)
        similarity_matrix = outputs @ outputs.T / self.config.contrastive_temperature
        
        targets_expanded = targets.unsqueeze(1)
        mask = (targets_expanded == targets.unsqueeze(0)).float()
        
        pos_sim = similarity_matrix * mask
        neg_sim = similarity_matrix * (1 - mask)
        
        pos_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim).sum(dim=1, keepdim=True)))
        return pos_loss.sum() / mask.sum()


class AdvancedRegularization:
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0
    
    def apply_mixup(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.config.use_mixup:
            return inputs, targets, torch.tensor(1.0)
        
        batch_size = inputs.size(0)
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
        
        index = torch.randperm(batch_size)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        
        return mixed_inputs, targets, targets[index], lam
    
    def apply_cutmix(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.config.use_cutmix:
            return inputs, targets, torch.tensor(1.0)
        
        batch_size = inputs.size(0)
        lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
        
        rand_index = torch.randperm(batch_size)
        targets_a, targets_b = targets, targets[rand_index]
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        
        return inputs, targets_a, targets_b, lam
    
    def _rand_bbox(self, size: torch.Size, lam: float) -> tuple[int, int, int, int]:
        W = size[2]
        H = size[3]
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()
        
        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(W, cx + cut_w // 2)
        bby2 = min(H, cy + cut_h // 2)
        
        return bbx1, bby1, bbx2, bby2


class AdvancedTrainingEngine:
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        self.optimizer_engine = AdvancedOptimizer()
        self.scheduler_engine = AdvancedScheduler()
        self.loss_engine = AdvancedLossFunctions(config)
        self.regularization = AdvancedRegularization(config)
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.tensorboard_writer = None

        self.device = None
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = (
            GradScaler(self.device.type) if (config.mixed_precision and TORCH_AVAILABLE) else None
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self._micro_step = 0
        self.nan_skips = 0
        
        if config.log_tensorboard and TORCH_AVAILABLE:
            try:
                self.tensorboard_writer = SummaryWriter()
            except Exception:
                self.tensorboard_writer = None
        else:
            self.tensorboard_writer = None
    
    def initialize_model(self, model: nn.Module) -> nn.Module:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model.to(self.device)
        
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        if self.config.use_distributed:
            self.model = DDP(self.model)
        
        return self.model
    
    def _enable_gradient_checkpointing(self) -> None:
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        for module in self.model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
    
    def initialize_optimizer(self) -> optim.Optimizer:
        if self.model is None:
            raise RuntimeError("Model must be initialized first")
        
        self.optimizer = self.optimizer_engine.get_optimizer(self.model, self.config)
        return self.optimizer
    
    def initialize_scheduler(self) -> Any:
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be initialized first")
        
        self.scheduler = self.scheduler_engine.get_scheduler(self.optimizer, self.config)
        return self.scheduler
    
    def train_step(self, batch: Any, targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        if not TORCH_AVAILABLE or self.model is None or self.optimizer is None:
            raise RuntimeError("Training not properly initialized")
        
        self.model.train()
        
        if self.config.use_mixup:
            batch, targets_a, targets_b, lam = self.regularization.apply_mixup(batch, targets)
        elif self.config.use_cutmix:
            batch, targets_a, targets_b, lam = self.regularization.apply_cutmix(batch, targets)
        else:
            targets_a, targets_b, lam = targets, None, 1.0
        
        batch = batch.to(self.device)
        targets = targets.to(self.device)
        
        accum = max(int(self.config.gradient_accumulation_steps), 1)
        grad_norm = 0.0

        if self.config.mixed_precision:
            with autocast(device_type=self.device.type):
                outputs = self.model(batch)
                loss = self.loss_engine.compute_loss(outputs, targets, self.model)

                if lam != 1.0:
                    loss_a = self.loss_engine.compute_loss(outputs, targets_a)
                    loss_b = self.loss_engine.compute_loss(outputs, targets_b)
                    loss = lam * loss_a + (1 - lam) * loss_b

            if not torch.isfinite(loss):
                self.nan_skips += 1
                self.optimizer.zero_grad(set_to_none=True)
                self._micro_step += 1
                return 0.0, 0.0
            self.scaler.scale(loss / accum).backward()
        else:
            outputs = self.model(batch)
            loss = self.loss_engine.compute_loss(outputs, targets, self.model)

            if lam != 1.0:
                loss_a = self.loss_engine.compute_loss(outputs, targets_a)
                loss_b = self.loss_engine.compute_loss(outputs, targets_b)
                loss = lam * loss_a + (1 - lam) * loss_b

            if not torch.isfinite(loss):
                self.nan_skips += 1
                self.optimizer.zero_grad(set_to_none=True)
                self._micro_step += 1
                return 0.0, 0.0
            (loss / accum).backward()

        self._micro_step += 1
        stepped = False
        if self._micro_step % accum == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm_t = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            grad_norm = float(grad_norm_t)
            if torch.isfinite(grad_norm_t):
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                stepped = True
            else:
                self.nan_skips += 1
                if self.scaler is not None:
                    self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        if stepped:
            self.global_step += 1

        return loss.item(), grad_norm
    
    def validate_step(self, batch: Any, targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("Model not initialized")
        
        self.model.eval()
        
        batch = batch.to(self.device)
        targets = targets.to(self.device)
        
        with torch.no_grad():
            if self.config.mixed_precision:
                with autocast(device_type=self.device.type):
                    outputs = self.model(batch)
                    loss = self.loss_engine.compute_loss(outputs, targets, self.model)
            else:
                outputs = self.model(batch)
                loss = self.loss_engine.compute_loss(outputs, targets, self.model)
        
        return loss.item(), outputs
    
    def train_epoch(self, train_loader: Any, val_loader: Any | None = None) -> TrainingMetrics:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        start_time = time.time()
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        correct = 0
        total_samples = 0
        
        for batch, targets in train_loader:
            loss, grad_norm = self.train_step(batch, targets)
            total_loss += loss
            total_grad_norm += grad_norm
            num_batches += 1
            
            with torch.no_grad():
                outputs = self.model(batch.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct += (predicted == targets.to(self.device)).sum().item()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
        train_accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        val_loss = 0.0
        val_accuracy = 0.0
        
        if val_loader is not None:
            val_loss, val_accuracy = self._validate(val_loader)
        
        training_time = time.time() - start_time
        throughput = num_batches / training_time if training_time > 0 else 0.0
        
        memory_usage = 0.0
        if self.device.type == "cuda":
            memory_usage = torch.cuda.max_memory_allocated() / (1024**3)
        
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        
        metrics = TrainingMetrics(
            train_loss=avg_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            learning_rate=current_lr,
            gradient_norm=avg_grad_norm,
            throughput=throughput,
            memory_usage=memory_usage
        )
        
        if self.tensorboard_writer:
            self._log_metrics(metrics)
        
        if self.config.use_early_stopping:
            self._check_early_stopping(val_loss)
        
        return metrics
    
    def _validate(self, val_loader: Any) -> tuple[float, float]:
        total_loss = 0.0
        correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch, targets in val_loader:
            loss, outputs = self.validate_step(batch, targets)
            total_loss += loss
            num_batches += 1
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct += (predicted == targets.to(self.device)).sum().item()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        if self.tensorboard_writer is None:
            return
        
        self.tensorboard_writer.add_scalar("Loss/train", metrics.train_loss, self.global_step)
        self.tensorboard_writer.add_scalar("Loss/val", metrics.val_loss, self.global_step)
        self.tensorboard_writer.add_scalar("Accuracy/train", metrics.train_accuracy, self.global_step)
        self.tensorboard_writer.add_scalar("Accuracy/val", metrics.val_accuracy, self.global_step)
        self.tensorboard_writer.add_scalar("Learning_Rate", metrics.learning_rate, self.global_step)
        self.tensorboard_writer.add_scalar("Gradient_Norm", metrics.gradient_norm, self.global_step)
        self.tensorboard_writer.add_scalar("Throughput", metrics.throughput, self.global_step)
        self.tensorboard_writer.add_scalar("Memory_Usage", metrics.memory_usage, self.global_step)
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss - self.config.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def should_stop(self) -> bool:
        if not self.config.use_early_stopping:
            return False
        return self.patience_counter >= self.config.patience
    
    def save_checkpoint(self, path: Path, metrics: TrainingMetrics) -> None:
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'metrics': metrics,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> TrainingMetrics:
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        
        return checkpoint['metrics']
    
    def close(self) -> None:
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.config.use_distributed:
            destroy_process_group()
