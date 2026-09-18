"""Advanced distributed training with FSDP, DeepSpeed, and ZeRO optimizations."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import always_wrap_policy
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DistributedConfig:
    world_size: int = 1
    backend: str = "nccl"
    use_fsdp: bool = False
    use_ddp: bool = False
    sharding_strategy: str = "full_shard"
    cpu_offload: bool = False
    mixed_precision: bool = True
    bucket_cap_mb: int = 25


class DistributedTrainer:
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.rank = 0
        self.world_size = config.world_size
        self.local_rank = 0
        self.initialized = False
    
    def setup(self, rank: int, world_size: int) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
        
        dist.init_process_group(
            backend=self.config.backend,
            rank=rank,
            world_size=world_size
        )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        
        self.initialized = True
    
    def cleanup(self) -> None:
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        if not self.initialized:
            return model
        
        if self.config.use_fsdp:
            return self._wrap_fsdp(model)
        elif self.config.use_ddp:
            return self._wrap_ddp(model)
        
        return model
    
    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        model = model.to(self.local_rank)
        return DDP(model, device_ids=[self.local_rank])
    
    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        sharding_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        
        sharding_strategy = sharding_map.get(
            self.config.sharding_strategy,
            ShardingStrategy.FULL_SHARD
        )
        
        auto_wrap_policy = always_wrap_policy
        
        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=self.config.cpu_offload,
            mixed_precision=self.config.mixed_precision
        )
        
        return model
    
    def is_main_process(self) -> bool:
        return self.rank == 0
    
    def barrier(self) -> None:
        if self.initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        if not self.initialized:
            return tensor
        
        op_map = {"sum": dist.ReduceOp.SUM, "mean": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}
        dist_op = op_map.get(op, dist.ReduceOp.SUM)
        
        dist.all_reduce(tensor, op=dist_op)
        
        if op == "mean":
            tensor.div_(self.world_size)
        
        return tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if not self.initialized:
            return tensor
        
        dist.broadcast(tensor, src=src)
        return tensor
    
    def gather(self, tensor: torch.Tensor, dst: int = 0) -> list[torch.Tensor] | None:
        if not self.initialized:
            return [tensor] if self.rank == dst else None
        
        if self.rank == dst:
            gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.gather(tensor, gather_list=gather_list, dst=dst)
            return gather_list
        else:
            dist.gather(tensor, gather_list=None, dst=dst)
            return None


class PipelineParallel:
    def __init__(self, model: nn.Module, num_stages: int = 4):
        self.model = model
        self.num_stages = num_stages
        self.stages = self._split_model()
    
    def _split_model(self) -> list[nn.Module]:
        modules = list(self.model.children())
        stage_size = len(modules) // self.num_stages
        stages = []
        
        for i in range(self.num_stages):
            start_idx = i * stage_size
            end_idx = start_idx + stage_size if i < self.num_stages - 1 else len(modules)
            stage_modules = modules[start_idx:end_idx]
            stages.append(nn.Sequential(*stage_modules))
        
        return stages
    
    def forward(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        if stage_idx == 0:
            output = self.stages[0](x)
            for stage in self.stages[1:]:
                output = stage(output)
            return output
        else:
            return self.stages[stage_idx](x)


class TensorParallel:
    def __init__(self, model: nn.Module, world_size: int):
        self.model = model
        self.world_size = world_size
        self._apply_tensor_parallelism()
    
    def _apply_tensor_parallelism(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._parallelize_linear(module)
            elif isinstance(module, nn.Embedding):
                self._parallelize_embedding(module)
    
    def _parallelize_linear(self, module: nn.Linear) -> None:
        if self.world_size > 1:
            with torch.no_grad():
                module.weight.data = module.weight.data.chunk(self.world_size, dim=0)[0]
                if module.bias is not None:
                    module.bias.data = module.bias.data.chunk(self.world_size, dim=0)[0]
    
    def _parallelize_embedding(self, module: nn.Embedding) -> None:
        if self.world_size > 1:
            with torch.no_grad():
                module.weight.data = module.weight.data.chunk(self.world_size, dim=0)[0]


def launch_distributed(
    train_fn: callable,
    world_size: int,
    config: DistributedConfig
) -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    mp.spawn(
        train_fn,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
