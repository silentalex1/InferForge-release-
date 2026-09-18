from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WorkerNode:
    id: str
    address: str
    gpu_count: int
    memory_gb: int
    status: str
    current_task: str | None


@dataclass
class TaskDistribution:
    task_id: str
    worker_id: str
    layers: list[int]
    data_shard: str
    status: str


class DistributedTraining:
    def __init__(self):
        self.workers: dict[str, WorkerNode] = {}
        self.task_queue: list[TaskDistribution] = []
        self.active_tasks: dict[str, TaskDistribution] = {}
        self.fault_tolerance_enabled = True
    
    def register_worker(self, worker_id: str, address: str, gpu_count: int, memory_gb: int) -> WorkerNode:
        worker = WorkerNode(
            id=worker_id,
            address=address,
            gpu_count=gpu_count,
            memory_gb=memory_gb,
            status="idle",
            current_task=None
        )
        self.workers[worker_id] = worker
        return worker
    
    def distribute_model(self, total_layers: int, strategy: str = "data_parallel") -> dict[str, Any]:
        available_workers = [w for w in self.workers.values() if w.status == "idle"]
        
        if not available_workers:
            return {"status": "no_workers_available"}
        
        if strategy == "data_parallel":
            return self._distribute_data_parallel(total_layers, available_workers)
        elif strategy == "tensor_parallel":
            return self._distribute_tensor_parallel(total_layers, available_workers)
        elif strategy == "pipeline_parallel":
            return self._distribute_pipeline_parallel(total_layers, available_workers)
        else:
            return self._auto_distribute(total_layers, available_workers)
    
    def _distribute_data_parallel(self, total_layers: int, workers: list[WorkerNode]) -> dict[str, Any]:
        distribution = {}
        
        for worker in workers:
            distribution[worker.id] = {
                "strategy": "data_parallel",
                "layers": list(range(total_layers)),
                "data_shard": f"shard_{worker.id}",
                "batch_size": 4 // len(workers)
            }
        
        return {"status": "distributed", "distribution": distribution}
    
    def _distribute_tensor_parallel(self, total_layers: int, workers: list[WorkerNode]) -> dict[str, Any]:
        layers_per_worker = total_layers // len(workers)
        distribution = {}
        
        for i, worker in enumerate(workers):
            start_layer = i * layers_per_worker
            end_layer = start_layer + layers_per_worker if i < len(workers) - 1 else total_layers
            
            distribution[worker.id] = {
                "strategy": "tensor_parallel",
                "layers": list(range(start_layer, end_layer)),
                "data_shard": "full",
                "batch_size": 4
            }
        
        return {"status": "distributed", "distribution": distribution}
    
    def _distribute_pipeline_parallel(self, total_layers: int, workers: list[WorkerNode]) -> dict[str, Any]:
        stages = len(workers)
        layers_per_stage = total_layers // stages
        distribution = {}
        
        for i, worker in enumerate(workers):
            start_layer = i * layers_per_stage
            end_layer = start_layer + layers_per_stage if i < stages - 1 else total_layers
            
            distribution[worker.id] = {
                "strategy": "pipeline_parallel",
                "layers": list(range(start_layer, end_layer)),
                "stage": i,
                "data_shard": "full",
                "batch_size": 4
            }
        
        return {"status": "distributed", "distribution": distribution}
    
    def _auto_distribute(self, total_layers: int, workers: list[WorkerNode]) -> dict[str, Any]:
        total_gpus = sum(w.gpu_count for w in workers)
        total_memory = sum(w.memory_gb for w in workers)
        
        if total_gpus >= 8:
            return self._distribute_tensor_parallel(total_layers, workers)
        elif total_memory >= 48:
            return self._distribute_data_parallel(total_layers, workers)
        else:
            return self._distribute_pipeline_parallel(total_layers, workers)
    
    def handle_worker_failure(self, worker_id: str) -> dict[str, Any]:
        if worker_id not in self.workers:
            return {"status": "worker_not_found"}
        
        failed_worker = self.workers[worker_id]
        failed_tasks = [t for t in self.active_tasks.values() if t.worker_id == worker_id]
        
        if not self.fault_tolerance_enabled:
            return {"status": "failure", "reason": "fault_tolerance_disabled"}
        
        available_workers = [w for w in self.workers.values() if w.status == "idle" and w.id != worker_id]
        
        if not available_workers:
            return {"status": "failure", "reason": "no_backup_workers"}
        
        redistribution = {}
        
        for task in failed_tasks:
            backup_worker = available_workers[0]
            new_task = TaskDistribution(
                task_id=task.task_id + "_recovered",
                worker_id=backup_worker.id,
                layers=task.layers,
                data_shard=task.data_shard,
                status="pending"
            )
            redistribution[task.task_id] = new_task
        
        self.workers[worker_id].status = "failed"
        
        return {
            "status": "recovered",
            "failed_worker": worker_id,
            "redistributed_tasks": len(redistribution),
            "redistribution": redistribution
        }
    
    def synchronize_checkpoints(self, checkpoint_dir: str) -> dict[str, Any]:
        sync_status = {}
        
        for worker_id, worker in self.workers.items():
            if worker.status == "training":
                sync_status[worker_id] = {
                    "status": "synced",
                    "last_checkpoint": "latest"
                }
            else:
                sync_status[worker_id] = {
                    "status": "idle",
                    "last_checkpoint": "none"
                }
        
        return {"status": "synchronized", "workers": sync_status}
    
    def balance_workload(self) -> dict[str, Any]:
        worker_loads = {}
        
        for worker_id, worker in self.workers.items():
            if worker.status == "training":
                worker_loads[worker_id] = len([t for t in self.active_tasks.values() if t.worker_id == worker_id])
            else:
                worker_loads[worker_id] = 0
        
        max_load = max(worker_loads.values()) if worker_loads else 0
        min_load = min(worker_loads.values()) if worker_loads else 0
        
        if max_load - min_load <= 1:
            return {"status": "balanced", "worker_loads": worker_loads}
        
        rebalanced = 0
        
        for worker_id, load in worker_loads.items():
            if load > max_load - 1:
                idle_workers = [w for w in self.workers.values() if w.status == "idle"]
                if idle_workers:
                    target_worker = idle_workers[0]
                    rebalanced += 1
        
        return {
            "status": "rebalanced",
            "worker_loads": worker_loads,
            "rebalanced_tasks": rebalanced
        }
    
    def get_cluster_status(self) -> dict[str, Any]:
        return {
            "total_workers": len(self.workers),
            "active_workers": len([w for w in self.workers.values() if w.status == "training"]),
            "idle_workers": len([w for w in self.workers.values() if w.status == "idle"]),
            "failed_workers": len([w for w in self.workers.values() if w.status == "failed"]),
            "total_gpus": sum(w.gpu_count for w in self.workers.values()),
            "total_memory": sum(w.memory_gb for w in self.workers.values()),
            "active_tasks": len(self.active_tasks),
            "workers": {w.id: {"status": w.status, "gpu_count": w.gpu_count} for w in self.workers.values()}
        }
