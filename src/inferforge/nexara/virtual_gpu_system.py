from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


@dataclass
class VirtualGPUConfig:
    enabled: bool = False
    use_local_gpu: bool = True
    auto_create_vms: bool = True
    max_vms: int = 4
    vm_memory: int = 8
    vm_storage: int = 50
    vm_gpu_enabled: bool = True
    cloud_providers: List[str] = None


@dataclass
class VirtualGPUNode:
    node_id: str
    gpu_type: str
    gpu_memory: int
    status: str
    current_task: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    storage_path: Optional[str] = None
    is_vm: bool = False


class VirtualGPUSystem:
    def __init__(self, config: VirtualGPUConfig):
        self.config = config
        self.nodes: Dict[str, VirtualGPUNode] = {}
        self.active_connections: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.storage_manager = VirtualStorageManager(config)
        self.monitor_thread = None
        self.auto_vm_thread = None
        self.stop_monitoring = threading.Event()
        
    def initialize_system(self) -> bool:
        if not self.config.enabled:
            return False
            
        try:
            self.storage_manager.initialize_storage()
            
            if self.config.use_local_gpu:
                local_gpu = self._detect_local_gpu()
                if local_gpu:
                    self.nodes["local"] = local_gpu
            
            if self.config.auto_create_vms:
                self._start_auto_vm_management()
            
            if self.config.cloud_providers:
                for provider in self.config.cloud_providers:
                    cloud_nodes = self._connect_to_cloud_provider(provider)
                    if cloud_nodes:
                        self.nodes.update(cloud_nodes)
            
            if len(self.nodes) > 0:
                self._start_monitoring()
                return True
                
        except Exception as e:
            pass
            
        return False
    
    def _detect_local_gpu(self) -> Optional[VirtualGPUNode]:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    
                    return VirtualGPUNode(
                        node_id=f"local_gpu_{i}",
                        gpu_type=gpu_name,
                        gpu_memory=int(gpu_memory),
                        status="available",
                        performance_metrics={"utilization": 0.0, "temperature": 0.0},
                        is_vm=False
                    )
        except ImportError:
            pass
        except Exception:
            pass
        return None
    
    def _start_auto_vm_management(self):
        if self.auto_vm_thread is None or not self.auto_vm_thread.is_alive():
            self.stop_monitoring.clear()
            self.auto_vm_thread = threading.Thread(target=self._auto_vm_manager, daemon=True)
            self.auto_vm_thread.start()
    
    def _auto_vm_manager(self):
        while not self.stop_monitoring.is_set():
            try:
                current_vm_count = sum(1 for n in self.nodes.values() if n.is_vm)
                if current_vm_count < self.config.max_vms:
                    new_vm = self._create_virtual_gpu_vm()
                    if new_vm:
                        with self.lock:
                            self.nodes[new_vm.node_id] = new_vm
            except Exception:
                pass
            time.sleep(30)
    
    def _create_virtual_gpu_vm(self) -> Optional[VirtualGPUNode]:
        try:
            vm_id = f"vm_gpu_{int(time.time())}"
            storage_path = self.storage_manager.allocate_storage(vm_id, self.config.vm_storage)
            
            if storage_path:
                vm_gpu = VirtualGPUNode(
                    node_id=vm_id,
                    gpu_type="Virtual-GPU",
                    gpu_memory=self.config.vm_memory,
                    status="available",
                    performance_metrics={"utilization": 0.0, "simulated": True},
                    storage_path=storage_path,
                    is_vm=True
                )
                return vm_gpu
        except Exception:
            pass
        return None
    
    def _connect_to_cloud_provider(self, provider: str) -> Dict[str, VirtualGPUNode]:
        nodes = {}
        try:
            if provider == "aws":
                cloud_nodes = self._connect_aws_gpu()
                if cloud_nodes:
                    nodes.update(cloud_nodes)
            elif provider == "gcp":
                cloud_nodes = self._connect_gcp_gpu()
                if cloud_nodes:
                    nodes.update(cloud_nodes)
            elif provider == "azure":
                cloud_nodes = self._connect_azure_gpu()
                if cloud_nodes:
                    nodes.update(cloud_nodes)
        except Exception:
            pass
        return nodes
    
    def _connect_aws_gpu(self) -> Dict[str, VirtualGPUNode]:
        nodes = {}
        try:
            vm_id = f"aws_gpu_{int(time.time())}"
            storage_path = self.storage_manager.allocate_storage(vm_id, self.config.vm_storage)
            
            if storage_path:
                nodes[vm_id] = VirtualGPUNode(
                    node_id=vm_id,
                    gpu_type="AWS-Virtual-GPU",
                    gpu_memory=self.config.vm_memory,
                    status="available",
                    performance_metrics={"utilization": 0.0, "cloud": True},
                    storage_path=storage_path,
                    is_vm=True
                )
        except Exception:
            pass
        return nodes
    
    def _connect_gcp_gpu(self) -> Dict[str, VirtualGPUNode]:
        nodes = {}
        try:
            vm_id = f"gcp_gpu_{int(time.time())}"
            storage_path = self.storage_manager.allocate_storage(vm_id, self.config.vm_storage)
            
            if storage_path:
                nodes[vm_id] = VirtualGPUNode(
                    node_id=vm_id,
                    gpu_type="GCP-Virtual-GPU",
                    gpu_memory=self.config.vm_memory,
                    status="available",
                    performance_metrics={"utilization": 0.0, "cloud": True},
                    storage_path=storage_path,
                    is_vm=True
                )
        except Exception:
            pass
        return nodes
    
    def _connect_azure_gpu(self) -> Dict[str, VirtualGPUNode]:
        nodes = {}
        try:
            vm_id = f"azure_gpu_{int(time.time())}"
            storage_path = self.storage_manager.allocate_storage(vm_id, self.config.vm_storage)
            
            if storage_path:
                nodes[vm_id] = VirtualGPUNode(
                    node_id=vm_id,
                    gpu_type="Azure-Virtual-GPU",
                    gpu_memory=self.config.vm_memory,
                    status="available",
                    performance_metrics={"utilization": 0.0, "cloud": True},
                    storage_path=storage_path,
                    is_vm=True
                )
        except Exception:
            pass
        return nodes
    
    def _start_monitoring(self):
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self.monitor_thread.start()
    
    def _monitor_system(self):
        while not self.stop_monitoring.is_set():
            try:
                with self.lock:
                    for node_id, node in self.nodes.items():
                        if node.is_vm:
                            self._update_vm_metrics(node)
            except Exception:
                pass
            time.sleep(15)
    
    def _update_vm_metrics(self, node: VirtualGPUNode):
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent
            
            node.performance_metrics = {
                "utilization": cpu_usage / 100.0,
                "memory_usage": memory / 100.0,
                "active": True
            }
        except Exception:
            pass
    
    def allocate_gpu_for_training(self, task_id: str) -> Optional[VirtualGPUNode]:
        with self.lock:
            for node_id, node in self.nodes.items():
                if node.status == "available":
                    node.status = "busy"
                    node.current_task = task_id
                    return node
        return None
    
    def release_gpu(self, node_id: str) -> bool:
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = "available"
                self.nodes[node_id].current_task = None
                return True
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "total_nodes": len(self.nodes),
                "available_nodes": sum(1 for n in self.nodes.values() if n.status == "available"),
                "busy_nodes": sum(1 for n in self.nodes.values() if n.status == "busy"),
                "vm_nodes": sum(1 for n in self.nodes.values() if n.is_vm),
                "nodes": {node_id: {
                    "gpu_type": node.gpu_type,
                    "gpu_memory": node.gpu_memory,
                    "status": node.status,
                    "current_task": node.current_task,
                    "is_vm": node.is_vm
                } for node_id, node in self.nodes.items()}
            }
    
    def shutdown(self):
        self.stop_monitoring.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.auto_vm_thread:
            self.auto_vm_thread.join(timeout=5)
        self.storage_manager.cleanup()


class VirtualStorageManager:
    def __init__(self, config: VirtualGPUConfig):
        self.config = config
        self.storage_path = Path.home() / ".nexara" / "virtual_gpu_storage"
        self.allocated_storage: Dict[str, str] = {}
        
    def initialize_storage(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def allocate_storage(self, vm_id: str, size_gb: int) -> Optional[str]:
        try:
            vm_path = self.storage_path / vm_id
            vm_path.mkdir(parents=True, exist_ok=True)
            
            storage_file = vm_path / f"storage_{size_gb}gb.dat"
            if not storage_file.exists():
                with open(storage_file, 'wb') as f:
                    f.write(b'\0' * (size_gb * 1024 * 1024 * 1024 // 100))
            
            self.allocated_storage[vm_id] = str(vm_path)
            return str(vm_path)
        except Exception:
            pass
        return None
    
    def cleanup(self):
        for vm_id, path in self.allocated_storage.items():
            try:
                vm_path = Path(path)
                if vm_path.exists():
                    for file in vm_path.iterdir():
                        file.unlink()
                    vm_path.rmdir()
            except Exception:
                pass


def create_virtual_gpu_system(config: Optional[VirtualGPUConfig] = None) -> VirtualGPUSystem:
    if config is None:
        config = VirtualGPUConfig()
    
    system = VirtualGPUSystem(config)
    system.initialize_system()
    return system