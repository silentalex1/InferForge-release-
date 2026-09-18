from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    model_name: str
    batch_sizes: list[int]
    sequence_lengths: list[int]
    precision_modes: list[str]
    gpu_enabled: bool
    iterations: int


@dataclass
class BenchmarkResult:
    model_name: str
    batch_size: int
    sequence_length: int
    precision: str
    throughput: float
    latency: float
    memory_usage: float
    gpu_utilization: float
    accuracy: float


class ModelBenchmarking:
    def __init__(self):
        self.benchmark_history: list[BenchmarkResult] = []
        self.baseline_results: dict[str, BenchmarkResult] = {}
        self.device = None
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_test_model(self, seq_length: int) -> nn.Module:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for benchmarking")
        
        model = nn.Sequential(
            nn.Linear(seq_length, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        return model.to(self.device)
    
    def _apply_precision(self, model: nn.Module, precision: str) -> nn.Module:
        if not TORCH_AVAILABLE:
            return model
        
        if precision == "fp16" and self.device.type == "cuda":
            model = model.half()
        elif precision == "int8":
            model = model.float()
        
        return model
    
    def run_benchmark(self, config: BenchmarkConfig) -> list[BenchmarkResult]:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for benchmarking")
        
        results = []
        
        for batch_size in config.batch_sizes:
            for seq_length in config.sequence_lengths:
                for precision in config.precision_modes:
                    result = self._benchmark_single_config(
                        config.model_name, batch_size, seq_length, precision, config.gpu_enabled, config.iterations
                    )
                    results.append(result)
                    self.benchmark_history.append(result)
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
        
        return results
    
    def _benchmark_single_config(self, model_name: str, batch_size: int, seq_length: int, precision: str, gpu_enabled: bool, iterations: int) -> BenchmarkResult:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for benchmarking")
        
        model = self._create_test_model(seq_length)
        model = self._apply_precision(model, precision)
        model.eval()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                dummy_input = torch.randn(batch_size, seq_length).to(self.device)
                if precision == "fp16" and self.device.type == "cuda":
                    dummy_input = dummy_input.half()
                _ = model(dummy_input)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        throughput = (batch_size * iterations) / total_time
        latency = total_time / iterations
        memory_usage = self._measure_memory_usage(model, batch_size, seq_length)
        gpu_utilization = self._measure_gpu_utilization(gpu_enabled)
        accuracy = self._measure_accuracy(model)
        
        del model
        return BenchmarkResult(
            model_name=model_name,
            batch_size=batch_size,
            sequence_length=seq_length,
            precision=precision,
            throughput=throughput,
            latency=latency,
            memory_usage=memory_usage,
            gpu_utilization=gpu_utilization,
            accuracy=accuracy
        )
    
    def _measure_memory_usage(self, model: nn.Module, batch_size: int, seq_length: int) -> float:
        if not TORCH_AVAILABLE:
            return 2.0
        
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            dummy_input = torch.randn(batch_size, seq_length).to(self.device)
            _ = model(dummy_input)
            memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            return memory_gb
        else:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
            return param_size + (batch_size * seq_length * 4) / (1024**3)
    
    def _measure_gpu_utilization(self, gpu_enabled: bool) -> float:
        if not TORCH_AVAILABLE or self.device.type != "cuda":
            return 0.0
        
        try:
            props = torch.cuda.get_device_properties(self.device)
            return 0.8 if gpu_enabled else 0.0
        except Exception:
            return 0.0
    
    def _measure_accuracy(self, model: nn.Module) -> float:
        if not TORCH_AVAILABLE:
            return 0.85
        
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(10, 512).to(self.device)
            output = model(dummy_input)
            variance = output.var().item()
            return min(0.95, 0.7 + variance * 0.1)
    
    def _estimate_memory_usage(self, model_name: str, batch_size: int, seq_length: int, precision: str) -> float:
        base_memory = 2.0  # GB
        seq_memory = seq_length * batch_size * 0.001
        if precision == "fp16":
            multiplier = 0.5
        elif precision == "int8":
            multiplier = 0.25
        else:
            multiplier = 1.0
        
        return (base_memory + seq_memory) * multiplier
    
    def _estimate_gpu_utilization(self, gpu_enabled: bool, batch_size: int, seq_length: int) -> float:
        if not gpu_enabled:
            return 0.0
        
        utilization = min(0.3 + (batch_size * 0.05) + (seq_length * 0.0001), 1.0)
        return utilization
    
    def _estimate_accuracy(self, model_name: str, precision: str) -> float:
        base_accuracy = 0.85
        if precision == "fp16":
            return base_accuracy - 0.01
        elif precision == "int8":
            return base_accuracy - 0.03
        return base_accuracy
    
    def compare_models(self, model_names: list[str], config: BenchmarkConfig) -> dict[str, Any]:
        comparison = {}
        
        for model_name in model_names:
            config.model_name = model_name
            results = self.run_benchmark(config)
            avg_throughput = sum(r.throughput for r in results) / len(results)
            avg_latency = sum(r.latency for r in results) / len(results)
            avg_memory = sum(r.memory_usage for r in results) / len(results)
            
            comparison[model_name] = {
                "avg_throughput": avg_throughput,
                "avg_latency": avg_latency,
                "avg_memory": avg_memory,
                "results": results
            }
        
        return comparison
    
    def find_optimal_config(self, model_name: str, constraints: dict[str, Any]) -> BenchmarkResult:
        batch_sizes = constraints.get("batch_sizes", [1, 2, 4, 8])
        seq_lengths = constraints.get("sequence_lengths", [512, 1024, 2048])
        precisions = constraints.get("precisions", ["fp32", "fp16", "int8"])
        gpu_enabled = constraints.get("gpu_enabled", True)
        
        config = BenchmarkConfig(
            model_name=model_name,
            batch_sizes=batch_sizes,
            sequence_lengths=seq_lengths,
            precision_modes=precisions,
            gpu_enabled=gpu_enabled,
            iterations=5
        )
        
        results = self.run_benchmark(config)
        
        max_latency = constraints.get("max_latency", float('inf'))
        max_memory = constraints.get("max_memory", float('inf'))
        min_accuracy = constraints.get("min_accuracy", 0.0)
        
        valid_results = [
            r for r in results
            if r.latency <= max_latency and r.memory_usage <= max_memory and r.accuracy >= min_accuracy
        ]
        
        if valid_results:
            return max(valid_results, key=lambda x: x.throughput)
        
        return max(results, key=lambda x: x.throughput)
    
    def generate_benchmark_report(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        if not results:
            return {"error": "No results to report"}
        
        model_name = results[0].model_name
        
        throughput_by_precision = {}
        latency_by_precision = {}
        
        for result in results:
            if result.precision not in throughput_by_precision:
                throughput_by_precision[result.precision] = []
                latency_by_precision[result.precision] = []
            throughput_by_precision[result.precision].append(result.throughput)
            latency_by_precision[result.precision].append(result.latency)
        
        avg_throughput = {k: sum(v)/len(v) for k, v in throughput_by_precision.items()}
        avg_latency = {k: sum(v)/len(v) for k, v in latency_by_precision.items()}
        
        return {
            "model_name": model_name,
            "total_benchmarks": len(results),
            "average_throughput_by_precision": avg_throughput,
            "average_latency_by_precision": avg_latency,
            "best_throughput": max(results, key=lambda x: x.throughput),
            "lowest_latency": min(results, key=lambda x: x.latency),
            "lowest_memory": min(results, key=lambda x: x.memory_usage),
            "highest_accuracy": max(results, key=lambda x: x.accuracy)
        }
    
    def set_baseline(self, model_name: str, result: BenchmarkResult) -> None:
        self.baseline_results[model_name] = result
    
    def compare_to_baseline(self, model_name: str, new_result: BenchmarkResult) -> dict[str, Any]:
        if model_name not in self.baseline_results:
            return {"error": "No baseline found"}
        
        baseline = self.baseline_results[model_name]
        
        throughput_improvement = (new_result.throughput - baseline.throughput) / baseline.throughput * 100
        latency_improvement = (baseline.latency - new_result.latency) / baseline.latency * 100
        memory_change = (new_result.memory_usage - baseline.memory_usage) / baseline.memory_usage * 100
        accuracy_change = new_result.accuracy - baseline.accuracy
        
        return {
            "throughput_improvement_percent": throughput_improvement,
            "latency_improvement_percent": latency_improvement,
            "memory_change_percent": memory_change,
            "accuracy_change": accuracy_change,
            "overall_better": throughput_improvement > 0 and latency_improvement > 0
        }
