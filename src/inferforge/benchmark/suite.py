"""Comprehensive benchmarking suite for InferForge."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from inferforge.core.registry import ModelRecord
from inferforge.engine.base import ChatMessage, GenerationConfig
from inferforge.engine.unified_router import BackendType, get_unified_router


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    model: str
    backend: str
    duration: float
    tokens_per_second: float | None = None
    memory_used_mb: float | None = None
    gpu_memory_mb: float | None = None
    first_token_latency: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Configuration for a benchmark suite."""
    name: str
    prompt: str
    max_tokens: int = 100
    num_runs: int = 3
    warmup_runs: int = 1
    backends: list[BackendType] = field(default_factory=list)


class PerformanceBenchmark:
    """Performance benchmarking system for InferForge."""
    
    def __init__(self):
        self.router = get_unified_router()
        self.results: list[BenchmarkResult] = []
    
    def run_benchmark(
        self,
        model: ModelRecord,
        suite: BenchmarkSuite,
        backend: BackendType | None = None,
    ) -> list[BenchmarkResult]:
        """Run benchmark suite on a model.
        
        Args:
            model: Model to benchmark
            suite: Benchmark configuration
            backend: Optional backend override
        
        Returns:
            List of benchmark results
        """
        results = []
        
        # Determine backends to test
        backends_to_test = suite.backends if suite.backends else [backend] if backend else [None]
        
        for test_backend in backends_to_test:
            # Get engine
            try:
                engine = self.router.get_engine(model, test_backend)
            except Exception as e:
                results.append(BenchmarkResult(
                    name=suite.name,
                    model=model.name,
                    backend=test_backend.value if test_backend else "auto",
                    duration=0.0,
                    error=str(e),
                ))
                continue
            
            # Warmup runs
            for _ in range(suite.warmup_runs):
                try:
                    self._run_single_inference(engine, suite.prompt, suite.max_tokens)
                except Exception:
                    pass
            
            # Actual benchmark runs
            run_results = []
            for run_idx in range(suite.num_runs):
                try:
                    result = self._benchmark_inference(
                        engine,
                        model.name,
                        test_backend.value if test_backend else "auto",
                        suite.name,
                        suite.prompt,
                        suite.max_tokens,
                    )
                    run_results.append(result)
                except Exception as e:
                    run_results.append(BenchmarkResult(
                        name=suite.name,
                        model=model.name,
                        backend=test_backend.value if test_backend else "auto",
                        duration=0.0,
                        error=str(e),
                    ))
            
            # Average results
            if run_results and not run_results[0].error:
                avg_result = self._average_results(run_results)
                results.append(avg_result)
            elif run_results:
                results.append(run_results[0])  # Return first error
        
        self.results.extend(results)
        return results
    
    def _run_single_inference(
        self,
        engine: Any,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Run a single inference."""
        messages = [ChatMessage(role="user", content=prompt)]
        config = GenerationConfig(max_tokens=max_tokens)
        return engine.generate(messages, config)
    
    def _benchmark_inference(
        self,
        engine: Any,
        model_name: str,
        backend: str,
        bench_name: str,
        prompt: str,
        max_tokens: int,
    ) -> BenchmarkResult:
        """Benchmark a single inference run."""
        # Measure memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        gpu_mem_before = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        
        # Run inference with timing
        messages = [ChatMessage(role="user", content=prompt)]
        config = GenerationConfig(max_tokens=max_tokens)
        
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        
        # Use streaming to measure first token latency
        try:
            for idx, chunk in enumerate(engine.stream(messages, config)):
                if idx == 0:
                    first_token_time = time.perf_counter() - start_time
                token_count += len(chunk.split())
        except Exception:
            # Fallback to non-streaming
            _ = engine.generate(messages, config)
            token_count = max_tokens
        
        duration = time.perf_counter() - start_time
        
        # Measure memory after
        mem_after = process.memory_info().rss / (1024 * 1024)
        memory_used = mem_after - mem_before
        
        gpu_mem_after = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        
        gpu_memory_used = gpu_mem_after - gpu_mem_before if gpu_mem_after > 0 else None
        
        # Calculate tokens per second
        tokens_per_sec = token_count / duration if duration > 0 else None
        
        return BenchmarkResult(
            name=bench_name,
            model=model_name,
            backend=backend,
            duration=duration,
            tokens_per_second=tokens_per_sec,
            memory_used_mb=memory_used,
            gpu_memory_mb=gpu_memory_used,
            first_token_latency=first_token_time,
            metadata={
                "prompt_length": len(prompt),
                "max_tokens": max_tokens,
                "estimated_tokens": token_count,
            },
        )
    
    def _average_results(self, results: list[BenchmarkResult]) -> BenchmarkResult:
        """Average multiple benchmark results."""
        if not results:
            raise ValueError("No results to average")
        
        avg_duration = sum(r.duration for r in results) / len(results)
        
        # Average tokens per second
        tps_values = [r.tokens_per_second for r in results if r.tokens_per_second]
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else None
        
        # Average memory
        mem_values = [r.memory_used_mb for r in results if r.memory_used_mb]
        avg_mem = sum(mem_values) / len(mem_values) if mem_values else None
        
        # Average GPU memory
        gpu_mem_values = [r.gpu_memory_mb for r in results if r.gpu_memory_mb]
        avg_gpu_mem = sum(gpu_mem_values) / len(gpu_mem_values) if gpu_mem_values else None
        
        # Average first token latency
        ftl_values = [r.first_token_latency for r in results if r.first_token_latency]
        avg_ftl = sum(ftl_values) / len(ftl_values) if ftl_values else None
        
        return BenchmarkResult(
            name=results[0].name,
            model=results[0].model,
            backend=results[0].backend,
            duration=avg_duration,
            tokens_per_second=avg_tps,
            memory_used_mb=avg_mem,
            gpu_memory_mb=avg_gpu_mem,
            first_token_latency=avg_ftl,
            metadata={
                **results[0].metadata,
                "num_runs": len(results),
            },
        )
    
    def run_standard_suite(self, model: ModelRecord) -> list[BenchmarkResult]:
        """Run standard benchmark suite on a model."""
        suites = [
            BenchmarkSuite(
                name="short_completion",
                prompt="Write a hello world program in Python",
                max_tokens=50,
                num_runs=5,
            ),
            BenchmarkSuite(
                name="medium_completion",
                prompt="Explain how binary search works and provide a Python implementation",
                max_tokens=200,
                num_runs=3,
            ),
            BenchmarkSuite(
                name="long_completion",
                prompt="Write a comprehensive guide on building a REST API with FastAPI, including authentication, database integration, and testing",
                max_tokens=500,
                num_runs=3,
            ),
            BenchmarkSuite(
                name="code_generation",
                prompt="Create a Python class for a binary search tree with insert, search, and delete methods",
                max_tokens=300,
                num_runs=3,
            ),
        ]
        
        all_results = []
        for suite in suites:
            results = self.run_benchmark(model, suite)
            all_results.extend(results)
        
        return all_results
    
    def compare_models(
        self,
        models: list[ModelRecord],
        suite: BenchmarkSuite,
    ) -> dict[str, list[BenchmarkResult]]:
        """Compare multiple models on the same benchmark."""
        comparison = {}
        
        for model in models:
            results = self.run_benchmark(model, suite)
            comparison[model.name] = results
        
        return comparison
    
    def compare_backends(
        self,
        model: ModelRecord,
        suite: BenchmarkSuite,
        backends: list[BackendType],
    ) -> list[BenchmarkResult]:
        """Compare different backends for the same model."""
        suite.backends = backends
        return self.run_benchmark(model, suite)
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of all benchmarks."""
        if not self.results:
            return {"message": "No benchmark results yet"}
        
        # Group by model
        by_model: dict[str, list[BenchmarkResult]] = {}
        for result in self.results:
            if result.model not in by_model:
                by_model[result.model] = []
            by_model[result.model].append(result)
        
        # Calculate statistics
        summary = {}
        for model_name, results in by_model.items():
            successful = [r for r in results if not r.error]
            if not successful:
                continue
            
            avg_tps = sum(r.tokens_per_second for r in successful if r.tokens_per_second) / len(successful)
            avg_duration = sum(r.duration for r in successful) / len(successful)
            
            summary[model_name] = {
                "total_runs": len(results),
                "successful_runs": len(successful),
                "avg_tokens_per_second": avg_tps,
                "avg_duration": avg_duration,
                "backends_tested": list(set(r.backend for r in successful)),
            }
        
        return summary
    
    def save_results(self, output_path: Path) -> None:
        """Save benchmark results to JSON file."""
        import json
        
        data = {
            "timestamp": time.time(),
            "results": [
                {
                    "name": r.name,
                    "model": r.model,
                    "backend": r.backend,
                    "duration": r.duration,
                    "tokens_per_second": r.tokens_per_second,
                    "memory_used_mb": r.memory_used_mb,
                    "gpu_memory_mb": r.gpu_memory_mb,
                    "first_token_latency": r.first_token_latency,
                    "error": r.error,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
            "summary": self.get_summary(),
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(data, f, indent=2)
    
    def clear_results(self) -> None:
        """Clear all benchmark results."""
        self.results.clear()
        gc.collect()


class ContinuousMonitor:
    """Continuous performance monitoring system."""
    
    def __init__(self):
        self.metrics: list[dict[str, Any]] = []
        self._monitoring = False
    
    def start(self, interval: float = 1.0) -> None:
        """Start continuous monitoring.
        
        Args:
            interval: Sampling interval in seconds
        """
        import threading
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                metric = self._collect_metrics()
                self.metrics.append(metric)
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop continuous monitoring."""
        self._monitoring = False
        if hasattr(self, "_monitor_thread"):
            self._monitor_thread.join(timeout=2.0)
    
    def _collect_metrics(self) -> dict[str, Any]:
        """Collect system metrics."""
        process = psutil.Process()
        
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / (1024 * 1024),
            "memory_percent": process.memory_percent(),
        }
        
        # GPU metrics
        try:
            import torch
            if torch.cuda.is_available():
                metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
                metrics["gpu_utilization"] = torch.cuda.utilization()
        except ImportError:
            pass
        
        return metrics
    
    def get_metrics(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Get collected metrics.
        
        Args:
            last_n: Optional number of recent metrics to return
        
        Returns:
            List of metric dictionaries
        """
        if last_n:
            return self.metrics[-last_n:]
        return self.metrics
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()
