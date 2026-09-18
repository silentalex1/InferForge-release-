"""CLI commands for benchmarking."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from inferforge.benchmark.suite import BenchmarkSuite, PerformanceBenchmark
from inferforge.core.registry import Registry
from inferforge.engine.unified_router import BackendType

console = Console()


@click.group("benchmark")
def benchmark_command():
    """Performance benchmarking commands."""
    pass


@benchmark_command.command("run")
@click.argument("model")
@click.option("--prompt", "-p", default="Write a hello world program", help="Benchmark prompt")
@click.option("--max-tokens", "-t", default=100, type=int, help="Maximum tokens to generate")
@click.option("--runs", "-n", default=3, type=int, help="Number of benchmark runs")
@click.option("--backend", "-b", type=click.Choice(["native", "ollama", "huggingface", "remote"]), help="Specific backend to test")
@click.option("--save", "-s", type=click.Path(), help="Save results to file")
def run_command(
    model: str,
    prompt: str,
    max_tokens: int,
    runs: int,
    backend: str | None,
    save: str | None,
):
    """Run benchmark on a model."""
    console.print(f"[cyan]Benchmarking model:[/] {model}")
    
    # Get model
    registry = Registry()
    model_record = registry.get(model)
    
    if not model_record:
        console.print(f"[red]Model not found:[/] {model}")
        raise SystemExit(1)
    
    # Create benchmark
    benchmark = PerformanceBenchmark()
    suite = BenchmarkSuite(
        name="custom_benchmark",
        prompt=prompt,
        max_tokens=max_tokens,
        num_runs=runs,
    )
    
    # Parse backend
    backend_type = None
    if backend:
        backend_type = BackendType(backend)
    
    # Run benchmark
    console.print("[dim]Running benchmark...[/]")
    results = benchmark.run_benchmark(model_record, suite, backend_type)
    
    # Display results
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    for result in results:
        if result.error:
            console.print(f"[red]✗ Benchmark failed:[/] {result.error}")
            continue
        
        table.add_row("Backend", result.backend)
        table.add_row("Duration", f"{result.duration:.3f}s")
        
        if result.tokens_per_second:
            table.add_row("Tokens/Second", f"{result.tokens_per_second:.2f}")
        
        if result.first_token_latency:
            table.add_row("First Token Latency", f"{result.first_token_latency:.3f}s")
        
        if result.memory_used_mb:
            table.add_row("Memory Used", f"{result.memory_used_mb:.2f} MB")
        
        if result.gpu_memory_mb:
            table.add_row("GPU Memory", f"{result.gpu_memory_mb:.2f} MB")
    
    console.print(table)
    
    # Save results
    if save:
        from pathlib import Path
        benchmark.save_results(Path(save))
        console.print(f"[green]✓[/] Results saved to: {save}")


@benchmark_command.command("suite")
@click.argument("model")
@click.option("--save", "-s", type=click.Path(), help="Save results to file")
def suite_command(model: str, save: str | None):
    """Run standard benchmark suite on a model."""
    console.print(f"[cyan]Running standard benchmark suite on:[/] {model}")
    
    # Get model
    registry = Registry()
    model_record = registry.get(model)
    
    if not model_record:
        console.print(f"[red]Model not found:[/] {model}")
        raise SystemExit(1)
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    console.print("[dim]This may take several minutes...[/]")
    
    results = benchmark.run_standard_suite(model_record)
    
    # Display results
    table = Table(title=f"Standard Benchmark Suite - {model}")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Duration", style="white")
    table.add_column("Tokens/s", style="green")
    table.add_column("First Token", style="blue")
    table.add_column("Memory", style="yellow")
    
    for result in results:
        if result.error:
            table.add_row(
                result.name,
                "[red]Error[/]",
                "[dim]—[/]",
                "[dim]—[/]",
                "[dim]—[/]",
            )
        else:
            table.add_row(
                result.name,
                f"{result.duration:.2f}s",
                f"{result.tokens_per_second:.1f}" if result.tokens_per_second else "—",
                f"{result.first_token_latency:.3f}s" if result.first_token_latency else "—",
                f"{result.memory_used_mb:.1f}MB" if result.memory_used_mb else "—",
            )
    
    console.print(table)
    
    # Summary
    summary = benchmark.get_summary()
    if model in summary:
        stats = summary[model]
        console.print("\n[bold]Summary:[/]")
        console.print(f"  Successful runs: {stats['successful_runs']}/{stats['total_runs']}")
        console.print(f"  Avg tokens/second: {stats['avg_tokens_per_second']:.2f}")
        console.print(f"  Avg duration: {stats['avg_duration']:.2f}s")
    
    # Save results
    if save:
        from pathlib import Path
        benchmark.save_results(Path(save))
        console.print(f"[green]✓[/] Results saved to: {save}")


@benchmark_command.command("compare")
@click.argument("models", nargs=-1, required=True)
@click.option("--prompt", "-p", default="Write a Python function", help="Benchmark prompt")
@click.option("--max-tokens", "-t", default=100, type=int, help="Maximum tokens")
@click.option("--save", "-s", type=click.Path(), help="Save results to file")
def compare_command(models: tuple[str, ...], prompt: str, max_tokens: int, save: str | None):
    """Compare multiple models."""
    console.print(f"[cyan]Comparing {len(models)} models...[/]")
    
    # Get models
    registry = Registry()
    model_records = []
    
    for model_name in models:
        model = registry.get(model_name)
        if not model:
            console.print(f"[yellow]Warning: Model not found: {model_name}[/]")
            continue
        model_records.append(model)
    
    if not model_records:
        console.print("[red]No valid models found[/]")
        raise SystemExit(1)
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    suite = BenchmarkSuite(
        name="comparison",
        prompt=prompt,
        max_tokens=max_tokens,
        num_runs=3,
    )
    
    comparison = benchmark.compare_models(model_records, suite)
    
    # Display results
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Duration", style="white")
    table.add_column("Tokens/s", style="green")
    table.add_column("Memory", style="yellow")
    
    for model_name, results in comparison.items():
        if not results or results[0].error:
            table.add_row(model_name, "[red]Error[/]", "—", "—")
        else:
            result = results[0]
            table.add_row(
                model_name,
                f"{result.duration:.2f}s",
                f"{result.tokens_per_second:.1f}" if result.tokens_per_second else "—",
                f"{result.memory_used_mb:.1f}MB" if result.memory_used_mb else "—",
            )
    
    console.print(table)
    
    # Save results
    if save:
        from pathlib import Path
        benchmark.save_results(Path(save))
        console.print(f"[green]✓[/] Results saved to: {save}")


@benchmark_command.command("backends")
@click.argument("model")
@click.option("--prompt", "-p", default="Write hello world", help="Benchmark prompt")
@click.option("--save", "-s", type=click.Path(), help="Save results to file")
def backends_command(model: str, prompt: str, save: str | None):
    """Compare different backends for a model."""
    console.print(f"[cyan]Comparing backends for:[/] {model}")
    
    # Get model
    registry = Registry()
    model_record = registry.get(model)
    
    if not model_record:
        console.print(f"[red]Model not found:[/] {model}")
        raise SystemExit(1)
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    suite = BenchmarkSuite(
        name="backend_comparison",
        prompt=prompt,
        max_tokens=100,
        num_runs=3,
    )
    
    backends = [BackendType.NATIVE, BackendType.OLLAMA, BackendType.HUGGINGFACE]
    results = benchmark.compare_backends(model_record, suite, backends)
    
    # Display results
    table = Table(title="Backend Comparison")
    table.add_column("Backend", style="cyan")
    table.add_column("Duration", style="white")
    table.add_column("Tokens/s", style="green")
    table.add_column("Memory", style="yellow")
    
    for result in results:
        if result.error:
            table.add_row(result.backend, "[red]Error[/]", "—", "—")
        else:
            table.add_row(
                result.backend,
                f"{result.duration:.2f}s",
                f"{result.tokens_per_second:.1f}" if result.tokens_per_second else "—",
                f"{result.memory_used_mb:.1f}MB" if result.memory_used_mb else "—",
            )
    
    console.print(table)
    
    # Save results
    if save:
        from pathlib import Path
        benchmark.save_results(Path(save))
        console.print(f"[green]✓[/] Results saved to: {save}")
