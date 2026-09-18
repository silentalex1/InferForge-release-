from __future__ import annotations

import time
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from inferforge.core.registry import Registry
from inferforge.engine import ChatMessage, get_router

console = Console()


@click.command("compare")
@click.argument("models", nargs=-1, required=True)
@click.option("--prompt", "-p", default="Write a Python function for fibonacci sequence", help="Test prompt")
@click.option("--iterations", "-n", default=3, type=int, help="Number of test iterations")
@click.option("--metrics", "-m", multiple=True, help="Metrics to measure: speed, quality, tokens, memory")
@click.option("--export", "-e", help="Export results to file")
def compare_command(models: tuple[str, ...], prompt: str, iterations: int, metrics: tuple[str, ...], export: str | None):
    """Compare multiple models side-by-side."""
    
    if len(models) < 2:
        console.print("[red]Error:[/] Need at least 2 models to compare")
        return
    
    if not metrics:
        metrics = ("speed", "quality", "tokens")
    
    console.print("\n[bold cyan]Model Comparison[/]\n")
    console.print(f"Prompt: [dim]{prompt}[/]")
    console.print(f"Iterations: {iterations}")
    console.print(f"Metrics: {', '.join(metrics)}\n")
    
    reg = Registry()
    router = get_router()
    results = {}
    
    for model_name in models:
        record = reg.get(model_name)
        if not record:
            console.print(f"[yellow]Model not found: {model_name}")
            continue
        
        console.print(f"Testing [cyan]{model_name}[/]...")
        
        timings = []
        responses = []
        token_counts = []
        memory_usage = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Running {iterations} iterations...", total=iterations)
            
            for i in range(iterations):
                engine = router.resolve(record)
                try:
                    import psutil
                    process = psutil.Process()
                    mem_before = process.memory_info().rss
                    
                    start = time.perf_counter()
                    response = engine.chat([ChatMessage(role="user", content=prompt)])
                    end = time.perf_counter()
                    
                    mem_after = process.memory_info().rss
                    memory_usage.append((mem_after - mem_before) / (1024**2))
                    
                    timings.append(end - start)
                    token_counts.append(len(response.split()))
                    
                    if i == 0:
                        responses.append(response)
                finally:
                    engine.close()
                    progress.update(task, advance=1)
        
        avg_time = sum(timings) / len(timings) if timings else 0
        min_time = min(timings) if timings else 0
        max_time = max(timings) if timings else 0
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        
        results[model_name] = {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "avg_tokens": avg_tokens,
            "avg_memory": avg_memory,
            "response": responses[0] if responses else "",
            "response_length": len(responses[0]) if responses else 0
        }
        
        console.print(f"[green]Completed[/] {model_name}: {avg_time:.2f}s avg\n")
    
    table = Table(title="Performance Comparison")
    table.add_column("Model", style="cyan")
    
    if "speed" in metrics:
        table.add_column("Avg Time", style="yellow")
        table.add_column("Min Time", style="green")
        table.add_column("Max Time", style="red")
    if "tokens" in metrics:
        table.add_column("Avg Tokens", style="blue")
    if "memory" in metrics:
        table.add_column("Avg Memory (MB)", style="magenta")
    if "quality" in metrics:
        table.add_column("Response Length", style="white")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["avg_time"])
    
    for model_name, data in sorted_results:
        row = [model_name]
        
        if "speed" in metrics:
            row.extend([
                f"{data['avg_time']:.2f}s",
                f"{data['min_time']:.2f}s",
                f"{data['max_time']:.2f}s"
            ])
        if "tokens" in metrics:
            row.append(f"{data['avg_tokens']:.0f}")
        if "memory" in metrics:
            row.append(f"{data['avg_memory']:.1f}")
        if "quality" in metrics:
            row.append(str(data['response_length']))
        
        table.add_row(*row)
    
    console.print(table)
    
    if "speed" in metrics:
        fastest = sorted_results[0]
        console.print(f"\n[bold green]Fastest:[/] {fastest[0]} ({fastest[1]['avg_time']:.2f}s)")
    
    if export:
        import json
        export_data = {
            "prompt": prompt,
            "iterations": iterations,
            "metrics": list(metrics),
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "response"} for k, v in results.items()}
        }
        
        export_path = Path(export)
        export_path.write_text(json.dumps(export_data, indent=2))
        console.print(f"\n[green]Results exported to:[/] {export_path}")
    
    if "quality" in metrics:
        console.print("\n[bold]Sample Responses:[/]\n")
        for model_name, data in sorted_results[:3]:
            console.print(f"[cyan]{model_name}:[/]")
            console.print(f"[dim]{data['response'][:300]}...[/]\n")
