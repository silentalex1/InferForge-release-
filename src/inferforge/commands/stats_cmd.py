from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from inferforge.core.analytics import get_analytics_manager

console = Console()


@click.group("stats")
def stats_command():
    """Usage analytics and performance metrics."""
    pass


@stats_command.command()
@click.option("--detailed", is_flag=True, help="Show detailed statistics")
def show(detailed: bool):
    """Show usage statistics."""
    manager = get_analytics_manager()
    
    model_stats = manager.get_model_stats()
    command_stats = manager.get_command_stats()
    error_stats = manager.get_error_stats()
    daily_stats = manager.get_daily_stats(days=7)
    
    console.print("\n[bold cyan]Usage Statistics[/]\n")
    
    if model_stats:
        table = Table(title="Model Usage")
        table.add_column("Model", style="cyan")
        table.add_column("Requests", style="yellow")
        table.add_column("Tokens", style="green")
        table.add_column("Total Time", style="white")
        table.add_column("Avg Time", style="magenta")
        
        for model, stats in sorted(model_stats.items(), key=lambda x: x[1]["count"], reverse=True):
            avg_time = stats["time"] / stats["count"] if stats["count"] > 0 else 0
            table.add_row(
                model,
                str(stats["count"]),
                str(stats["tokens"]),
                f"{stats['time']:.2f}s",
                f"{avg_time:.2f}s"
            )
        
        console.print(table)
    
    if command_stats:
        console.print("\n[bold]Command Usage[/]\n")
        for command, stats in sorted(command_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:10]:
            console.print(f"  {command}: {stats['count']} uses")
    
    if error_stats and detailed:
        console.print("\n[bold]Error Statistics[/]\n")
        for error, stats in sorted(error_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:5]:
            console.print(f"  {error}: {stats['count']} occurrences")
    
    if daily_stats and detailed:
        console.print("\n[bold]Daily Statistics (Last 7 Days)[/]\n")
        table = Table()
        table.add_column("Date", style="cyan")
        table.add_column("Requests", style="yellow")
        table.add_column("Tokens", style="green")
        table.add_column("Errors", style="red")
        
        for date, stats in sorted(daily_stats.items(), reverse=True):
            table.add_row(
                date,
                str(stats["requests"]),
                str(stats["tokens"]),
                str(stats["errors"])
            )
        
        console.print(table)


@stats_command.command("export")
@click.option("--format", default="json", help="Export format: json, csv")
@click.option("--output", "-o", help="Output file path")
@click.option("--period", type=int, default=30, help="Period in days")
def stats_export(format: str, output: str | None, period: int):
    """Export analytics data."""
    manager = get_analytics_manager()
    
    if not output:
        output = f"inferforge_analytics_{period}d.{format}"
    
    export_path = Path(output)
    
    if manager.export_analytics(export_path, format):
        console.print(f"[green]✓[/] Analytics exported to {export_path}")
    else:
        console.print("[red]Failed to export analytics[/]")


@stats_command.command("performance")
@click.argument("model", required=False)
def stats_performance(model: str | None):
    """Show performance metrics for models."""
    manager = get_analytics_manager()
    
    perf_stats = manager.get_performance_stats(model)
    
    if not perf_stats:
        console.print("[yellow]No performance data available[/]")
        return
    
    console.print("\n[bold cyan]Performance Metrics[/]\n")
    
    if model:
        console.print(f"[bold]Model:[/] {model}\n")
        for metric, values in perf_stats.items():
            if values:
                latest = values[-1]
                avg = sum(v["value"] for v in values) / len(values)
                console.print(f"  {metric}:")
                console.print(f"    Latest: {latest['value']:.2f}")
                console.print(f"    Average: {avg:.2f}")
                console.print(f"    Samples: {len(values)}")
    else:
        for model_name, metrics in perf_stats.items():
            console.print(f"\n[bold]{model_name}[/]")
            for metric, values in metrics.items():
                if values:
                    latest = values[-1]
                    console.print(f"  {metric}: {latest['value']:.2f}")


@stats_command.command("clear")
@click.option("--older-than", type=int, help="Clear data older than N days")
@click.option("--force", is_flag=True, help="Force clear without confirmation")
def stats_clear(older_than: int | None, force: bool):
    """Clear analytics data."""
    manager = get_analytics_manager()
    
    if not force:
        if older_than:
            if not click.confirm(f"Clear analytics data older than {older_than} days?"):
                return
        else:
            if not click.confirm("Clear all analytics data?"):
                return
    
    manager.clear_analytics(older_than)
    
    if older_than:
        console.print(f"[green]✓[/] Cleared analytics data older than {older_than} days")
    else:
        console.print("[green]✓[/] Cleared all analytics data")