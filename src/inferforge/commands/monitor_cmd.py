from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()


class TrainingMonitor:
    def __init__(self):
        self.monitoring = False
        self.current_metrics: dict[str, Any] = {}
        self.metrics_history: list[dict] = []
    
    def start_monitoring(self, model_name: str) -> None:
        self.monitoring = True
        self.current_metrics = {
            "model": model_name,
            "epoch": 0,
            "loss": 0.0,
            "learning_rate": 0.0001,
            "gpu_utilization": 0.0,
            "memory_usage": 0.0,
            "samples_per_second": 0.0,
            "start_time": time.time()
        }
    
    def update_metrics(self, metrics: dict[str, Any]) -> None:
        self.current_metrics.update(metrics)
        self.metrics_history.append({
            **self.current_metrics,
            "timestamp": datetime.now().isoformat()
        })
    
    def stop_monitoring(self) -> None:
        self.monitoring = False
    
    def get_current_metrics(self) -> dict[str, Any]:
        return self.current_metrics
    
    def get_metrics_history(self) -> list[dict]:
        return self.metrics_history
    
    def export_metrics(self, output_path: Path) -> bool:
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    "model": self.current_metrics.get("model"),
                    "history": self.metrics_history
                }, f, indent=2)
            return True
        except Exception:
            return False


class DashboardRenderer:
    def __init__(self, monitor: TrainingMonitor):
        self.monitor = monitor
    
    def create_dashboard(self) -> Layout:
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        return layout
    
    def render_header(self) -> Panel:
        metrics = self.monitor.get_current_metrics()
        elapsed = time.time() - metrics.get("start_time", time.time())
        
        header_text = f"""
Training Monitor: {metrics.get('model', 'Unknown')}
Epoch: {metrics.get('epoch', 0)} | Elapsed: {elapsed:.0f}s | Learning Rate: {metrics.get('learning_rate', 0)}
"""
        
        return Panel(header_text, style="bold cyan")
    
    def render_main(self) -> Panel:
        metrics = self.monitor.get_current_metrics()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="green")
        
        table.add_row("Loss", f"{metrics.get('loss', 0):.4f}", self._get_loss_status(metrics.get('loss', 0)))
        table.add_row("GPU Utilization", f"{metrics.get('gpu_utilization', 0):.1f}%", self._get_gpu_status(metrics.get('gpu_utilization', 0)))
        table.add_row("Memory Usage", f"{metrics.get('memory_usage', 0):.1f} GB", self._get_memory_status(metrics.get('memory_usage', 0)))
        table.add_row("Samples/Second", f"{metrics.get('samples_per_second', 0):.1f}", "Active")
        
        return Panel(table, title="Training Metrics")
    
    def render_footer(self) -> Panel:
        history = self.monitor.get_metrics_history()
        
        if len(history) > 1:
            recent_losses = [m.get('loss', 0) for m in history[-10:]]
            trend = "decreasing" if recent_losses[-1] < recent_losses[0] else "increasing"
            
            footer_text = f"Recent Loss Trend: {trend} | Total Samples: {len(history)}"
        else:
            footer_text = "Monitoring in progress..."
        
        return Panel(footer_text, style="dim")
    
    def _get_loss_status(self, loss: float) -> str:
        if loss < 0.5:
            return "Excellent"
        elif loss < 1.0:
            return "Good"
        elif loss < 2.0:
            return "Fair"
        else:
            return "Poor"
    
    def _get_gpu_status(self, utilization: float) -> str:
        if utilization > 80:
            return "High"
        elif utilization > 50:
            return "Medium"
        else:
            return "Low"
    
    def _get_memory_status(self, memory: float) -> str:
        if memory > 8:
            return "High"
        elif memory > 4:
            return "Medium"
        else:
            return "Low"


@click.group("monitor")
def monitor_group():
    """Training monitor dashboard."""
    pass


@monitor_group.command("start")
@click.argument("model")
@click.option("--update-interval", type=int, default=1, help="Update interval in seconds")
def monitor_start(model: str, update_interval: int):
    """Start real-time training monitoring dashboard."""
    monitor = TrainingMonitor()
    renderer = DashboardRenderer(monitor)
    
    monitor.start_monitoring(model)
    
    console.print(f"[cyan]Starting training monitor for {model}...[/]")
    console.print("[dim]Press Ctrl+C to stop monitoring[/]\n")
    
    try:
        with Live(renderer.create_dashboard(), refresh_per_second=update_interval) as live:
            while monitor.monitoring:
                live.update(renderer.render_header(), "header")
                live.update(renderer.render_main(), "main")
                live.update(renderer.render_footer(), "footer")
                
                time.sleep(update_interval)
                
                if not monitor.monitoring:
                    break
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user[/]")
    
    monitor.stop_monitoring()


@monitor_group.command("metrics")
@click.argument("model")
def monitor_metrics(model: str):
    """Show current training metrics."""
    monitor = TrainingMonitor()
    monitor.start_monitoring(model)
    
    metrics = monitor.get_current_metrics()
    
    table = Table(title=f"Training Metrics: {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    for key, value in metrics.items():
        if key != "start_time":
            table.add_row(key, str(value))
    
    console.print(table)


@monitor_group.command("export")
@click.argument("model")
@click.option("--output", "-o", help="Output file path")
def monitor_export(model: str, output: str | None):
    """Export training metrics to file."""
    monitor = TrainingMonitor()
    monitor.start_monitoring(model)
    
    if not output:
        output = f"{model.replace(':', '-')}_metrics.json"
    
    output_path = Path(output)
    
    if monitor.export_metrics(output_path):
        console.print(f"[green]✓[/] Metrics exported to {output_path}")
    else:
        console.print("[red]Failed to export metrics[/]")


monitor_command = monitor_group