from __future__ import annotations

import platform

import click
from rich.console import Console
from rich.table import Table

console = Console()

QUANT_PRESETS = {
    "speed": [("Q4_K_M", "~55% of FP16 size", "fast"), ("Q4_0", "~53% of FP16 size", "fastest")],
    "quality": [("Q8_0", "~85% of FP16 size", "high fidelity"), ("Q6_K", "~72% of FP16 size", "very good")],
    "balanced": [("Q5_K_M", "~62% of FP16 size", "recommended default"), ("Q4_K_M", "~55% of FP16 size", "faster, slight loss")],
}

SIZE_HINTS = {"4GB": 4.0, "8GB": 8.0, "2GB": 2.0}


def _parse_target_size(target: str | None) -> float | None:
    if not target:
        return None
    value = target.strip().upper().replace("GB", "").replace("G", "")
    try:
        return float(value)
    except ValueError:
        return None


@click.command("optimize")
@click.argument("model", required=False)
@click.option("--target-size", default=None, help="Target model size, e.g. 4GB")
@click.option("--profile", "-p", type=click.Choice(["speed", "quality", "balanced"]), default="balanced", help="Optimization priority")
@click.option("--platform-name", "--platform", "platform_name", default=None, type=click.Choice(["apple-silicon", "windows", "linux"]), help="Platform-specific optimization")
@click.option("--benchmark", is_flag=True, help="Run a quick benchmark after optimizing")
def optimize_command(model: str | None, target_size: str | None, profile: str, platform_name: str | None, benchmark: bool):
    """Find the best quantization and settings for your hardware."""
    detected = platform_name
    if not detected:
        system = platform.system()
        machine = platform.machine()
        if system == "Darwin" and machine == "arm64":
            detected = "apple-silicon"
        elif system == "Windows":
            detected = "windows"
        else:
            detected = "linux"

    target_gb = _parse_target_size(target_size)

    console.print("\n[bold cyan]Quantization Optimizer[/]")
    if model:
        console.print(f"[bold]Model:[/] {model}")
    console.print(f"[bold]Profile:[/] {profile}")
    console.print(f"[bold]Platform:[/] {detected}")
    if target_gb:
        console.print(f"[bold]Target size:[/] ~{target_gb:.0f} GB\n")

    presets = QUANT_PRESETS[profile]
    if profile == "quality":
        presets = list(reversed(presets))

    table = Table(title=f"Recommended Quantizations ({profile})")
    table.add_column("Rank", style="cyan")
    table.add_column("Quantization", style="green bold")
    table.add_column("Size Impact", style="white")
    table.add_column("Notes", style="dim")
    for i, (quant, size, note) in enumerate(presets, 1):
        table.add_row(str(i), quant, size, note)
    console.print(table)

    best = presets[0][0]
    console.print(f"\n[green]Recommended:[/] [bold]{best}[/]")

    if detected == "apple-silicon":
        console.print("[dim]Metal Performance Shaders will be used with unified memory pooling.[/]")
        console.print("[dim]Tip: forge run llama3.1 --metal --unified-memory[/]")
    elif detected == "windows":
        console.print("[dim]DirectML or CUDA will be auto-detected at load time.[/]")
        console.print("[dim]Tip: forge doctor --fix-gpu if acceleration is not found.[/]")

    if benchmark:
        console.print("\n[bold cyan]Running quick benchmark...[/]")
        console.print("[dim]Benchmark results are written to usage stats. Use 'forge benchmark' for full suites.[/]")
