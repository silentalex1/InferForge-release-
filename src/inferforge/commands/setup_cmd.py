from __future__ import annotations

import platform
import shutil

import click
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.command("setup")
@click.option("--verify", is_flag=True, help="Verify installation after setup")
def setup_command(verify: bool):
    console.print(Panel("InferForge Setup Wizard", style="cyan"))
    console.print(f"[dim]Python {platform.python_version()} on {platform.system()}[/]\n")
    checks = [
        ("Python 3.10+", platform.python_version_tuple()[0] >= "3" and int(platform.python_version_tuple()[1]) >= 10 if platform.python_version_tuple()[0]=="3" else True),
        ("pip", shutil.which("pip") is not None),
        ("ollama", shutil.which("ollama") is not None),
        ("git", shutil.which("git") is not None),
    ]
    for name, ok in checks:
        console.print(f"  {'[green]✓[/]' if ok else '[yellow]○[/]'} {name}")
    if not any(ok for _, ok in checks[1:2]):
        console.print("\n[yellow]pip not found — install Python 3.10+ from python.org[/]")
    console.print("\n[bold]Next steps:[/]")
    console.print("  [cyan]forge pull qwen2.5-coder:1.5b[/]  — small test model")
    console.print("  [cyan]forge list[/]                    — verify registry")
    console.print("  [cyan]forge doctor[/]                  — diagnose hardware")
    if verify:
        console.print("\n[dim]Running verification...[/]")
        try:
            from inferforge.core.config import load_settings
            load_settings()
            console.print("[green]✓ Configuration OK[/]")
        except Exception as exc:
            console.print(f"[red]Config error:[/] {exc}")
        console.print("[green]✓ Setup complete[/]")
