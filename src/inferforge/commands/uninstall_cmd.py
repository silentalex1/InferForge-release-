from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()

@click.command("uninstall")
def uninstall_command() -> None:
    console.print("[yellow]Uninstalling InferForge...[/]")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "inferforge"])
        console.print("[green]OK[/] pip package removed.")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]pip uninstall failed:[/] {e}")
    for p in [Path.home() / ".inferforge", Path.home() / ".inferforge-cache"]:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            console.print(f"[dim]removed {p}[/]")
    console.print("[green]Done.[/] Restart your shell.")
