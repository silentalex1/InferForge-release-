from __future__ import annotations

import json
import platform
from pathlib import Path

import click
from rich.console import Console

from inferforge.core.config import data_dir

console = Console()

def _log_path() -> Path:
    return data_dir() / "logs" / "inferforge.log"

@click.group("logs")
def logs_group():
    pass

@logs_group.command("show")
@click.option("--lines", default=50, type=int, help="Lines to show")
def logs_show(lines: int):
    p = _log_path()
    if not p.exists():
        console.print(f"[yellow]No log file yet at {p}[/]")
        return
    text = p.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in text[-lines:]:
        console.print(line)

@logs_group.command("export")
@click.option("--output", default="inferforge-diagnostics.json", help="Output file")
def logs_export(output: str):
    out = Path(output)
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    p = _log_path()
    logs = p.read_text(encoding="utf-8", errors="replace")[-8000:] if p.exists() else ""
    out.write_text(json.dumps({"system": info, "logs": logs}, indent=2), encoding="utf-8")
    console.print(f"[green]✓[/] Diagnostics exported to {out}")

logs_command = logs_group
