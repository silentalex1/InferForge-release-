from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from inferforge.core.config import trained_models_dir
from inferforge.core.registry import Registry

console = Console()


@click.group("model")
def model_group():
    """Model versioning and management commands."""
    pass


@model_group.command("history")
@click.argument("model_name")
def model_history(model_name: str):
    """Show version history for a model."""
    reg = Registry()
    record = reg.get(model_name)
    
    if not record:
        console.print(f"[red]Model '{model_name}' not found[/]")
        return
    
    model_path = Path(record.path) if record.path else trained_models_dir() / model_name.replace(":", "-")
    versions_dir = model_path / "versions"
    
    if not versions_dir.exists():
        console.print(f"[yellow]No version history found for '{model_name}'[/]")
        return
    
    versions = sorted(versions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    
    table = Table(title=f"Version History: {model_name}")
    table.add_column("Version", style="cyan")
    table.add_column("Date", style="white")
    table.add_column("Size", style="white")
    table.add_column("Notes", style="gray")
    
    for version_path in versions:
        metadata_file = version_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            size = sum(f.stat().st_size for f in version_path.rglob("*") if f.is_file()) / (1024**2)
            size_str = f"{size:.1f} MB"
            
            table.add_row(
                metadata.get("version", "unknown"),
                metadata.get("date", "unknown"),
                size_str,
                metadata.get("notes", "")
            )
    
    console.print(table)


@model_group.command("rollback")
@click.argument("model_name")
@click.argument("version")
@click.option("--force", is_flag=True, help="Force rollback without confirmation")
def model_rollback(model_name: str, version: str, force: bool):
    """Rollback a model to a specific version."""
    reg = Registry()
    record = reg.get(model_name)
    
    if not record:
        console.print(f"[red]Model '{model_name}' not found[/]")
        return
    
    model_path = Path(record.path) if record.path else trained_models_dir() / model_name.replace(":", "-")
    versions_dir = model_path / "versions"
    target_version = versions_dir / version
    
    if not target_version.exists():
        console.print(f"[red]Version '{version}' not found[/]")
        return
    
    if not force:
        if not click.confirm(f"Rollback '{model_name}' to version '{version}'?"):
            return
    
    backup_current = model_path / "current_backup"
    if backup_current.exists():
        shutil.rmtree(backup_current)
    
    current_files = [f for f in model_path.iterdir() if f.is_file() or (f.is_dir() and f.name != "versions")]
    for item in current_files:
        if item.is_file():
            shutil.copy2(item, backup_current / item.name)
        else:
            shutil.copytree(item, backup_current / item.name, dirs_exist_ok=True)
    
    version_files = [f for f in target_version.iterdir() if f.is_file() or (f.is_dir() and f.name != "metadata.json")]
    for item in version_files:
        target = model_path / item.name
        if target.exists():
            if target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        
        if item.is_file():
            shutil.copy2(item, target)
        else:
            shutil.copytree(item, target, dirs_exist_ok=True)
    
    console.print(f"[green]✓[/] Rolled back '{model_name}' to version '{version}'")
    console.print(f"[dim]Backup saved to: {backup_current}[/]")


@model_group.command("diff")
@click.argument("model_name")
@click.argument("version1")
@click.argument("version2")
def model_diff(model_name: str, version1: str, version2: str):
    """Compare two versions of a model."""
    reg = Registry()
    record = reg.get(model_name)
    
    if not record:
        console.print(f"[red]Model '{model_name}' not found[/]")
        return
    
    model_path = Path(record.path) if record.path else trained_models_dir() / model_name.replace(":", "-")
    versions_dir = model_path / "versions"
    
    v1_path = versions_dir / version1
    v2_path = versions_dir / version2
    
    if not v1_path.exists() or not v2_path.exists():
        console.print("[red]One or both versions not found[/]")
        return
    
    v1_meta = v1_path / "metadata.json"
    v2_meta = v2_path / "metadata.json"
    
    if v1_meta.exists() and v2_meta.exists():
        with open(v1_meta) as f:
            v1_data = json.load(f)
        with open(v2_meta) as f:
            v2_data = json.load(f)
        
        console.print(f"\n[bold cyan]Comparing:[/] {version1} vs {version2}")
        console.print(f"[bold]Version 1 ({version1}):[/]")
        console.print(f"  Date: {v1_data.get('date', 'unknown')}")
        console.print(f"  Notes: {v1_data.get('notes', 'none')}")
        console.print(f"  Parameters: {v1_data.get('parameters', {})}")
        
        console.print(f"\n[bold]Version 2 ({version2}):[/]")
        console.print(f"  Date: {v2_data.get('date', 'unknown')}")
        console.print(f"  Notes: {v2_data.get('notes', 'none')}")
        console.print(f"  Parameters: {v2_data.get('parameters', {})}")
        
        console.print("\n[bold]Parameter Changes:[/]")
        v1_params = v1_data.get('parameters', {})
        v2_params = v2_data.get('parameters', {})
        
        all_keys = set(v1_params.keys()) | set(v2_params.keys())
        for key in sorted(all_keys):
            v1_val = v1_params.get(key, "N/A")
            v2_val = v2_params.get(key, "N/A")
            
            if v1_val != v2_val:
                console.print(f"  {key}: {v1_val} → {v2_val}")


@model_group.command("tag")
@click.argument("model_name")
@click.argument("version")
@click.option("--notes", default="", help="Notes for this version")
def model_tag(model_name: str, version: str, notes: str):
    """Tag a model version with metadata."""
    reg = Registry()
    record = reg.get(model_name)
    
    if not record:
        console.print(f"[red]Model '{model_name}' not found[/]")
        return
    
    model_path = Path(record.path) if record.path else trained_models_dir() / model_name.replace(":", "-")
    versions_dir = model_path / "versions"
    version_path = versions_dir / version
    
    if not version_path.exists():
        console.print(f"[red]Version '{version}' not found[/]")
        return
    
    metadata_file = version_path / "metadata.json"
    metadata = {}
    
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
    
    metadata["version"] = version
    metadata["date"] = datetime.now().isoformat()
    metadata["notes"] = notes
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    console.print(f"[green]✓[/] Tagged version '{version}' with metadata")


model_command = model_group