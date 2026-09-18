"""CLI commands for registry management."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from inferforge.remote.registry_sync import RegistrySyncManager

console = Console()


@click.group("registry")
def registry_command():
    """Manage model registry and remote synchronization."""
    pass


@registry_command.command("push")
@click.argument("model")
@click.option("--tag", "-t", multiple=True, help="Version tags (can specify multiple)")
def push_command(model: str, tag: tuple[str, ...]):
    """Push a model to the remote registry."""
    console.print(f"[cyan]Pushing model:[/] {model}")
    
    try:
        sync_manager = RegistrySyncManager()
        tags = list(tag) if tag else ["latest"]
        
        result = sync_manager.push_model(model, tags=tags)
        
        console.print("[green]✓[/] Model pushed successfully")
        console.print(f"  Remote ID: {result.get('id')}")
        console.print(f"  Tags: {', '.join(tags)}")
        console.print(f"  URL: {result.get('url')}")
    
    except Exception as e:
        console.print(f"[red]✗ Failed to push model:[/] {e}")
        raise SystemExit(1)


@registry_command.command("pull")
@click.argument("model")
@click.option("--tag", "-t", default="latest", help="Version tag to pull")
@click.option("--force", "-f", is_flag=True, help="Force re-download")
def pull_command(model: str, tag: str, force: bool):
    """Pull a model from the remote registry."""
    console.print(f"[cyan]Pulling model:[/] {model}:{tag}")
    
    try:
        sync_manager = RegistrySyncManager()
        result = sync_manager.pull_model(model, tag=tag, force=force)
        
        console.print("[green]✓[/] Model pulled successfully")
        console.print(f"  Name: {result.name}")
        console.print(f"  Size: {result.display_size()}")
        console.print(f"  Format: {result.format}")
    
    except Exception as e:
        console.print(f"[red]✗ Failed to pull model:[/] {e}")
        raise SystemExit(1)


@registry_command.command("list-remote")
@click.option("--search", "-s", help="Search query")
def list_remote_command(search: str | None):
    """List models available in the remote registry."""
    try:
        sync_manager = RegistrySyncManager()
        models = sync_manager.list_remote_models(search=search)
        
        if not models:
            console.print("[yellow]No models found in remote registry[/]")
            return
        
        table = Table(title="Remote Models")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="blue")
        table.add_column("Size", style="green")
        table.add_column("Downloads", style="yellow")
        
        for model in models:
            table.add_row(
                model["name"],
                model.get("version", "latest"),
                _format_size(model.get("size", 0)),
                str(model.get("downloads", 0)),
            )
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]✗ Failed to list remote models:[/] {e}")
        raise SystemExit(1)


@registry_command.command("sync")
@click.option(
    "--direction",
    "-d",
    type=click.Choice(["push", "pull", "both"]),
    default="both",
    help="Sync direction",
)
@click.option("--force", "-f", is_flag=True, help="Force sync")
def sync_command(direction: str, force: bool):
    """Synchronize local and remote registries."""
    console.print(f"[cyan]Synchronizing registry ({direction})...[/]")
    
    try:
        sync_manager = RegistrySyncManager()
        result = sync_manager.sync(direction=direction, force=force)
        
        if result.get("skipped"):
            console.print(f"[yellow]Sync skipped:[/] {result['reason']}")
            return
        
        console.print("[green]✓ Sync complete[/]")
        console.print(f"  Pushed: {result['pushed']}")
        console.print(f"  Pulled: {result['pulled']}")
        console.print(f"  Conflicts: {result['conflicts']}")
        
        if result["errors"]:
            console.print("\n[yellow]Errors:[/]")
            for error in result["errors"]:
                console.print(f"  - {error}")
    
    except Exception as e:
        console.print(f"[red]✗ Sync failed:[/] {e}")
        raise SystemExit(1)


@registry_command.command("versions")
@click.argument("model")
def versions_command(model: str):
    """List all versions of a model."""
    try:
        sync_manager = RegistrySyncManager()
        versions = sync_manager.get_model_versions(model)
        
        if not versions:
            console.print(f"[yellow]No versions found for {model}[/]")
            return
        
        table = Table(title=f"Versions of {model}")
        table.add_column("Version", style="cyan")
        table.add_column("Created", style="blue")
        table.add_column("Size", style="green")
        table.add_column("Checksum", style="dim")
        
        for version in versions:
            from datetime import datetime
            created = datetime.fromtimestamp(version.created_at).strftime("%Y-%m-%d %H:%M")
            
            table.add_row(
                version.version,
                created,
                _format_size(version.size),
                version.checksum[:12] + "...",
            )
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]✗ Failed to get versions:[/] {e}")
        raise SystemExit(1)


@registry_command.command("status")
def status_command():
    """Show registry sync status."""
    try:
        sync_manager = RegistrySyncManager()
        status = sync_manager.get_sync_status()
        
        table = Table(title="Registry Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Endpoint", status["endpoint"] or "[dim]Not configured[/]")
        
        if status["last_sync"] > 0:
            from datetime import datetime
            last_sync = datetime.fromtimestamp(status["last_sync"]).strftime("%Y-%m-%d %H:%M:%S")
            table.add_row("Last Sync", last_sync)
            table.add_row("Time Since Sync", f"{status['time_since_sync']:.0f}s")
        else:
            table.add_row("Last Sync", "[dim]Never[/]")
        
        table.add_row("Sync Interval", f"{status['sync_interval']}s")
        table.add_row("Synced Models", str(status["synced_models_count"]))
        table.add_row("Conflicts", str(status["conflicts_count"]))
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]✗ Failed to get status:[/] {e}")
        raise SystemExit(1)


@registry_command.command("delete-remote")
@click.argument("model")
@click.option("--tag", "-t", help="Specific version tag to delete")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def delete_remote_command(model: str, tag: str | None, yes: bool):
    """Delete a model from the remote registry."""
    if not yes:
        if tag:
            message = f"Delete version '{tag}' of model '{model}' from remote registry?"
        else:
            message = f"Delete ALL versions of model '{model}' from remote registry?"
        
        if not click.confirm(message):
            console.print("[yellow]Cancelled[/]")
            return
    
    try:
        sync_manager = RegistrySyncManager()
        sync_manager.delete_remote_model(model, tag=tag)
        
        if tag:
            console.print(f"[green]✓[/] Deleted {model}:{tag} from remote registry")
        else:
            console.print(f"[green]✓[/] Deleted {model} from remote registry")
    
    except Exception as e:
        console.print(f"[red]✗ Failed to delete model:[/] {e}")
        raise SystemExit(1)


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    if size_bytes <= 0:
        return "—"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    
    return f"{size_bytes} B"
