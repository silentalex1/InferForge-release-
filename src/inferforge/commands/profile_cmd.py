from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from inferforge.core.profiles import get_profile_manager

console = Console()


@click.group("profile")
def profile_command():
    """Manage configuration profiles for different workflows."""
    pass


@profile_command.command("create")
@click.argument("name")
@click.option("--backend", default="ollama", help="Backend: native, ollama, remote")
@click.option("--host", default=None, help="Host URL for backend")
@click.option("--gpu-layers", type=int, default=None, help="Number of GPU layers")
@click.option("--threads", type=int, default=None, help="Number of CPU threads")
@click.option("--context-length", type=int, default=2048, help="Context window size")
def create_profile(name: str, backend: str, host: str | None, gpu_layers: int | None, threads: int | None, context_length: int):
    """Create a new configuration profile."""
    manager = get_profile_manager()
    
    config = {
        "backend": backend,
        "context_length": context_length,
    }
    
    if host:
        config["host"] = host
    if gpu_layers is not None:
        config["gpu_layers"] = gpu_layers
    if threads is not None:
        config["threads"] = threads
    
    manager.create_profile(name, config)


@profile_command.command("list")
def list_profiles():
    """List all available profiles."""
    manager = get_profile_manager()
    profiles = manager.list_profiles()
    active = manager.get_active()
    
    if not profiles:
        console.print("[yellow]No profiles found[/]")
        return
    
    table = Table(title="Configuration Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Backend", style="white")
    table.add_column("Active", style="green")
    
    for profile_name in profiles:
        profile_config = manager.get_profile(profile_name)
        is_active = "✓" if profile_name == active else ""
        table.add_row(
            profile_name,
            profile_config.get("backend", "unknown"),
            is_active
        )
    
    console.print(table)


@profile_command.command("use")
@click.argument("name")
def use_profile(name: str):
    """Set active profile."""
    manager = get_profile_manager()
    manager.set_active(name)


@profile_command.command("show")
@click.argument("name", required=False)
def show_profile(name: str | None):
    """Show profile configuration."""
    manager = get_profile_manager()
    
    if name is None:
        name = manager.get_active()
    
    config = manager.get_profile(name)
    if not config:
        console.print(f"[red]Profile '{name}' not found[/]")
        return
    
    console.print(f"\n[bold cyan]Profile:[/] {name}")
    console.print("[bold]Configuration:[/]")
    for key, value in config.items():
        console.print(f"  {key}: {value}")


@profile_command.command("delete")
@click.argument("name")
def delete_profile(name: str):
    """Delete a profile."""
    manager = get_profile_manager()
    
    if name == "default":
        console.print("[red]Cannot delete default profile[/]")
        return
    
    if click.confirm(f"Delete profile '{name}'?"):
        manager.delete_profile(name)
