"""API key management commands."""

from __future__ import annotations

import json
from pathlib import Path

import click
from platformdirs import user_config_dir
from rich.console import Console
from rich.table import Table

from inferforge.server.auth import get_api_key_manager

console = Console()


class ExternalApiKeyManager:
    def __init__(self):
        self.config_dir = Path(user_config_dir("inferforge"))
        self.keys_file = self.config_dir / "external_keys.json"
        self.keys: dict[str, dict] = {}
        self._load_keys()
    
    def _load_keys(self) -> None:
        if self.keys_file.exists():
            with open(self.keys_file, 'r') as f:
                self.keys = json.load(f)
    
    def _save_keys(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, indent=2)
    
    def add_key(self, provider: str, key: str, name: str = "") -> bool:
        provider_lower = provider.lower()
        if provider_lower not in self.keys:
            self.keys[provider_lower] = []
        
        self.keys[provider_lower].append({
            "key": key,
            "name": name,
            "added_at": None
        })
        
        self._save_keys()
        return True
    
    def list_keys(self, provider: str | None = None) -> dict:
        if provider:
            return {provider: self.keys.get(provider.lower(), [])}
        return self.keys
    
    def remove_key(self, provider: str, key: str) -> bool:
        provider_lower = provider.lower()
        if provider_lower not in self.keys:
            return False
        
        self.keys[provider_lower] = [k for k in self.keys[provider_lower] if k["key"] != key]
        self._save_keys()
        return True
    
    def rotate_key(self, provider: str, old_key: str, new_key: str) -> bool:
        if self.remove_key(provider, old_key):
            return self.add_key(provider, new_key, "Rotated key")
        return False


@click.group("api-key")
def apikey_command():
    """Manage API keys for InferForge server and external services."""
    pass


@apikey_command.command("create")
@click.option("--name", default="", help="Name/description for this key")
def create_key(name: str):
    """Create a new InferForge API key."""
    manager = get_api_key_manager()
    key = manager.create_key(name)
    
    console.print("\n[bold green]✓ API Key Created[/]\n")
    console.print(f"[yellow]Key:[/] [bold]{key}[/]\n")
    console.print("[dim]Save this key securely! You won't be able to see it again.[/]")
    console.print(f"[dim]Keys are stored in: {manager.keys_file}[/]\n")
    console.print("[cyan]Usage:[/]")
    console.print("  curl -H 'Authorization: Bearer <key>' https://hyperneural.cfd/api/v1/models")


@apikey_command.command("list")
@click.option("--provider", "-p", help="Filter by provider (openai, anthropic, etc.)")
def list_keys(provider: str | None):
    """List all API keys (truncated)."""
    manager = get_api_key_manager()
    inferforge_keys = manager.list_keys()
    
    external_manager = ExternalApiKeyManager()
    external_keys = external_manager.list_keys(provider)
    
    console.print("\n[bold cyan]InferForge API Keys[/]\n")
    
    if not inferforge_keys:
        console.print("[yellow]No InferForge API keys found[/]")
        console.print("[dim]Create one with: forge api-key create[/]")
    else:
        table = Table()
        table.add_column("Key (truncated)", style="cyan")
        table.add_column("Status", style="green")
        
        for key in inferforge_keys:
            table.add_row(key, "Active")
        
        console.print(table)
        console.print(f"\n[dim]Total InferForge keys: {len(inferforge_keys)}[/]")
    
    if external_keys:
        console.print("\n[bold cyan]External API Keys[/]\n")
        
        for provider, keys in external_keys.items():
            if keys:
                console.print(f"\n[bold]{provider.upper()}:[/]")
                for key_data in keys:
                    console.print(f"  • {key_data.get('name', 'Unnamed')}: {key_data['key'][:20]}...")


@apikey_command.command("add")
@click.argument("provider")
@click.argument("key")
@click.option("--name", default="", help="Name/description for this key")
def add_external_key(provider: str, key: str, name: str):
    """Add an external API key (OpenAI, Anthropic, etc.)."""
    manager = ExternalApiKeyManager()
    
    if manager.add_key(provider, key, name):
        console.print(f"[green]✓[/] Added {provider} API key")
    else:
        console.print(f"[red]Failed to add {provider} API key[/]")


@apikey_command.command("remove")
@click.argument("provider")
@click.argument("key")
def remove_external_key(provider: str, key: str):
    """Remove an external API key."""
    manager = ExternalApiKeyManager()
    
    if manager.remove_key(provider, key):
        console.print(f"[green]✓[/] Removed {provider} API key")
    else:
        console.print(f"[red]Failed to remove {provider} API key[/]")


@apikey_command.command("rotate")
@click.argument("provider")
@click.argument("old_key")
@click.argument("new_key")
def rotate_key(provider: str, old_key: str, new_key: str):
    """Rotate an external API key."""
    manager = ExternalApiKeyManager()
    
    if manager.rotate_key(provider, old_key, new_key):
        console.print(f"[green]✓[/] Rotated {provider} API key")
    else:
        console.print(f"[red]Failed to rotate {provider} API key[/]")


@apikey_command.command("revoke")
@click.argument("key")
def revoke_key(key: str):
    """Revoke an InferForge API key."""
    manager = get_api_key_manager()
    
    if click.confirm(f"Revoke key {key[:20]}...? This cannot be undone."):
        if manager.revoke_key(key):
            console.print("[green]✓[/] Key revoked successfully")
        else:
            console.print("[red]✗[/] Key not found")


@apikey_command.command("validate")
@click.argument("key")
def validate_key(key: str):
    """Validate an InferForge API key."""
    manager = get_api_key_manager()
    
    if manager.validate_key(key):
        console.print("[green]✓[/] Key is valid")
    else:
        console.print("[red]✗[/] Key is invalid or revoked")
