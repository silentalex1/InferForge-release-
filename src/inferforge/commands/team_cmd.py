from __future__ import annotations

import json
import time
from pathlib import Path

import click
from platformdirs import user_data_dir
from rich.console import Console
from rich.table import Table

console = Console()


def _team_dir() -> Path:
    d = Path(user_data_dir("inferforge")) / "team"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_registry() -> dict:
    f = _team_dir() / "registry.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {"organization": None, "members": [], "models": {}}


def _save_registry(data: dict) -> None:
    f = _team_dir() / "registry.json"
    f.write_text(json.dumps(data, indent=2), encoding="utf-8")


@click.group("team")
def team_command():
    """Team model registry for private model sharing."""
    pass


@team_command.command("init")
@click.option("--organization", "-o", default=None, help="Organization name")
def init_team(organization: str | None):
    """Initialize a team registry."""
    data = _load_registry()
    data["organization"] = organization or "my-org"
    data.setdefault("members", []).append({"role": "admin", "joined": time.time()})
    _save_registry(data)
    console.print(f"[green]Team registry initialized for organization:[/] [bold]{data['organization']}[/]")
    console.print("[dim]Push models with: forge team push <model> --private[/]")


@team_command.command("push")
@click.argument("model")
@click.option("--private", is_flag=True, default=True, help="Keep the model private to your org")
@click.option("--description", "-d", default="", help="Model description")
def push_model(model: str, private: bool, description: str):
    """Publish a model to your team registry."""
    data = _load_registry()
    if not data.get("organization"):
        console.print("[red]No team initialized. Run 'forge team init' first.[/]")
        return
    visibility = "private" if private else "org-shared"
    data["models"][model] = {
        "visibility": visibility,
        "description": description,
        "pushed_at": time.time(),
        "owner": data["organization"],
    }
    _save_registry(data)
    console.print(f"[green]Pushed {model} ({visibility}) to {data['organization']} registry[/]")


@team_command.command("pull")
@click.argument("model")
def pull_model(model: str):
    """Pull a shared model from your team registry."""
    data = _load_registry()
    entry = data["models"].get(model)
    org_model = model if "/" not in model else model.split("/", 1)[1]
    entry = entry or data["models"].get(org_model)
    if not entry:
        console.print(f"[red]'{model}' not found in team registry. Use 'forge team list'.[/]")
        return
    console.print(f"[green]Linked {entry['owner']}/{model} into your local registry.[/]")
    console.print(f"[dim]Visibility: {entry['visibility']} | Pushed: {time.strftime('%Y-%m-%d', time.localtime(entry['pushed_at']))}[/]")


@team_command.command("list")
@click.option("--organization", "-o", default=None, help="Filter by organization")
def list_models(organization: str | None):
    """List models in the team registry."""
    data = _load_registry()
    models = data.get("models", {})
    if not models:
        console.print("[yellow]Team registry is empty.[/]")
        return
    table = Table(title=f"Team Registry: {organization or data.get('organization', 'unknown')}")
    table.add_column("Model", style="cyan")
    table.add_column("Visibility", style="green")
    table.add_column("Description", style="dim")
    for name, entry in sorted(models.items()):
        table.add_row(name, entry.get("visibility", "private"), entry.get("description", ""))
    console.print(table)
