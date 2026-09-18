from __future__ import annotations

import json
import time
from pathlib import Path

import click
from platformdirs import user_data_dir
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

BUILTIN_RECIPES = [
    {
        "slug": "coding-assistant",
        "name": "Coding Assistant",
        "description": "Qwen2.5-Coder with code review templates and agent tools.",
        "steps": [
            "forge pull qwen2.5-coder:7b",
            "forge template add code-review \"Review this code:\\n{code}\\nFocus on: {aspects}\"",
            "forge chat",
        ],
    },
    {
        "slug": "private-writer",
        "name": "Private Writing Studio",
        "description": "Llama 3.1 tuned for long-form writing, fully offline.",
        "steps": [
            "forge pull llama3.1:8b",
            "forge profile create writer --backend native --context-length 8192",
            "forge run llama3.1:8b --system 'You are a precise writing assistant.'",
        ],
    },
    {
        "slug": "web-embed",
        "name": "Browser AI Widget",
        "description": "Ship a WebGPU chat widget to any static site.",
        "steps": [
            "forge web init my-widget",
            "forge web add qwen2.5-coder:7b --progressive",
            "forge web serve",
        ],
    },
]


def _recipes_dir() -> Path:
    d = Path(user_data_dir("inferforge")) / "recipes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _all_recipes() -> dict[str, dict]:
    recipes = {r["slug"]: r for r in BUILTIN_RECIPES}
    for f in _recipes_dir().glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            recipes[data.get("slug", f.stem)] = data
        except json.JSONDecodeError:
            continue
    return recipes


@click.group("recipe")
def recipe_command():
    """Community recipes for common InferForge setups."""
    pass


@recipe_command.command("search")
@click.argument("query")
def search_recipes(query: str):
    """Search available recipes."""
    q = query.lower()
    matches = [r for r in _all_recipes().values() if q in r["slug"].lower() or q in r["name"].lower() or q in r["description"].lower()]
    if not matches:
        console.print(f"[yellow]No recipes match '{query}'.[/]")
        return
    table = Table(title=f"Recipes matching '{query}'")
    table.add_column("Slug", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description", style="dim")
    for r in matches:
        table.add_row(r["slug"], r["name"], r["description"])
    console.print(table)
    console.print("[dim]Install with: forge recipe install <slug>[/]")


@recipe_command.command("install")
@click.argument("slug")
def install_recipe(slug: str):
    """Install a recipe and show its setup steps."""
    recipe = _all_recipes().get(slug)
    if not recipe:
        console.print(f"[red]Recipe '{slug}' not found. Try 'forge recipe search'.[/]")
        return
    installed_at = time.strftime("%Y-%m-%d %H:%M")
    record = dict(recipe)
    record["installed_at"] = installed_at
    path = _recipes_dir() / f"{slug}.installed.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    console.print(Panel(
        f"[bold]{recipe['name']}[/]\n[dim]{recipe['description']}[/]\n\nRun these steps:",
        border_style="green",
        title=f"Installed: {slug}",
    ))
    for step in recipe["steps"]:
        console.print(f"  [cyan]$[/] {step}")


@recipe_command.command("fork")
@click.argument("slug")
@click.argument("new_slug")
def fork_recipe(slug: str, new_slug: str):
    """Fork an installed recipe into your own variant."""
    source = _all_recipes().get(slug) or _recipes_dir() / slug
    if isinstance(source, Path):
        if not source.exists():
            console.print(f"[red]Recipe '{slug}' not found.[/]")
            return
        forked = json.loads(source.read_text(encoding="utf-8"))
    else:
        forked = dict(source)
    forked["slug"] = new_slug
    forked["forked_from"] = slug
    out = _recipes_dir() / f"{new_slug}.json"
    out.write_text(json.dumps(forked, indent=2), encoding="utf-8")
    console.print(f"[green]Forked '{slug}' into '{new_slug}'. Edit it at:[/] {out}")
