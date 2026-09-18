from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
import httpx
from rich.console import Console

console = Console()
UPDATE_API = "https://hyperneural.cfd/api/updates/latest"
ADMIN_UPDATE_API = "https://hyperneural.cfd/admin-panel/api/updates"

@click.command("update")
@click.option("--new", "is_new", is_flag=True, help="Install the latest update pushed from the admin panel.")
@click.option("--check", is_flag=True, help="Check for available updates without installing.")
def update_command(is_new: bool, check: bool) -> None:
    if not is_new and not check:
        console.print("Use [cyan]forge update --new[/] to install the latest admin-pushed update.")
        console.print("Use [cyan]forge update --check[/] to check without installing.")
        return
    try:
        from inferforge.core.auth import load_auth_state
        state = load_auth_state()
        token = None
        try:
            import os
            token = open(Path.home() / ".inferforge" / "admin_token").read_text().strip()
        except Exception:
            pass
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with console.status("[cyan]Checking for updates...[/]"):
            resp = httpx.get(UPDATE_API, timeout=15.0)
        if resp.status_code == 401 or not resp.is_success:
            resp = httpx.get(ADMIN_UPDATE_API, headers=headers, timeout=15.0)
        if not resp.is_success:
            console.print(f"[red]Could not check updates:[/] {resp.status_code}")
            return
        data = resp.json()
        updates = data.get("updates") or ([data.get("update")] if data.get("update") else [])
        if not updates:
            console.print("[yellow]No updates pushed yet.[/]")
            return
        latest = updates[0]
        version = latest.get("version", "unknown")
        notes = latest.get("notes", "")
        pushed = latest.get("pushedAt", "")
        console.print(f"[bold]Latest update:[/] [cyan]{version}[/]  [dim]{pushed}[/]")
        if notes:
            console.print(f"[dim]Notes:[/] {notes}")
        public_version = None
        try:
            with open(Path("public/version.json"), "r", encoding="utf-8") as f:
                public_version = json.load(f).get("version")
        except Exception:
            pass
        if public_version:
            console.print(f"[dim]Public version: {public_version}[/]")
        if check:
            return
        console.print("[cyan]Updating from HyperNeural index...[/]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "inferforge", "--index-url", "https://hyperneural.cfd/pypi/simple/", "--extra-index-url", "https://pypi.org/simple/"])
        console.print(f"[green]OK[/] Updated to {version}. Restart your shell if needed.")
    except Exception as e:
        console.print(f"[red]Update failed:[/] {e}")
        raise SystemExit(1)
