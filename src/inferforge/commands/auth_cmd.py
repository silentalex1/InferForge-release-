from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from inferforge.core.config import config_dir

console = Console()

LICENSE_FILE = "license.json"
KEY_PATTERN = re.compile(r"^[A-Z0-9]{4}(-[A-Z0-9]{4}){3}$")
EDITIONS = {
    "IF-": "Premium",
    "IFP-": "Premium Plus",
    "IFS-": "Starter",
}


def _license_path() -> Path:
    return config_dir() / LICENSE_FILE


def _load_license() -> dict | None:
    path = _license_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("key") else None
    except (OSError, json.JSONDecodeError):
        return None


def _save_license(data: dict | None) -> None:
    path = _license_path()
    if data is None:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _edition_for(key: str) -> str:
    for prefix, edition in EDITIONS.items():
        if key.upper().startswith(prefix):
            return edition
    return "Pro"


def _is_valid_format(key: str) -> bool:
    return bool(KEY_PATTERN.match(key.strip().upper()))


@click.group("auth")
def auth_group():
    """License activation and account status."""


@auth_group.command("status")
def auth_status():
    """Show the current license status."""
    lic = _load_license()
    if lic is None:
        console.print(Panel(
            "[yellow]No license activated.[/]\n\n"
            "InferForge is running in [bold]Community[/] edition.\n"
            "Activate with: [bold]forge auth activate <license-key>[/]",
            title="License Status",
            border_style="dim",
        ))
        return
    table = Table(show_header=False, border_style="dim", title="License Status")
    table.add_column("field", style="dim")
    table.add_column("value", style="cyan")
    table.add_row("edition", str(lic.get("edition", "Pro")))
    table.add_row("key", f"****-****-****-{lic.get('key', '')[-4:]}")
    table.add_row("fingerprint", str(lic.get("fingerprint", "—")))
    table.add_row("activated", str(lic.get("activated_at", "—")))
    table.add_row("status", "[green]active[/]")
    console.print(table)


@auth_group.command("activate")
@click.argument("license_key")
def auth_activate(license_key: str):
    """Activate a license key."""
    key = license_key.strip().upper()
    if not _is_valid_format(key):
        console.print(
            "[red]Invalid license key format.[/] "
            "Expected: [bold]XXXX-XXXX-XXXX-XXXX[/] (letters and digits)."
        )
        raise SystemExit(1)
    _save_license({
        "key": key,
        "fingerprint": _fingerprint(key),
        "edition": _edition_for(key),
        "activated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    console.print(f"[green]✓[/] License activated — [bold cyan]{_edition_for(key)}[/] edition.")


@auth_group.command("deactivate")
def auth_deactivate():
    """Deactivate the current license."""
    if _load_license() is None:
        console.print("[yellow]No license is currently activated.[/]")
        return
    _save_license(None)
    console.print("[green]✓[/] License deactivated. Community edition restored.")


auth_command = auth_group
