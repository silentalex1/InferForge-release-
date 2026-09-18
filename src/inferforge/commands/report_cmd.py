from __future__ import annotations

import platform
import sys
import uuid
from datetime import datetime, timezone

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()

ADMIN_API = "https://hyperneural.cfd/admin-panel/api"


def _system_info() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "architecture": platform.machine(),
        "forge_version": "0.2.0-beta.1",
    }


@click.command("report")
@click.argument("message", required=False, default=None)
@click.option(
    "--type",
    "report_type",
    type=click.Choice(["bug", "feature", "feedback", "security"], case_sensitive=False),
    default="feedback",
    help="Type of report.",
)
@click.option(
    "--priority",
    type=click.Choice(["low", "medium", "high", "critical"], case_sensitive=False),
    default="medium",
    help="Priority level.",
)
@click.option("--anonymous", is_flag=True, help="Submit without account info.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
def report_command(message: str | None, report_type: str, priority: str, anonymous: bool, yes: bool) -> None:
    """Send a report to the HyperNeural admin panel.

    Examples:

      forge report "Crash when loading large context"

      forge report "Add GGUF quantization" --type feature --priority high
    """
    if not message:
        message = click.prompt("Describe your report")

    payload = {
        "id": str(uuid.uuid4()),
        "type": report_type.lower(),
        "priority": priority.lower(),
        "message": message,
        "anonymous": anonymous,
        "system_info": _system_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "open",
    }

    console.print(
        Panel(
            f"[bold]Message:[/] {message}\n"
            f"[bold]Type:[/] {report_type}   [bold]Priority:[/] {priority}\n"
            f"[bold]Anonymous:[/] {'yes' if anonymous else 'no'}",
            title="[bold cyan]forge report[/]",
            border_style="cyan",
        )
    )

    if not yes and not Confirm.ask("Submit this report?"):
        console.print("[yellow]Cancelled.[/]")
        return

    with console.status("[cyan]Sending report to hyperneural.cfd/admin-panel...[/]"):
        try:
            resp = httpx.post(f"{ADMIN_API}/reports", json=payload, timeout=30.0)
            resp.raise_for_status()
            result = resp.json()
        except httpx.HTTPStatusError as exc:
            console.print(f"[red]Server error {exc.response.status_code}:[/] {exc.response.text[:300]}")
            raise SystemExit(1)
        except httpx.RequestError as exc:
            console.print(f"[red]Could not reach hyperneural.cfd:[/] {exc}")
            console.print("[dim]Check your connection and try again.[/]")
            raise SystemExit(1)

    console.print(f"[green]OK[/] Report submitted. ID: [cyan]{result.get('id', payload['id'])}[/]")
    console.print("[dim]View it at hyperneural.cfd/admin-panel[/]")