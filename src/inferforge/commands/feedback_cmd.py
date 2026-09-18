from __future__ import annotations

import json
import os
import platform
import sys
import time

import click
from rich.console import Console

from inferforge import __version__
from inferforge.core.config import data_dir

console = Console()


def _payload(message: str) -> dict:
    return {
        "created_at": time.time(),
        "message": message,
        "inferforge": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


@click.command("feedback")
@click.argument("message", required=False, default="")
@click.option("--print", "print_payload", is_flag=True, help="Print the payload instead of saving.")
@click.option("--send", is_flag=True, help="Attempt to POST the payload (fails offline).")
def feedback_command(message: str, print_payload: bool, send: bool) -> None:
    """Save anonymous usage feedback locally."""
    if not message:
        try:
            message = click.prompt("Feedback", default="", show_default=False)
        except (EOFError, click.Abort):
            message = ""
    payload = _payload(message.strip())
    if print_payload:
        console.print_json(data=payload)
        return
    dest_dir = data_dir() / "feedback"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"feedback-{int(time.time())}.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]Saved[/] {dest}")
    if send:
        url = os.environ.get("INFERFORGE_FEEDBACK_URL", "")
        if not url:
            console.print("[yellow]No INFERFORGE_FEEDBACK_URL set. Payload kept local.[/]")
            return
        try:
            import httpx

            httpx.post(url, json=payload, timeout=8.0)
            console.print("[green]Sent[/]")
        except Exception as exc:
            console.print(f"[yellow]Could not send (offline or unreachable):[/] {exc}")
