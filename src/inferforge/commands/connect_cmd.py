from __future__ import annotations

import secrets
import string
import time
import webbrowser
from datetime import datetime, timezone

import click
import httpx
from rich.console import Console
from rich.panel import Panel

from inferforge.core.auth import load_auth_state, save_auth_state

console = Console()

CONNECT_API = "https://inferforge-email.asdwwas233.workers.dev"
SITE_URL = "https://inferforge.org/account"


def _gen_code() -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(5))


def _get_username(state: dict) -> str | None:
    """Get username from auth state, falling back to last_used_username, or prompt."""
    if state.get("connected") and state.get("username"):
        return state["username"]
    return state.get("last_used_username")


def _prompt_username() -> str:
    """Prompt the user for their InferForge username."""
    username = input("Enter your InferForge username: ").strip()
    return username


@click.command("connect")
@click.option("--no-browser", is_flag=True, help="Do not open the browser automatically.")
@click.option("--timeout", default=300, type=int, help="Seconds to wait for verification.")
def connect_command(no_browser: bool, timeout: int) -> None:
    """Connect your InferForge account to the CLI."""
    state = load_auth_state()
    if state.get("connected"):
        console.print(
            Panel(
                f"[green]Already connected as[/] [cyan]{state.get('username')}[/] ({state.get('email')})\n"
                "[dim]Run 'forge disconnect' first to switch accounts.[/]",
                border_style="green",
            )
        )
        return

    username = _get_username(state)
    if not username:
        username = _prompt_username()
    if not username:
        console.print(
            "[red]Username not found. "
            "Run 'forge connect' after setting up an account via the website, "
            "or run 'forge connect --no-browser --timeout 60' to start a new session.[/]"
        )
        raise SystemExit(1)

    code = _gen_code()

    with console.status("[cyan]Registering verification session...[/]"):
        try:
            resp = httpx.post(
                f"{CONNECT_API}/connect",
                json={"username": username, "code": code},
                timeout=30.0,
            )
            resp.raise_for_status()
        except httpx.RequestError as exc:
            console.print(f"[red]Could not reach the verification service:[/] {exc}")
            raise SystemExit(1)

    console.print("[cyan]Opening to account verification...[/]")
    url = f"{SITE_URL}?verify={username}"
    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    console.print(f"  connect: [cyan][link={url}]{url}[/link][/]")
    console.print("  [dim]Enter the code below on the website to link your account.[/]")

    console.print(
        Panel(
            f"[bold]Paste in this code to connect:[/]\n\n"
            f"[bold cyan]{code}[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print("[dim]Waiting for verification on inferforge.org...[/]")

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{CONNECT_API}/connect/{username}", timeout=15.0)
            if r.is_success:
                data = r.json()
                if data.get("confirmed") and data.get("code", "").upper() == code:
                    email = data.get("email", "")
                    save_auth_state({
                        "connected": True,
                        "username": username,
                        "email": email,
                        "connected_at": datetime.now(timezone.utc).isoformat(),
                        "last_used_username": username,
                    })
                    console.print(f"\n[green]account connected. Welcome {username}.[/]")
                    console.print(f"[dim]Connected to account of:[/] {email} - {username}")
                    console.print("[green]OK[/] Run [bold]forge account[/] for details.")
                    return
        except httpx.RequestError:
            pass
        time.sleep(2)

    console.print("[red]Verification timed out.[/] Run 'forge connect' to try again.")
    raise SystemExit(1)


@click.command("disconnect")
def disconnect_command() -> None:
    """Disconnect your InferForge account from the CLI."""
    state = load_auth_state()
    if not state.get("connected"):
        console.print("[yellow]Not connected.[/]")
        return
    save_auth_state({"connected": False, "last_used_username": state.get("username")})
    console.print(f"[green]OK[/] Disconnected {state.get('username')}.")


@click.group("account", invoke_without_command=True)
@click.pass_context
def account_group(ctx: click.Context) -> None:
    """Show the connected InferForge account."""
    if ctx.invoked_subcommand is not None:
        return
    state = load_auth_state()
    if not state.get("connected"):
        console.print("[yellow]Not connected.[/] Run [bold]forge connect[/] to link your account.")
        return
    console.print(
        Panel(
            f"[bold]Username:[/] [cyan]{state.get('username')}[/]\n"
            f"[bold]Email:[/] [cyan]{state.get('email')}[/]\n"
            f"[bold]Connected:[/] {state.get('connected_at', '')}",
            title="[bold cyan]Account[/]",
            border_style="cyan",
        )
    )


@account_group.command("reset")
@click.option("--username", "username_opt", default=None, help="Account username (defaults to the connected account).")
@click.option("--no-browser", is_flag=True, help="Do not open the browser automatically.")
def account_reset_command(username_opt: str | None, no_browser: bool) -> None:
    """Reset your account password on inferforge.org."""
    state = load_auth_state()
    username = (username_opt or _get_username(state) or "").strip()
    if not username:
        username = _prompt_username()
    if not username:
        console.print("[red]No username given.[/] Run [bold]forge account reset --username <name>[/].")
        raise SystemExit(1)

    code = _gen_code()
    try:
        resp = httpx.post(
            f"{CONNECT_API}/api/auth/reset-request",
            json={"username": username, "code": code},
            timeout=30.0,
        )
    except httpx.RequestError as exc:
        console.print(f"[red]Could not reach the account service:[/] {exc}")
        raise SystemExit(1)

    if resp.status_code == 404:
        console.print(f"[red]No inferforge.org account named[/] [bold]{username}[/].")
        console.print("  Create one at [cyan]https://inferforge.org/create-account[/] and run this again.")
        raise SystemExit(1)
    if not resp.is_success:
        detail = ""
        try:
            detail = str(resp.json().get("error") or "")
        except ValueError:
            detail = resp.text.strip()
        console.print(f"[red]Reset request failed:[/] {detail or resp.status_code}")
        raise SystemExit(1)

    url = f"https://inferforge.org/reset/{username}?code={code}"
    console.print("[cyan]opening the reset account page..[/]")
    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    console.print(f"  reset page: [cyan][link={url}]{url}[/link][/]")
    console.print("  [dim]The link is valid for 10 minutes.[/]")
