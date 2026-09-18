from __future__ import annotations

import sys

import click
from rich.console import Console

from inferforge.core.config import DEFAULT_HOST, DEFAULT_PORT, load_settings

console = Console(force_terminal=True, stderr=True)


@click.command("serve")
@click.option("--host", default=None, help="Bind host.")
@click.option("--port", default=None, type=int, help="Bind port.")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev).")
@click.option("--hot-models", "-m", multiple=True, help="Models to preload and keep in memory. Repeatable.")
def serve_command(host: str | None, port: int | None, reload: bool, hot_models: tuple[str, ...]) -> None:
    """Start the InferForge HTTP server (OpenAI-compatible)."""
    settings = load_settings()
    bind_host = host or settings.get("host") or DEFAULT_HOST
    bind_port = port or int(settings.get("port") or DEFAULT_PORT)

    if hot_models:
        from inferforge.commands.preload_cmd import preload_add

        ctx = click.Context(preload_add)
        ctx.invoke(preload_add, models=tuple(hot_models), parallel=min(len(hot_models), 3))

    import socket as _socket

    import httpx as _httpx

    def _port_in_use(host: str, port: int) -> bool:
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.settimeout(0.8)
            return s.connect_ex((host, port)) == 0

    if _port_in_use(bind_host, bind_port):
        try:
            r = _httpx.get(f"http://{bind_host}:{bind_port}/health", timeout=2.0)
            if r.status_code == 200:
                console.print(f"[yellow]Already serving on http://{bind_host}:{bind_port} — opening chat...[/]")
                import webbrowser
                try:
                    webbrowser.open("https://inferforge.org/chatui")
                except Exception:
                    pass
                console.print("[dim]Press Ctrl+C in the original terminal to stop.[/]")
                return
        except Exception:
            pass
        console.print(f"[yellow]Port {bind_port} in use — waiting 2s to retry...[/]")
        import time as _t
        _t.sleep(2)
        if _port_in_use(bind_host, bind_port):
            console.print(f"[red]Port {bind_port} still in use.[/] Try [cyan]forge serve --port {bind_port + 1}[/] or close the other server.")
            return

    console.print(
        f"[bold dark_orange]◈ InferForge[/] serving on "
        f"[cyan]http://{bind_host}:{bind_port}[/]\n"
        f"[dim]OpenAI-compatible: POST /v1/chat/completions[/]\n"
        f"[dim]Ollama-compatible:  POST /api/chat · GET /v1/models[/]\n"
        f"[dim]Health:             GET  /health[/]\n"
        f"[dim]Chat UI:            https://inferforge.org/chatui[/]"
    )

    always_on = list(settings.get("always_on_models") or [])
    if always_on:
        try:
            from inferforge.server.keepalive import start_keepalive

            start_keepalive(
                always_on,
                interval=settings.get("keep_alive_interval", 240),
                on_event=lambda m: console.print(f"[dim]{m}[/]"),
            )
            console.print(f"[green]✓[/] 24/7 keep-alive active: [cyan]{', '.join(always_on)}[/]")
        except Exception as e:
            console.print(f"[yellow]keep-alive could not start:[/] {e}")

    try:
        from inferforge.core.auth import load_auth_state
        state = load_auth_state()
        if state.get("connected"):
            console.print(f"[green]✓[/] Signed in as [cyan]{state.get('username')}[/] — chat will be auto-signed in")
    except Exception:
        pass

    import threading
    import time
    import webbrowser

    def _open_chat():
        time.sleep(1.2)
        try:
            webbrowser.open("https://inferforge.org/chatui")
        except Exception:
            pass

    threading.Thread(target=_open_chat, daemon=True).start()

    import uvicorn

    try:
        uvicorn.run(
            "inferforge.server.api:app",
            host=bind_host,
            port=bind_port,
            log_level="info",
            reload=reload,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Server shutdown gracefully[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Server error:[/] {e}")
        sys.exit(1)
