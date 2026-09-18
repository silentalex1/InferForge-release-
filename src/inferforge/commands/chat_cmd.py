"""forge chat — open InferForge beta chat UI (own model, agentic coding)."""

from __future__ import annotations

from pathlib import Path

import click
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from inferforge.agent.loop import run_agent_chat
from inferforge.agent.security import SecurityConfig
from inferforge.core.config import load_settings
from inferforge.core.registry import Registry
from inferforge.engine import get_router
from inferforge.model.identity import (
    INFERFORGE_BETA,
    INFERFORGE_BETA_DISPLAY,
    ensure_inferforge_beta,
)
from inferforge.ui.animation import play_boot_animation

console = Console(force_terminal=True, stderr=True)


def _ready_panel(subtitle: str = "type /help · /exit to quit") -> None:
    """Match the classic ready panel, but brand InferForge beta (beta in orange)."""
    title = Text()
    title.append("InferForge ", style="bold green")
    title.append("beta", style="bold dark_orange")
    title.append(" online", style="bold green")

    body = Group(
        Align.center(title),
        Align.center(Text("chatting with InferForge beta", style="cyan")),
        Align.center(Text(subtitle, style="dim")),
    )
    console.print(
        Panel(
            body,
            border_style="green",
            title="[bold]ready[/]",
        )
    )
    console.print()


@click.command("chat")
@click.option("--model", "model_name", default=None, help="Override model (default: inferforge-beta).")
@click.option("--system", default=None, help="Optional extra system prompt.")
@click.option("--no-animation", is_flag=True, help="Skip the boot animation.")
@click.option("--rebuild", is_flag=True, help="Force rebuild InferForge beta before chatting.")
@click.option("--base", default=None, help="Base Ollama model to train from when building.")
@click.option("--workspace", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=None)
@click.option("--no-agent", is_flag=True, help="Disable file tools.")
@click.option("--verbose", "-v", is_flag=True, help="Extra diagnostics.")
@click.option("--allow-web", is_flag=True, help="Enable web request tool.")
@click.option("--web-domains", default=None, help="Comma-separated list of allowed web domains.")
@click.option("--no-backup", is_flag=True, help="Disable automatic backups.")
@click.option("--no-audit", is_flag=True, help="Disable audit logging.")
@click.option("--no-consent", is_flag=True, help="Disable consent prompts for operations.")
@click.option("--unrestricted", is_flag=True, help="Allow access to any directory on the system (use with caution).")
@click.option("--mcp-server", multiple=True, help="Add MCP server (format: name=url).")
def chat_command(
    model_name: str | None,
    system: str | None,
    no_animation: bool,
    rebuild: bool,
    base: str | None,
    workspace: Path | None,
    no_agent: bool,
    verbose: bool,
    allow_web: bool,
    web_domains: str | None,
    no_backup: bool,
    no_audit: bool,
    no_consent: bool,
    unrestricted: bool,
    mcp_server: tuple[str, ...],
) -> None:
    """Open the InferForge beta chat UI (own model · coding agent)."""
    settings = load_settings()
    animate = (not no_animation) and bool(settings.get("animation", True))
    target = (model_name or INFERFORGE_BETA).strip()

    # Ensure house model exists / is trained
    if target in {INFERFORGE_BETA, "inferforge", "beta", INFERFORGE_BETA_DISPLAY.lower()}:
        target = INFERFORGE_BETA
        console.print(
            Text.from_markup(
                "[bold dark_orange]◈[/] preparing [bold]InferForge [/][bold dark_orange]beta[/]…"
            )
        )
        try:
            with Progress(
                SpinnerColumn(style="dark_orange"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=28),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("ensuring model…", total=100)

                def on_prog(status: str, frac: float) -> None:
                    progress.update(
                        task,
                        completed=max(1, int(frac * 100)),
                        description=f"[cyan]{status or 'building'}[/]",
                    )

                record = ensure_inferforge_beta(
                    force_rebuild=rebuild,
                    base=base,
                    progress=on_prog if rebuild else None,
                )
                # If not rebuilt, still verify quickly
                if not rebuild:
                    progress.update(task, completed=100, description="ready")
                else:
                    progress.update(task, completed=100, description="trained")
        except Exception as exc:
            console.print(f"[bold red]failed to prepare InferForge beta:[/] {exc}")
            console.print(
                "[dim]Tips: start Ollama (`ollama serve`), import models "
                "(`forge import ollama`), or pick a base with --base qwen2.5-coder:7b[/]"
            )
            raise SystemExit(1) from exc
    else:
        record = Registry().get(target)
        if record is None:
            console.print(
                f"[red]model not found:[/] [bold]{target}[/]\n"
                "  • [bold]forge list[/]\n"
                "  • [bold]forge chat[/]  (uses InferForge beta)\n"
            )
            raise SystemExit(1)

    if verbose:
        console.print(
            f"[dim]model={record.name} backend={record.backend} "
            f"ollama={record.ollama_name} base={record.meta.get('base_model')}[/]"
        )

    if animate:
        play_boot_animation(INFERFORGE_BETA_DISPLAY if record.name == INFERFORGE_BETA else record.name)
    else:
        _ready_panel()

    router = get_router()
    engine = None
    try:
        engine = router.resolve(record)
    except Exception as exc:
        console.print(f"[red]Error resolving engine:[/] {exc}")
        console.print("[dim]Is Ollama running?  ollama serve[/]")
        raise SystemExit(1) from exc

    options = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "repeat_penalty": 1.15,
    }
    try:
        from inferforge.optimizer import get_generation_profile

        options.update(get_generation_profile(record.name).get_sampling_options())
    except Exception:
        pass

    # Configure security
    security_config = SecurityConfig(
        allow_web_access=allow_web,
        allowed_web_domains=set(web_domains.split(",")) if web_domains else set(),
        enable_backups=not no_backup,
        enable_audit_log=not no_audit,
        require_consent_for_delete=not no_consent,
        require_consent_for_command=not no_consent,
        unrestricted_access=unrestricted,
    )
    
    # Parse MCP servers
    mcp_servers = None
    if mcp_server:
        mcp_servers = []
        for server_spec in mcp_server:
            if "=" in server_spec:
                name, url = server_spec.split("=", 1)
                mcp_servers.append((name, url))
    
    try:
        code = run_agent_chat(
            record,
            engine,
            system=system,
            options=options,
            workspace=workspace or Path.cwd(),
            agentic=not no_agent,
            security_config=security_config,
            mcp_servers=mcp_servers,
        )
    except Exception as exc:
        console.print(f"[red]chat error:[/] {exc}")
        code = 1
    finally:
        if engine:
            engine.close()
    raise SystemExit(code)
