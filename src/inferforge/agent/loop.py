from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from inferforge.agent.mcp_client import get_mcp_client
from inferforge.agent.security import (
    SecurityConfig,
    SecurityManager,
    get_security_manager,
)
from inferforge.agent.tools import (
    execute_tool_calls,
    format_tool_results,
    parse_tool_calls,
    strip_tool_calls,
)
from inferforge.core.registry import ModelRecord
from inferforge.engine.base import ChatEngine, ChatMessage
from inferforge.model.identity import INFERFORGE_BETA, INFERFORGE_BETA_DISPLAY
from inferforge.training.coding_dataset import SYSTEM_PROMPT
from inferforge.ui.render import CHAR_DELAY, render_final_markdown, type_markdown

console = Console(force_terminal=True, stderr=True)

HELP = """
[bold]commands[/]
  /help           show this help
  /clear          clear conversation
  /model          show current model
  /tools          list agent file tools
  /pwd            show workspace
  /cd <path>      change workspace
  /security       show security status
  /audit          show recent audit log
  /undo <path>    restore file from backup
  /exit           quit chat

[bold]agent tools[/]
  create_file · edit_file · delete_file · read_file · open_file · list_dir · run_command · web_request
""".strip()

MAX_TOOL_ROUNDS = 8
TYPE_DELAY = CHAR_DELAY


def _is_beta(model: ModelRecord) -> bool:
    return (
        model.name == INFERFORGE_BETA
        or model.meta.get("own_model")
        or model.meta.get("agentic")
        or "inferforge" in (model.name or "").lower()
    )


def _display_name(model: ModelRecord) -> str:
    if _is_beta(model) or model.name == INFERFORGE_BETA:
        return INFERFORGE_BETA_DISPLAY
    return model.name


def run_agent_chat(
    model: ModelRecord,
    engine: ChatEngine,
    *,
    system: str | None = None,
    options: dict[str, Any] | None = None,
    workspace: Path | None = None,
    agentic: bool | None = None,
    security_config: SecurityConfig | None = None,
    mcp_servers: list[tuple[str, str]] | None = None,
) -> int:
    history: list[ChatMessage] = []
    workspace = (workspace or Path.cwd()).resolve()
    use_agent = agentic if agentic is not None else _is_beta(model)
    sys_prompt = system or (SYSTEM_PROMPT if use_agent else None)
    label = _display_name(model)
    
    # Initialize security manager
    if security_config:
        # Add workspace to allowed workspaces
        security_config.allowed_workspaces.append(workspace)
    security = get_security_manager(security_config)
    
    # Initialize MCP client
    mcp_client = get_mcp_client()
    if mcp_servers:
        for server_name, server_url in mcp_servers:
            mcp_client.add_server(server_name, server_url)

    size_bit = model.parameter_size or "beta"
    quant_bit = model.quantization or model.format or "model"

    console.print(
        Text.from_markup(
            f"[dim]You →[/] [bold]{label}[/]  "
            f"[dim]({size_bit} · {quant_bit})[/]"
        )
    )
    if use_agent:
        console.print(
            Text.from_markup(f"[dim]workspace[/] [cyan]{workspace}[/]  ·  agent tools on")
        )
    if security.config.enable_audit_log:
        console.print(
            Text.from_markup(f"[dim]security[/] [green]audit log enabled[/]  ·  [cyan]{security.config.audit_log_path}[/]")
        )
    if mcp_servers:
        console.print(
            Text.from_markup(f"[dim]MCP servers:[/] {len(mcp_servers)} connected")
        )
    console.print()

    while True:
        try:
            user = console.input("[bold bright_cyan]❯ [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]leaving the forge…[/]")
            return 0

        if not user:
            continue

        if user.startswith("/"):
            code = _handle_slash(user, history, model, label, workspace, security)
            if code == "exit":
                return 0
            if isinstance(code, Path):
                workspace = code
            continue

        history.append(ChatMessage(role="user", content=user))
        console.print()
        console.print("[bold dark_orange]◈[/] [bold]forge[/]", end=" ")
        console.print("[dim]thinking…[/]", end="\r")

        try:
            final_text = _generate_with_tools(
                engine,
                history,
                sys_prompt,
                options,
                workspace=workspace,
                use_agent=use_agent,
                security=security,
            )
        except Exception as exc:
            console.print(f"\n[bold red]error:[/] {exc}")
            if history and history[-1].role == "user":
                history.pop()
            continue

        history.append(ChatMessage(role="assistant", content=final_text))


def _handle_slash(
    user: str,
    history: list[ChatMessage],
    model: ModelRecord,
    label: str,
    workspace: Path,
    security: SecurityManager,
) -> Any:
    parts = user.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in {"/exit", "/quit", "/q"}:
        console.print("[dim]leaving the forge…[/]")
        return "exit"
    if cmd == "/help":
        console.print(Panel(HELP, title="help", border_style="blue"))
        return None
    if cmd == "/clear":
        history.clear()
        console.print("[dim]conversation cleared[/]")
        return None
    if cmd == "/model":
        console.print(
            Panel(
                f"[bold]{label}[/]\n"
                f"id: {model.name}\n"
                f"source: {model.source} · backend: {model.backend}\n"
                f"family: {model.family or '—'} · size: {model.display_size()}\n"
                f"base (internal): {model.meta.get('base_model', '—')}",
                title="model",
                border_style="cyan",
            )
        )
        return None
    if cmd == "/tools":
        tools_list = "create_file · edit_file · delete_file · read_file · open_file · list_dir · run_command · check_storage"
        if security.config.allow_web_access:
            tools_list += " · web_request"
        console.print(
            Panel(
                f"{tools_list}\n"
                "Ask naturally, e.g. [bold]create hello.py that prints hi[/] or [bold]how much storage do I have[/]",
                title="tools",
                border_style="dark_orange",
            )
        )
        return None
    if cmd == "/security":
        access_mode = "[bold red]UNRESTRICTED[/]" if security.config.unrestricted_access else f"{[str(w) for w in security.config.allowed_workspaces]}"
        console.print(
            Panel(
                f"[bold]Access Mode:[/] {access_mode}\n"
                f"[bold]Web Access:[/] {'enabled' if security.config.allow_web_access else 'disabled'}\n"
                f"[bold]Rate Limit:[/] {security.config.web_rate_limit} req/min\n"
                f"[bold]Audit Log:[/] {'enabled' if security.config.enable_audit_log else 'disabled'}\n"
                f"[bold]Backups:[/] {'enabled' if security.config.enable_backups else 'disabled'}\n"
                f"[bold]Consent Required:[/] delete={security.config.require_consent_for_delete}, edit={security.config.require_consent_for_edit}, command={security.config.require_consent_for_command}",
                title="security",
                border_style="cyan",
            )
        )
        return None
    if cmd == "/audit":
        entries = security.get_audit_summary(last_n=20)
        if not entries:
            console.print("[dim]No audit entries yet[/]")
        else:
            for entry in entries:
                status = "[green]✓[/]" if entry["success"] else "[red]✗[/]"
                console.print(
                    f"{status} [dim]{entry['timestamp']}[/] [cyan]{entry['operation']}[/] "
                    f"[yellow]{entry['risk_level']}[/] {entry.get('path') or entry.get('command') or ''}"
                )
        return None
    if cmd == "/undo":
        if not arg:
            console.print("[yellow]usage:[/] /undo <path>")
            return None
        target = Path(arg).expanduser()
        if not target.is_absolute():
            target = (workspace / target).resolve()
        
        # Find latest backup
        backups = list(security.config.backup_dir.glob(f"{target.name}.bak.*"))
        if not backups:
            console.print(f"[red]No backups found for:[/] {target.name}")
            return None
        
        latest = max(backups, key=lambda p: p.stat().st_mtime)
        console.print(f"[dim]Restoring from:[/] {latest.name}")
        
        if security.restore_backup(latest, target):
            console.print(f"[green]Restored:[/] {target}")
        else:
            console.print(f"[red]Restore failed:[/] {target}")
        return None
    if cmd == "/pwd":
        console.print(f"[cyan]{workspace}[/]")
        return None
    if cmd == "/cd":
        if not arg:
            console.print("[yellow]usage:[/] /cd <path>")
            return None
        target = Path(arg).expanduser()
        if not target.is_absolute():
            target = (workspace / target).resolve()
        else:
            target = target.resolve()
        if not target.is_dir():
            console.print(f"[red]not a directory:[/] {target}")
            return None
        
        # Add new workspace to security config if unrestricted access is enabled
        if security.config.unrestricted_access:
            security.config.allowed_workspaces.append(target)
        
        console.print(f"[dim]workspace →[/] [cyan]{target}[/]")
        return target

    console.print(f"[yellow]unknown command:[/] {cmd}  (try /help)")
    return None


def _collect_reply(
    engine: ChatEngine,
    messages: list[ChatMessage],
    system: str | None,
    options: dict[str, Any] | None,
) -> str:
    stream = getattr(engine, "stream_chat", None)
    if callable(stream):
        parts: list[str] = []
        for token in stream(messages, system, options):
            parts.append(token)
        return "".join(parts)
    return engine.chat(messages, system, options)


def _generate_with_tools(
    engine: ChatEngine,
    history: list[ChatMessage],
    system: str | None,
    options: dict[str, Any] | None,
    *,
    workspace: Path,
    use_agent: bool,
    security: SecurityManager,
) -> str:
    working = list(history)
    last_visible = ""
    console.print(" " * 28, end="\r")

    for _round in range(MAX_TOOL_ROUNDS if use_agent else 1):
        raw = _collect_reply(engine, working, system, options)
        last_visible = raw
        calls = parse_tool_calls(raw) if use_agent else []

        if not calls:
            visible = strip_tool_calls(raw) if use_agent else raw
            render_final_markdown(visible, retype=True, delay=TYPE_DELAY)
            return visible or raw

        visible_pre = strip_tool_calls(raw)
        if visible_pre.strip():
            type_markdown(visible_pre, delay=TYPE_DELAY, prefix="◈ ")

        console.print()
        results = execute_tool_calls(raw, workspace, security)
        for r in results:
            style = "green" if r.ok else "red"
            console.print(f"  [{style}]●[/{style}] [bold]{r.name}[/] — {r.message}")
            if r.ok and r.name in {"read_file", "open_file"} and r.data and r.data.get("content"):
                content = str(r.data["content"])
                if content.strip():
                    preview = content if len(content) <= 2000 else content[:2000] + "\n…"
                    console.print(
                        Panel(
                            preview,
                            title=f"[cyan]{r.data.get('path', r.name)}[/]",
                            border_style="cyan",
                            expand=False,
                        )
                    )

        tool_msg = format_tool_results(results)
        working.append(ChatMessage(role="assistant", content=raw))
        working.append(
            ChatMessage(
                role="user",
                content=(
                    "Tool results follow. The actions already ran on disk. "
                    "Give a short confirmation to the user. "
                    "Do not invent more tool JSON unless another action is needed.\n\n"
                    + tool_msg
                ),
            )
        )
        console.print()
        console.print("[bold dark_orange]◈[/] [bold]forge[/]", end=" ")
        console.print("[dim]continuing…[/]", end="\r")

    visible = strip_tool_calls(last_visible) or last_visible
    render_final_markdown(visible, retype=True, delay=TYPE_DELAY)
    return visible
