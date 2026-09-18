from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from inferforge import __app_name__, __version__

console = Console(force_terminal=True, stderr=True)

CORE_COMMANDS = {
    "pull", "list", "run", "chat", "serve", "import", "remove",
    "doctor", "setup", "logs", "version", "paths", "show", "help",
    "connect", "disconnect", "account", "report", "feedback", "uninstall", "update",
    "merge", "train",
}

BETA_ONLY = {
    "create", "nexara", "retrain",
    "benchmark", "registry", "web", "profile", "template", "compare",
    "optimize", "platform-optimize", "cache", "preload", "stats",
    "api-key", "model", "monitor", "curriculum", "generate-data",
    "test", "explore", "team", "recipe", "learn", "docker", "git",
    "storage", "remote", "plugin", "checkpoint", "forever",
}

GATED_COMMANDS = {"chat", "run", "serve", "train", "embedd", "merge", "pull", "create", "checkpoint"}

_COMMAND_IMPORTS: list[tuple[str, str]] = [
    ("import_cmd", "import_command"),
    ("connect_cmd", "connect_command"),
    ("connect_cmd", "disconnect_command"),
    ("connect_cmd", "account_group"),
    ("report_cmd", "report_command"),
    ("feedback_cmd", "feedback_command"),
    ("pull_cmd", "pull_command"),
    ("list_cmd", "list_command"),
    ("remove_cmd", "remove_command"),
    ("run_cmd", "run_command"),
    ("chat_cmd", "chat_command"),
    ("serve_cmd", "serve_command"),
    ("show_cmd", "show_command"),
    ("show_cmd", "version_command"),
    ("show_cmd", "paths_command"),
    ("storage_cmd", "storage_command"),
    ("storage_cmd", "remote_command"),
    ("create_cmd", "create_command"),
    ("train_cmd", "train_command"),
    ("embedd_cmd", "embedd_command"),
    ("nexara_cmd", "nexara_group"),
    ("help_ai_cmd", "help_command"),
    ("registry_cmd", "registry_command"),
    ("benchmark_cmd", "benchmark_command"),
    ("web_cmd", "web_group"),
    ("profile_cmd", "profile_command"),
    ("template_cmd", "template_command"),
    ("apikey_cmd", "apikey_command"),
    ("compare_cmd", "compare_command"),
    ("cache_cmd", "cache_command"),
    ("stats_cmd", "stats_command"),
    ("version_cmd", "model_command"),
    ("monitor_cmd", "monitor_command"),
    ("platform_optimize_cmd", "platform_optimize_command"),
    ("plugin_cmd", "plugin_command"),
    ("preload_cmd", "preload_command"),
    ("curriculum_cmd", "curriculum_command"),
    ("docker_cmd", "docker_command"),
    ("git_cmd", "git_command"),
    ("learn_cmd", "learn_command"),
    ("recipe_cmd", "recipe_command"),
    ("team_cmd", "team_command"),
    ("test_cmd", "test_command"),
    ("explore_cmd", "explore_command"),
    ("generate_data_cmd", "generate_data_command"),
    ("optimize_cmd", "optimize_command"),
    ("doctor_cmd", "doctor_command"),
    ("setup_cmd", "setup_command"),
    ("logs_cmd", "logs_command"),
    ("checkpoint_cmd", "checkpoint_group"),
    ("merge_cmd", "merge_command"),
    ("retrain_cmd", "retrain_group"),
    ("forever_cmd", "forever_group"),
    ("uninstall_cmd", "uninstall_command"),
    ("update_cmd", "update_command"),
]

_SYMBOL_ALIASES = {
    "model_command": ("version_cmd", "model_group"),
    "monitor_command": ("monitor_cmd", "monitor_group"),
    "platform_optimize_command": ("platform_optimize_cmd", "platform_optimize_group"),
    "logs_command": ("logs_cmd", "logs_group"),
}

def _import_command(module_name: str, symbol: str):
    try:
        module = __import__(f"inferforge.commands.{module_name}", fromlist=[symbol])
        return getattr(module, symbol, None)
    except Exception:
        return None

def _load_commands(beta: bool) -> list[tuple[str, object]]:
    loaded: list[tuple[str, object]] = []
    for module_name, symbol in _COMMAND_IMPORTS:
        cmd = _import_command(module_name, symbol)
        if cmd is None and symbol in _SYMBOL_ALIASES:
            alt_module, alt_symbol = _SYMBOL_ALIASES[symbol]
            cmd = _import_command(alt_module, alt_symbol)
        if cmd is None:
            continue
        name = getattr(cmd, "name", None) or symbol.replace("_command", "").replace("_group", "")
        loaded.append((name, cmd))
    return loaded

class ForgeGroup(click.Group):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._beta = False
    def set_beta(self, beta: bool) -> None:
        self._beta = beta
    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(self.commands)
    def format_commands(self, ctx, formatter) -> None:
        allowed = []
        for name in self.list_commands(ctx):
            cmd = self.commands[name]
            hidden = getattr(cmd, "hidden_beta", False)
            if hidden and not self._beta:
                continue
            allowed.append((name, cmd))
        if not allowed:
            return
        limit = formatter.width - 6 - max(len(n) for n, _ in allowed)
        rows = []
        for name, cmd in allowed:
            help_text = cmd.get_short_help_str(limit)
            rows.append((name, help_text))
        with formatter.section("Commands"):
            formatter.write_dl(rows)
    def resolve_command(self, ctx: click.Context, args: list):
        attempted = args[0] if args else None
        if attempted and attempted in GATED_COMMANDS:
            try:
                from inferforge.core.auth import load_auth_state
                if not load_auth_state().get("connected"):
                    console.print("[red]your not connected. Please use the `/forge connect` command.[/]")
                    raise SystemExit(1)
            except SystemExit:
                raise
            except Exception:
                pass
        beta_enabled = self._beta or bool((ctx.params or {}).get("beta"))
        if attempted and not beta_enabled and attempted in BETA_ONLY and attempted not in CORE_COMMANDS:
            cmd = self.commands.get(attempted)
            if cmd is not None:
                console.print(Panel(f"[yellow]'{attempted}' is a beta command.[/]\n\nRun with the beta flag to enable experimental features:\n[bold]forge --beta {attempted} ...[/]", border_style="yellow"))
                raise SystemExit(2)
        try:
            name, cmd, rest = super().resolve_command(ctx, args)
        except click.exceptions.UsageError as exc:
            msg = str(exc)
            if "No such command" in msg:
                attempted2 = args[0] if args else "?"
                if attempted2 in BETA_ONLY and not beta_enabled:
                    console.print(Panel(f"[yellow]'{attempted2}' is a beta command.[/]\n\nRun with the beta flag to enable experimental features:\n[bold]forge --beta {attempted2} ...[/]", border_style="yellow"))
                    raise SystemExit(2) from exc
                console.print(Panel(f"[red]Unknown command:[/] '{attempted2}'\n\nRun [bold]forge --help[/] to see available commands.", border_style="red"))
                raise SystemExit(2) from exc
            raise
        except Exception as exc:
            console.print(f"[red]Command failed to load:[/] {exc}")
            raise SystemExit(1) from exc
        return name, cmd, rest

@click.group(
    cls=ForgeGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option("--version", "show_ver", is_flag=True, help="Show version and exit.")
@click.option("--beta", is_flag=True, help="Enable experimental beta commands.")
@click.pass_context
def forge(ctx: click.Context, show_ver: bool, beta: bool) -> None:
    import os
    if os.environ.get("FORGE_BETA") == "1":
        beta = True
    ctx.ensure_object(dict)
    forge.set_beta(beta)
    ctx.obj["beta"] = beta
    if show_ver:
        sys.stdout.write(f"{__app_name__} {__version__}\n")
        sys.stdout.flush()
        ctx.exit(0)
    if ctx.invoked_subcommand is None:
        console.print(f"[bold dark_orange]INFERFORGE[/] v{__version__} — faster local LLMs\n")
        _show_all_commands(console, beta)
        ctx.exit(0)

def _register_commands() -> None:
    for name, cmd in _load_commands(beta=False):
        is_beta = name in BETA_ONLY and name not in CORE_COMMANDS
        if is_beta:
            try:
                cmd.hidden_beta = True
            except Exception:
                pass
        forge.add_command(cmd, name=name)

def _show_all_commands(console: Console, beta: bool) -> None:
    core = [
        ("pull", "Pull a model from a remote registry"),
        ("list", "List all registered models"),
        ("run", "Run a model chat session"),
        ("chat", "Open the InferForge chat UI"),
        ("serve", "Start the OpenAI-compatible API server"),
        ("embedd", "Embed a model — AI connected link / SDK for websites"),
        ("import", "Import models from Ollama or other sources"),
        ("remove", "Remove a model from the registry"),
        ("show", "Show model details"),
        ("doctor", "Environment diagnostics and GPU auto-fix"),
        ("setup", "Verify / set up the InferForge environment"),
        ("logs", "View and manage InferForge logs"),
        ("connect", "Connect your InferForge account"),
        ("account", "Show the connected account"),
        ("report", "Send a report to the HyperNeural admin panel"),
        ("feedback", "Send product feedback"),
        ("uninstall", "Uninstall InferForge from this machine"),
        ("update", "Update InferForge to the latest pushed version"),
        ("version", "Show InferForge version"),
        ("paths", "Show InferForge file paths"),
        ("help", "AI-assisted help for any command"),
    ]
    beta_cmds = [
        ("train", "Train / fine-tune models (Nexara DSL + PyTorch checkpoints)"),
        ("create", "Create a new model from scratch"),
        ("checkpoint", "PyTorch checkpoint management tools"),
        ("merge", "Merge models (TIES, SLERP, DARE)"),
        ("nexara", "Nexara AI-native programming language"),
        ("benchmark", "Performance benchmarking and comparison"),
        ("optimize", "Quantization optimizer for your hardware"),
        ("test", "Model quality testing and regression suite"),
        ("monitor", "Live training monitor dashboard"),
        ("web", "Browser-based AI deployment with WebGPU"),
        ("plugin", "Manage custom plugins and extensions"),
    ]
    table = Table(title="Core Commands", show_header=True, header_style="bold dark_orange", border_style="dim", padding=(0, 1))
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    for cmd, desc in core:
        table.add_row(f"forge {cmd}", desc)
    console.print(table)
    if beta:
        btable = Table(title="Beta Commands (experimental)", show_header=True, header_style="bold yellow", border_style="dim", padding=(0, 1))
        btable.add_column("Command", style="yellow", no_wrap=True)
        btable.add_column("Description", style="white")
        for cmd, desc in beta_cmds:
            btable.add_row(f"forge {cmd}", desc)
        console.print(btable)
    else:
        console.print("\n[yellow]More experimental commands are available with [bold]forge --beta --help[/].[/]")
    console.print()
    console.print("[dim]Use [bold]forge <command> --help[/] for detailed command help.")

_register_commands()

# Standalone `run` entry point (see [project.scripts] in pyproject.toml).
run_cmd = forge.commands.get("run")
run = run_cmd

if __name__ == "__main__":
    forge()
