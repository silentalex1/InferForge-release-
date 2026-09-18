"""forge checkpoint — PyTorch checkpoint management tools.

Commands:
  info                     Inspect a checkpoint (params, dtypes, format)
  validate                 Validate tensors for NaN/Inf/shape issues
  compare                  Diff two checkpoints (missing keys, shape, drift)
  merge                    Merge several checkpoints (average / sum / first)
  prune                    Magnitude-prune a checkpoint
  convert                  Convert between formats (safetensors <-> torch, gguf export)
  list                     List checkpoints in a directory
"""
from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from inferforge.training import torch_checkpoint as tc

console = Console()


def _require_torch() -> None:
    if not tc.torch_available():
        console.print("[red]PyTorch is not installed.[/] Run: [bold]pip install torch[/]")
        raise SystemExit(1)


@click.group("checkpoint")
def checkpoint_group() -> None:
    """PyTorch checkpoint management tools (beta)."""


@checkpoint_group.command("info")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def info(path: Path) -> None:
    """Inspect a .pt/.pth/.ckpt/.bin/.safetensors checkpoint."""
    try:
        info_obj = tc.inspect_checkpoint(path)
    except Exception as exc:
        console.print(f"[red]Failed:[/] {exc}")
        raise SystemExit(1)

    table = Table(title=f"Checkpoint: {path.name}", show_header=True, header_style="bold")
    table.add_column("field", style="dim")
    table.add_column("value", style="cyan")
    table.add_row("path", info_obj.path)
    table.add_row("format", info_obj.format)
    table.add_row("tensors", str(info_obj.keys))
    table.add_row("layers", str(info_obj.layers))
    table.add_row("total params", f"{info_obj.total_params:,}")
    size_mb = info_obj.size_bytes / (1024 * 1024)
    table.add_row("size", f"{size_mb:.1f} MB")
    for dt, count in sorted(info_obj.dtype_counts.items()):
        table.add_row(f"dtype {dt}", str(count))
    if info_obj.embed_dim:
        table.add_row("hidden dim (detected)", str(info_obj.embed_dim))
    if info_obj.vocab_size:
        table.add_row("vocab (detected)", str(info_obj.vocab_size))
    table.add_row("lora weights", "yes" if info_obj.has_lora else "no")
    for note in info_obj.notes:
        table.add_row("note", note)
    console.print(table)


@checkpoint_group.command("validate")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def validate(path: Path) -> None:
    """Validate a checkpoint for NaN/Inf, empty tensors, format issues."""
    ok, issues = tc.validate_checkpoint(path)
    if ok:
        console.print(f"[green]OK[/] checkpoint is valid: {path.name}")
    else:
        console.print(f"[red]Invalid[/] — {len(issues)} issue(s):")
        for issue in issues:
            console.print(f"  [yellow]- {issue}[/]")
        raise SystemExit(1)


@checkpoint_group.command("compare")
@click.argument("a", type=click.Path(exists=True, path_type=Path))
@click.argument("b", type=click.Path(exists=True, path_type=Path))
def compare(a: Path, b: Path) -> None:
    """Diff two checkpoints: keys, shapes, weight drift."""
    try:
        result = tc.compare_checkpoints(a, b)
    except Exception as exc:
        console.print(f"[red]Failed:[/] {exc}")
        raise SystemExit(1)

    table = Table(title="Checkpoint comparison", show_header=True, header_style="bold")
    table.add_column("field", style="dim")
    table.add_column("value", style="cyan")
    table.add_row("shared tensors", str(result["shared_keys"]))
    table.add_row("only in A", str(len(result["only_in_a"])))
    table.add_row("only in B", str(len(result["only_in_b"])))
    table.add_row("identical tensors", str(result["identical_tensors"]))
    table.add_row("shape mismatches", str(len(result["shape_mismatches"])))
    table.add_row("mean drift", str(result["mean_drift"]))
    console.print(table)

    if result["only_in_a"]:
        console.print(f"[yellow]keys only in {Path(result['a']).name}:[/] {result['only_in_a'][:8]}")
    if result["only_in_b"]:
        console.print(f"[yellow]keys only in {Path(result['b']).name}:[/] {result['only_in_b'][:8]}")
    if result["max_drift_keys"]:
        console.print("[dim]largest drift:[/]")
        for entry in result["max_drift_keys"][:5]:
            console.print(f"  [dim]{entry['key']}: {entry['mean_abs_diff']}[/]")


@checkpoint_group.command("merge")
@click.argument("checkpoints", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output checkpoint path.")
@click.option("--strategy", type=click.Choice(["average", "sum", "first"]), default="average", help="Merge strategy.")
def merge(checkpoints: tuple[Path, ...], output: Path, strategy: str) -> None:
    """Merge two or more checkpoints into one."""
    _require_torch()
    if len(checkpoints) < 2:
        console.print("[red]Need at least 2 checkpoints to merge.[/]")
        raise SystemExit(1)
    with console.status("[cyan]Merging checkpoints...[/]"):
        out = tc.merge_checkpoints(list(checkpoints), output, strategy=strategy)
    console.print(f"[green]OK[/] merged {len(checkpoints)} checkpoints ({strategy}) -> [cyan]{out}[/]")


@checkpoint_group.command("prune")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output checkpoint path.")
@click.option("--ratio", default=0.5, type=float, help="Fraction of weights to prune (0.0-1.0).")
def prune(path: Path, output: Path, ratio: float) -> None:
    """Magnitude-prune a checkpoint's weight matrices."""
    _require_torch()
    if not 0 < ratio < 1:
        console.print("[red]ratio must be between 0 and 1[/]")
        raise SystemExit(1)
    with console.status("[cyan]Pruning...[/]"):
        out = tc.prune_checkpoint(path, output, ratio=ratio)
    console.print(f"[green]OK[/] pruned {ratio:.0%} of weights -> [cyan]{out}[/]")


@checkpoint_group.command("convert")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--to", "to_format", type=click.Choice(["torch", "safetensors", "gguf"]), required=True, help="Target format.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output path (defaults alongside input).")
def convert(path: Path, to_format: str, output: Path | None) -> None:
    """Convert a checkpoint between formats."""
    _require_torch()
    import torch  # noqa: F401  (local import — validated above)

    if output is None:
        stem = path.with_suffix("").name or path.stem
        ext = {"torch": ".pt", "safetensors": ".safetensors", "gguf": ".gguf"}[to_format]
        output = path.parent / f"{stem}{ext}"

    state = tc.load_checkpoint(path)

    if to_format == "torch":
        out = tc.save_checkpoint(state, output)
    elif to_format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            console.print("[red]safetensors is required.[/] Run: [bold]pip install safetensors[/]")
            raise SystemExit(1)
        clean = {k: v.contiguous() for k, v in state.items()}
        save_file(clean, str(output))
        out = output
    else:  # gguf
        out = tc.export_gguf(state, output)

    size_mb = out.stat().st_size / (1024 * 1024)
    console.print(f"[green]OK[/] converted {path.name} -> {to_format}: [cyan]{out}[/] ({size_mb:.1f} MB)")


@checkpoint_group.command("list")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def list_checkpoints(directory: Path) -> None:
    """List all checkpoints in a directory, newest first."""
    latest = tc.find_latest_checkpoint(directory)
    candidates = sorted(
        (f for f in directory.iterdir() if f.suffix.lower() in tc.CHECKPOINT_EXTENSIONS),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        console.print("[yellow]No checkpoints found.[/]")
        return
    table = Table(title=f"Checkpoints in {directory.name}", show_header=True, header_style="bold")
    table.add_column("file", style="cyan")
    table.add_column("size", style="white")
    table.add_column("modified", style="dim")
    table.add_column("", style="green")
    for f in candidates:
        size_mb = f.stat().st_size / (1024 * 1024)
        from datetime import datetime
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        marker = "[green]<- latest[/]" if latest and f == latest else ""
        table.add_row(f.name, f"{size_mb:.1f} MB", mtime, marker)
    console.print(table)
