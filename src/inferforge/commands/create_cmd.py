from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from inferforge.core.config import get_training_config
from inferforge.core.registry import Registry
from inferforge.model.identity import resolve_base_model
from inferforge.nexara.engine import NexaraEngine
from inferforge.training.forge_trainer import ForgeTrainer

console = Console(force_terminal=True, stderr=True)


@click.command("create")
@click.argument("name")
@click.option("--base", default=None, help="Base model to derive from.")
@click.option("--family", default=None, help="Model family label.")
@click.option("--size", default=None, help="Model parameter size label.")
@click.option("--context", default=None, type=int, help="Context length (num_ctx).")
@click.option("--quant", default=None, help="Quantization label.")
@click.option("--temperature", default=None, type=float, help="Sampling temperature.")
@click.option("--top-p", default=None, type=float, help="Sampling top_p.")
@click.option("--top-k", default=None, type=int, help="Sampling top_k.")
@click.option("--system", default=None, help="Custom system prompt.")
@click.option("--coding", is_flag=True, help="Include InferForge coding curriculum.")
@click.option("--nexara", type=click.Path(path_type=click.Path), default=None, help="Path to Nexara code file for AI-native model creation.")
def create_command(
    name: str,
    base: str | None,
    family: str | None,
    size: str | None,
    context: int | None,
    quant: str | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    system: str | None,
    coding: bool,
    nexara: click.Path | None,
) -> None:
    """Create a derived InferForge model (real Ollama create + registry)."""
    if nexara:
        if not nexara.exists():
            console.print(f"[red]Nexara file not found:[/] {nexara}")
            raise SystemExit(1)
        
        nexara_code = nexara.read_text(encoding="utf-8")
        engine = NexaraEngine()
        output_dir = Path.cwd() / "nexara_output"
        
        console.print("[bold dark_orange]◈ Nexara[/] AI-native model creation\n")
        result = engine.compile_and_train(nexara_code, output_dir)
        
        console.print(f"[green]✓[/] Hardware detected: {result['hardware']['cpu_cores']} cores, {result['hardware']['ram']}GB RAM")
        if result['hardware']['gpu_available']:
            console.print(f"[green]✓[/] GPU detected: {result['hardware']['gpu_memory']}GB VRAM")
        
        console.print(f"[green]✓[/] Created {len(result['compiled']['models'])} model(s)")
        for model_name in result['compiled']['models'].keys():
            console.print(f"  - {model_name}")
        
        engine.generate_training_script(result['compiled'], output_dir)
        console.print(f"[green]✓[/] Training script generated: {output_dir / 'train_nexara.py'}")
        console.print("\n[dim]Run training script to start training with Nexara optimizations.[/]")
        return
    
    config = get_training_config()
    if not config["enabled"]:
        console.print("[red]Training is not enabled in settings[/]")
        raise SystemExit(1)

    reg = Registry()
    if base:
        base_record = reg.get(base)
        if not base_record:
            try:
                base = resolve_base_model(base)
            except Exception:
                console.print(f"[red]Base model not found:[/] {base}")
                raise SystemExit(1)
    else:
        base = resolve_base_model()
        console.print(f"[dim]Using base[/] [cyan]{base}[/]")

    params: dict = {}
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if top_k is not None:
        params["top_k"] = top_k
    if context is not None:
        params["num_ctx"] = context

    meta_labels = {}
    if family:
        meta_labels["family"] = family
    if size:
        meta_labels["parameter_size"] = size
    if quant:
        meta_labels["quantization"] = quant

    console.print(f"[bold dark_orange]◈[/] creating [bold cyan]{name}[/] from [cyan]{base}[/]")

    trainer = ForgeTrainer()
    try:
        with Progress(
            SpinnerColumn(style="dark_orange"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("creating…", total=100)

            def on_prog(status: str, frac: float) -> None:
                progress.update(
                    task,
                    completed=max(1, min(99, int(frac * 100))),
                    description=f"[cyan]{status or 'creating'}[/]",
                )

            result = trainer.train_model(
                name,
                base,
                training_data=None,
                system=system,
                max_examples=64 if coding else 1,
                params=params or None,
                progress=on_prog,
                use_builtin_coding=coding,
            )
            progress.update(task, completed=100, description="done")

        if meta_labels:
            rec = reg.get(name)
            if rec:
                if meta_labels.get("family"):
                    rec.family = meta_labels["family"]
                if meta_labels.get("parameter_size"):
                    rec.parameter_size = meta_labels["parameter_size"]
                if meta_labels.get("quantization"):
                    rec.quantization = meta_labels["quantization"]
                rec.meta["own_model"] = True
                reg.upsert(rec)

    except Exception as e:
        console.print(f"[bold red]Model creation failed:[/] {e}")
        raise SystemExit(1) from e

    console.print()
    console.print(f"[green]✓[/] model ready: [bold]{name}[/]")
    console.print(f"  examples: {result.get('examples_embedded', 0)}")
    console.print(f"  path:     {result.get('path', '—')}")
    console.print(f"  train:    [bold]forge train {name} --data examples.json[/]")
    console.print(f"  run:      [bold]forge run {name}[/]")
