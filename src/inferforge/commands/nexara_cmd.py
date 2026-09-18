from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from inferforge.nexara.engine import NexaraEngine
from inferforge.nexara.parser import NexaraParser

console = Console(force_terminal=True, stderr=True)


@click.group("nexara")
def nexara_group():
    """Nexara AI-native programming language commands."""
    pass


@nexara_group.command("compile")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory.")
def compile_command(file: Path, output: Path | None) -> None:
    """Compile a Nexara code file."""
    engine = NexaraEngine()
    parser = NexaraParser()
    
    code = file.read_text(encoding="utf-8")
    output_dir = output or file.parent / "nexara_output"
    
    console.print("[bold dark_orange]◈ Nexara[/] AI-native compilation\n")
    
    parsed = parser.parse(code)
    result = engine.compile_and_train(code, output_dir)
    
    console.print(f"[green]✓[/] Hardware detected: {result['hardware']['cpu_cores']} cores, {result['hardware']['ram']}GB RAM")
    if result['hardware']['gpu_available']:
        console.print(f"[green]✓[/] GPU detected: {result['hardware']['gpu_memory']}GB VRAM")
    
    console.print(f"[green]✓[/] Compiled {len(result['compiled']['models'])} model(s)")
    for model_name in result['compiled']['models'].keys():
        console.print(f"  - {model_name}")
    
    if result['compiled'].get('evolutions'):
        console.print(f"[green]✓[/] {len(result['compiled']['evolutions'])} evolution(s) configured")
    
    if result['compiled'].get('swarms'):
        console.print(f"[green]✓[/] {len(result['compiled']['swarms'])} swarm(s) configured")
    
    engine.generate_training_script(result['compiled'], output_dir)
    console.print(f"[green]✓[/] Training script: {output_dir / 'train_nexara.py'}")
    console.print(f"[green]✓[/] Config: {output_dir / 'nexara_config.json'}")


@nexara_group.command("evolve")
@click.argument("model")
@click.option("--goal", default="reasoning", help="Evolution goal (reasoning, memory, speed).")
@click.option("--iterations", default=5, type=int, help="Number of evolution iterations.")
def evolve_command(model: str, goal: str, iterations: int) -> None:
    """Evolve a model architecture automatically."""
    engine = NexaraEngine()
    
    console.print(f"[bold dark_orange]◈ Nexara[/] evolving {model}\n")
    console.print(f"Goal: {goal}")
    console.print(f"Iterations: {iterations}\n")
    
    result = engine.evolve_model(model, goal, iterations)
    
    console.print("[green]✓[/] Evolution complete")
    console.print(f"Total improvement: {result['total_improvement']:.1%}")
    
    table = Table(title="Evolution History")
    table.add_column("Iteration", style="cyan")
    table.add_column("Changes", style="white")
    table.add_column("Improvement", style="green")
    
    for h in result['history']:
        table.add_row(str(h['iteration']), ", ".join(h['changes']), f"{h['improvement']:.1%}")
    
    console.print(table)


@nexara_group.command("compress")
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option("--ratio", default=0.8, type=float, help="Compression ratio (0.0-1.0).")
def compress_command(model_path: Path, ratio: float) -> None:
    """Compress a model using neural pruning."""
    engine = NexaraEngine()
    
    console.print(f"[bold dark_orange]◈ Nexara[/] compressing {model_path.name}\n")
    console.print(f"Target compression: {ratio:.0%}\n")
    
    result = engine.compress_model(model_path, ratio)
    
    console.print("[green]✓[/] Compression complete")
    console.print(f"Original size: {result['original_size']:,} bytes")
    console.print(f"Target size: {result['target_size']:,} bytes")
    console.print(f"Method: {result['method']}")


@nexara_group.command("swarm")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
def swarm_command(config_file: Path) -> None:
    """Setup a neural swarm for distributed training."""
    import json
    
    engine = NexaraEngine()
    
    with config_file.open() as f:
        config = json.load(f)
    
    console.print("[bold dark_orange]◈ Nexara[/] setting up swarm\n")
    
    result = engine.setup_swarm(config)
    
    console.print(f"[green]✓[/] Swarm active: {result['swarm_name']}")
    console.print(f"Devices connected: {result['devices_connected']}")
    console.print(f"Load balancing: {result['distribution']}")


@nexara_group.command("init")
@click.argument("name")
def init_command(name: str) -> None:
    """Initialize a new Nexara project."""
    output_dir = Path.cwd() / name
    output_dir.mkdir(exist_ok=True)
    
    template = f'''brain "{name}" {{
    architecture = transformer
    parameters = 7B
    learning = adaptive
    memory = intelligent
    
    reasoning
    memory
    creativity
}}

evolve {name} {{
    goal = "maximum reasoning"
    minimum memory
}}
'''
    
    nexara_file = output_dir / f"{name}.nexara"
    nexara_file.write_text(template)
    
    console.print(f"[green]✓[/] Nexara project created: {output_dir}")
    console.print(f"Edit: {nexara_file}")
    console.print(f"Compile: forge train --nexara {nexara_file}")
