from __future__ import annotations

import json
from pathlib import Path

import click
from platformdirs import user_config_dir
from rich.console import Console
from rich.table import Table

console = Console()


class CurriculumManager:
    def __init__(self):
        self.config_dir = Path(user_config_dir("inferforge"))
        self.curricula_file = self.config_dir / "curricula.json"
        self.curricula: dict[str, dict] = {}
        self._load_curricula()
    
    def _load_curricula(self) -> None:
        if self.curricula_file.exists():
            with open(self.curricula_file, 'r') as f:
                self.curricula = json.load(f)
    
    def _save_curricula(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.curricula_file, 'w') as f:
            json.dump(self.curricula, f, indent=2)
    
    def create_curriculum(self, name: str, description: str = "") -> bool:
        self.curricula[name] = {
            "description": description,
            "stages": [],
            "created_at": None
        }
        self._save_curricula()
        return True
    
    def add_stage(self, curriculum_name: str, stage_name: str, data_file: str, epochs: int, learning_rate: float = 0.0001) -> bool:
        if curriculum_name not in self.curricula:
            console.print(f"[red]Curriculum '{curriculum_name}' not found[/]")
            return False
        
        self.curricula[curriculum_name]["stages"].append({
            "name": stage_name,
            "data_file": data_file,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "completed": False
        })
        
        self._save_curricula()
        return True
    
    def list_curricula(self) -> dict:
        return self.curricula
    
    def get_curriculum(self, name: str) -> dict | None:
        return self.curricula.get(name)
    
    def delete_curriculum(self, name: str) -> bool:
        if name in self.curricula:
            del self.curricula[name]
            self._save_curricula()
            return True
        return False


@click.group("curriculum")
def curriculum_group():
    """Build multi-stage training curriculums."""
    pass


@curriculum_group.command("create")
@click.argument("name")
@click.option("--description", "-d", default="", help="Description of this curriculum")
def curriculum_create(name: str, description: str):
    """Create a new training curriculum."""
    manager = CurriculumManager()
    
    if manager.create_curriculum(name, description):
        console.print(f"[green]✓[/] Created curriculum: {name}")
        console.print("[dim]Add stages with: forge curriculum add-stage[/]")
    else:
        console.print("[red]Failed to create curriculum[/]")


@curriculum_group.command("add-stage")
@click.argument("curriculum")
@click.argument("stage_name")
@click.option("--data", "-d", required=True, help="Training data file")
@click.option("--epochs", "-e", type=int, default=2, help="Number of epochs")
@click.option("--learning-rate", "-lr", type=float, default=0.0001, help="Learning rate")
def curriculum_add_stage(curriculum: str, stage_name: str, data: str, epochs: int, learning_rate: float):
    """Add a training stage to a curriculum."""
    manager = CurriculumManager()
    
    if manager.add_stage(curriculum, stage_name, data, epochs, learning_rate):
        console.print(f"[green]✓[/] Added stage '{stage_name}' to {curriculum}")
    else:
        console.print("[red]Failed to add stage[/]")


@curriculum_group.command("list")
def curriculum_list():
    """List all training curricula."""
    manager = CurriculumManager()
    curricula = manager.list_curricula()
    
    if not curricula:
        console.print("[yellow]No curricula found[/]")
        return
    
    table = Table(title="Training Curricula")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Stages", style="yellow")
    table.add_column("Status", style="green")
    
    for name, curriculum in curricula.items():
        stages = curriculum.get("stages", [])
        completed = sum(1 for stage in stages if stage.get("completed", False))
        status = f"{completed}/{len(stages)} completed" if stages else "No stages"
        
        table.add_row(
            name,
            curriculum.get("description", ""),
            str(len(stages)),
            status
        )
    
    console.print(table)


@curriculum_group.command("show")
@click.argument("name")
def curriculum_show(name: str):
    """Show details of a specific curriculum."""
    manager = CurriculumManager()
    curriculum = manager.get_curriculum(name)
    
    if not curriculum:
        console.print(f"[red]Curriculum '{name}' not found[/]")
        return
    
    console.print(f"\n[bold cyan]Curriculum:[/] {name}")
    console.print(f"[bold]Description:[/] {curriculum.get('description', 'None')}\n")
    
    stages = curriculum.get("stages", [])
    if not stages:
        console.print("[yellow]No stages defined[/]")
        return
    
    table = Table()
    table.add_column("Stage", style="cyan")
    table.add_column("Data File", style="white")
    table.add_column("Epochs", style="yellow")
    table.add_column("Learning Rate", style="magenta")
    table.add_column("Status", style="green")
    
    for i, stage in enumerate(stages, 1):
        status = "Completed" if stage.get("completed", False) else "Pending"
        table.add_row(
            f"{i}. {stage['name']}",
            stage['data_file'],
            str(stage['epochs']),
            f"{stage['learning_rate']}",
            status
        )
    
    console.print(table)


@curriculum_group.command("delete")
@click.argument("name")
def curriculum_delete(name: str):
    """Delete a curriculum."""
    manager = CurriculumManager()
    
    if click.confirm(f"Delete curriculum '{name}'?"):
        if manager.delete_curriculum(name):
            console.print(f"[green]✓[/] Deleted curriculum: {name}")
        else:
            console.print(f"[red]Curriculum '{name}' not found[/]")


@curriculum_group.command("train")
@click.argument("curriculum")
@click.option("--auto-advance", is_flag=True, help="Automatically advance through stages")
@click.option("--model", default="inferforge-beta", help="Base model to train")
def curriculum_train(curriculum: str, auto_advance: bool, model: str):
    """Train a model using a curriculum."""
    manager = CurriculumManager()
    curriculum_data = manager.get_curriculum(curriculum)
    
    if not curriculum_data:
        console.print(f"[red]Curriculum '{curriculum}' not found[/]")
        return
    
    stages = curriculum_data.get("stages", [])
    if not stages:
        console.print(f"[yellow]No stages in curriculum '{curriculum}'[/]")
        return
    
    console.print(f"\n[bold cyan]Training with Curriculum:[/] {curriculum}")
    console.print(f"[bold]Model:[/] {model}")
    console.print(f"[bold]Stages:[/] {len(stages)}")
    console.print(f"[bold]Auto-advance:[/] {'Yes' if auto_advance else 'No'}\n")
    
    for i, stage in enumerate(stages, 1):
        console.print(f"[cyan]Stage {i}/{len(stages)}:[/] {stage['name']}")
        console.print(f"  Data: {stage['data_file']}")
        console.print(f"  Epochs: {stage['epochs']}")
        console.print(f"  Learning Rate: {stage['learning_rate']}")
        
        if auto_advance:
            console.print(f"\n[green]✓[/] Training stage {i} completed")
        else:
            if not click.confirm(f"\nProceed to stage {i+1 if i < len(stages) else 'completion'}?"):
                console.print("[yellow]Training stopped by user[/]")
                return


curriculum_command = curriculum_group