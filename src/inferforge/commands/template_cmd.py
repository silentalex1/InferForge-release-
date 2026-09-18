from __future__ import annotations

import json
from pathlib import Path

import click
from platformdirs import user_config_dir
from rich.console import Console
from rich.table import Table

console = Console()


class TemplateManager:
    def __init__(self):
        self.config_dir = Path(user_config_dir("inferforge"))
        self.templates_file = self.config_dir / "templates.json"
        self.templates: dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        if self.templates_file.exists():
            with open(self.templates_file, 'r') as f:
                self.templates = json.load(f)
        else:
            self.templates = {
                "code-review": "Review this code:\n{code}\n\nFocus on: {aspects}\n\nProvide specific suggestions for improvement.",
                "debug": "Help me debug this code:\n{code}\n\nError: {error}\n\nExplain the issue and provide a fix.",
                "explain": "Explain this code in detail:\n{code}\n\nBreak down what each part does and why.",
                "optimize": "Optimize this code for performance:\n{code}\n\nSuggest improvements for speed and memory usage.",
                "translate": "Translate this code from {from_lang} to {to_lang}:\n{code}\n\nMaintain the same functionality.",
                "summarize": "Summarize the following text:\n{text}\n\nProvide a concise summary covering key points.",
                "extract": "Extract {entity_type} from this text:\n{text}\n\nList all instances found.",
                "generate": "Generate {content_type} about {topic}:\n\nRequirements: {requirements}",
                "refactor": "Refactor this code for better maintainability:\n{code}\n\nImprove structure and readability.",
                "test": "Write unit tests for this code:\n{code}\n\nCover edge cases and common scenarios.",
            }
            self._save_templates()
    
    def _save_templates(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.templates_file, 'w') as f:
            json.dump(self.templates, f, indent=2)
    
    def add_template(self, name: str, template: str) -> None:
        self.templates[name] = template
        self._save_templates()
    
    def get_template(self, name: str) -> str | None:
        return self.templates.get(name)
    
    def list_templates(self) -> dict[str, str]:
        return self.templates
    
    def delete_template(self, name: str) -> bool:
        if name in self.templates:
            del self.templates[name]
            self._save_templates()
            return True
        return False


@click.group("template")
def template_command():
    """Manage reusable prompt templates."""
    pass


@template_command.command("list")
def list_templates():
    """List all available templates."""
    manager = TemplateManager()
    templates = manager.list_templates()
    
    if not templates:
        console.print("[yellow]No templates found[/]")
        return
    
    table = Table(title="Prompt Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Template", style="white")
    
    for name, template in templates.items():
        preview = template[:60] + "..." if len(template) > 60 else template
        table.add_row(name, preview)
    
    console.print(table)


@template_command.command("add")
@click.argument("name")
@click.argument("template")
def add_template(name: str, template: str):
    """Add a new template."""
    manager = TemplateManager()
    manager.add_template(name, template)
    console.print(f"[green]✓[/] Template '{name}' added")


@template_command.command("show")
@click.argument("name")
def show_template(name: str):
    """Show template content."""
    manager = TemplateManager()
    template = manager.get_template(name)
    
    if not template:
        console.print(f"[red]Template '{name}' not found[/]")
        return
    
    console.print(f"\n[bold cyan]Template:[/] {name}")
    console.print(f"\n{template}\n")


@template_command.command("delete")
@click.argument("name")
def delete_template(name: str):
    """Delete a template."""
    manager = TemplateManager()
    
    if click.confirm(f"Delete template '{name}'?"):
        if manager.delete_template(name):
            console.print(f"[green]✓[/] Template '{name}' deleted")
        else:
            console.print(f"[red]Template '{name}' not found[/]")


@template_command.command("use")
@click.argument("name")
@click.option("--var", "-v", multiple=True, help="Template variables (key=value)")
@click.option("--model", default="inferforge-beta", help="Model to use for generation")
def use_template(name: str, var: tuple[str, ...], model: str):
    """Use a template with variables."""
    manager = TemplateManager()
    template = manager.get_template(name)
    
    if not template:
        console.print(f"[red]Template '{name}' not found[/]")
        return
    
    variables = {}
    for v in var:
        if "=" in v:
            key, value = v.split("=", 1)
            variables[key] = value
    
    try:
        result = template.format(**variables)
        console.print("\n[bold]Generated Prompt:[/]\n")
        console.print(result)
        
        if click.confirm(f"\nRun this prompt with {model}?"):
            from inferforge.core.registry import Registry
            from inferforge.engine import get_router
            
            reg = Registry()
            record = reg.get(model)
            
            if record:
                router = get_router()
                engine = router.resolve(record)
                
                try:
                    response = engine.chat([{"role": "user", "content": result}])
                    console.print("\n[bold]Response:[/]\n")
                    console.print(response)
                except Exception as e:
                    console.print(f"[red]Error running model:[/] {e}")
            else:
                console.print(f"[red]Model '{model}' not found[/]")
                
    except KeyError as e:
        console.print(f"[red]Missing variable: {e}[/]")
