from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()

LESSONS = {
    "basics": [
        ("Welcome to InferForge", "InferForge runs AI models entirely on your hardware. No cloud, no telemetry."),
        ("Pulling a model", "Run: forge pull qwen2.5-coder:7b\nThis downloads a coding model from HuggingFace."),
        ("Chatting", "Run: forge chat\nThis starts an interactive session with agent tools (file read/write, search)."),
        ("Serving", "Run: forge serve\nExposes an OpenAI-compatible API on http://localhost:11435/v1"),
    ],
    "training": [
        ("Why fine-tune", "Fine-tuning specializes a base model on your data for better accuracy on your tasks."),
        ("Prepare data", "Create a JSON file with prompt/response pairs, or generate some:\nforge generate-data --topic 'Python coding' --count 500"),
        ("Train", "Run: forge train my-model --data training.json\nAdd --lora --lora-r 16 for parameter-efficient tuning."),
        ("Version it", "Tag each run so you can roll back later:\nforge model tag my-model --version v1.0.0"),
    ],
    "deployment": [
        ("Browser AI overview", "'forge web' creates projects that load models from a CDN and run inference with WebGPU in the browser."),
        ("Init a project", "Run: forge web init my-ai-app\nThis scaffolds a tiny static site."),
        ("Add models", "Run: forge web add qwen2.5-coder:7b --progressive\nProgressive loading serves basic responses before the full model arrives."),
        ("Deploy", "Upload the project folder to Vercel, Netlify, or Cloudflare Pages. No servers needed."),
    ],
}


@click.command("learn")
@click.argument("lesson", type=click.Choice(["basics", "training", "deployment"]))
def learn_command(lesson: str):
    """Interactive tutorials: basics, training, or deployment."""
    steps = LESSONS[lesson]
    console.print(Panel(
        f"[bold cyan]Interactive Tutorial:[/] {lesson}\n[dim]{len(steps)} steps - answer prompts to advance[/]",
        border_style="cyan",
    ))

    for i, (title, body) in enumerate(steps, 1):
        console.print(f"\n[bold]Step {i}/{len(steps)}:[/] [green]{title}[/]")
        console.print(body)
        if i < len(steps) and not Confirm.ask("Continue", default=True):
            console.print("[yellow]Tutorial paused. Re-run 'forge learn {}' to continue.[/]".format(lesson))
            return

    console.print("\n[bold green]Tutorial complete![/] Next: check 'forge --help' to explore related commands.")
