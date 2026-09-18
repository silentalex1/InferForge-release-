"""AI-friendly help command for InferForge."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@click.command("help")
@click.option("--ai", is_flag=True, help="Show AI-friendly help with simple explanations.")
def help_command(ai: bool) -> None:
    if not ai:
        return
    
    console = Console()
    
    console.print()
    console.print(Panel(
        Text("INFERFORGE", style="bold dark_orange"),
        title="What is InferForge?",
        border_style="dark_orange",
        padding=(1, 2)
    ))
    
    console.print()
    console.print(Panel(
        Text(
            "InferForge is a tool that helps you run AI models on your own computer.\n\n"
            "Think of it like this:\n"
            "- You can download AI models (like smart chat bots)\n"
            "- Run them on your computer without internet\n"
            "- Use them in your own projects (websites, apps, bots)\n"
            "- Make them work faster and better\n\n"
            "It's like having your own AI assistant that lives on your computer\n"
            "and works without needing to talk to big servers on the internet.",
            style="white"
        ),
        border_style="dim",
        padding=(1, 2)
    ))
    
    console.print()
    console.print(Panel(
        Text("How it works", style="bold cyan"),
        border_style="cyan",
        padding=(0, 2)
    ))
    
    console.print()
    steps = [
        "1. Import or pull a model (get the AI brain)",
        "2. Run or chat with the model (talk to it)",
        "3. Embed it in projects (use it in your apps)",
        "4. Train or improve it (make it smarter)"
    ]
    
    for step in steps:
        console.print(f"  {step}")
    
    console.print()
    console.print(Panel(
        Text("All Commands", style="bold green"),
        border_style="green",
        padding=(0, 2)
    ))
    
    console.print()
    _show_simple_commands(console)
    
    console.print()
    console.print(Panel(
        Text("Nexara - Custom AI Coding Language", style="bold magenta"),
        border_style="magenta",
        padding=(0, 2)
    ))
    
    console.print()
    console.print(Panel(
        Text(
            "Nexara is a special coding language made just for AI.\n\n"
            "What makes it special:\n"
            "- Built for AI models from the ground up\n"
            "- Simple words to tell the AI what to learn\n"
            "- No complex code needed\n"
            "- Works directly with AI training\n\n"
            "How it works:\n"
            "1. Write simple instructions in Nexara\n"
            "2. Nexara converts them to AI training code\n"
            "3. The AI learns from your instructions\n"
            "4. You get a smarter custom AI model\n\n"
            "It's like teaching a computer in a language it understands best.",
            style="white"
        ),
        border_style="dim",
        padding=(1, 2)
    ))
    
    console.print()
    console.print(Panel(
        Text("Training & Fine-Tuning AI Models", style="bold yellow"),
        border_style="yellow",
        padding=(0, 2)
    ))
    
    console.print()
    console.print(Panel(
        Text(
            "Training means teaching an AI to be better at something.\n\n"
            "Fine-tuning is like giving an AI extra lessons.\n\n"
            "How Forge helps:\n"
            "- Start with a base model (like Llama or GPT)\n"
            "- Use Nexara to write what you want it to learn\n"
            "- Forge trains the model with your data\n"
            "- Get a custom model that knows your stuff\n\n"
            "Example uses:\n"
            "- Teach it your company's writing style\n"
            "- Make it an expert in your field\n"
            "- Train it on your documents\n"
            "- Create specialized helpers\n\n"
            "The training happens on your computer, so your data stays private.",
            style="white"
        ),
        border_style="dim",
        padding=(1, 2)
    ))
    
    console.print()
    console.print(Panel(
        Text(
            "Key Concepts:\n"
            "- Model: The AI brain (like GPT, Llama, etc.)\n"
            "- Registry: Where Forge keeps track of your models\n"
            "- Embed: Put the model into your project files\n"
            "- Portable: Works without Forge installed\n"
            "- Reference-only: Small files, downloads model when needed",
            style="white"
        ),
        border_style="dim",
        padding=(1, 2)
    ))
    
    console.print()
    console.print(Panel(
        Text(
            "Why use InferForge?\n"
            "- Privacy: Your data stays on your computer\n"
            "- Speed: Runs faster than internet services\n"
            "- Cost: Free after you download the model\n"
            "- Control: You own the model and can change it\n"
            "- Offline: Works without internet connection",
            style="white"
        ),
        border_style="dim",
        padding=(1, 2)
    ))


def _show_simple_commands(console: Console) -> None:
    commands = [
        ("forge import", "Get models from Ollama or other places"),
        ("forge pull", "Download a model from the internet"),
        ("forge list", "See all models you have"),
        ("forge remove", "Delete a model you don't want"),
        ("forge run", "Talk to a model in your terminal"),
        ("forge chat", "Open a nice chat window to talk to AI"),
        ("forge serve", "Start a web server so other apps can use the AI"),
        ("forge show", "See details about a model"),
        ("forge version", "Check what version of Forge you have"),
        ("forge paths", "See where Forge keeps its files"),
        ("forge storage", "Manage where models are saved"),
        ("forge remote", "Manage places to get models from"),
        ("forge create", "Make a new AI model from scratch"),
        ("forge train", "Teach a model to be better"),
        ("forge embedd", "Put a model into your project files"),
        ("forge nexara", "Use the Nexara programming language"),
    ]
    
    table = Table(
        show_header=True,
        header_style="bold green",
        border_style="dim",
        padding=(0, 1)
    )
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("What it does", style="white")
    
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    
    console.print(table)
