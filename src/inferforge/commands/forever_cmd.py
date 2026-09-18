from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from inferforge.nexara.engine import NexaraEngine
from inferforge.nexara.forever_loop import ForeverConfig, ForeverRetrainLoop

console = Console(force_terminal=True, stderr=True)


@click.group("forever")
def forever_group():
    """Continuous model improvement with advanced training techniques."""
    pass


@forever_group.command("start")
@click.option("--model", default=None, help="Base model to start from (scratch if None)")
@click.option("--output-dir", default="./nexara_forever", help="Output directory for training")
@click.option("--scale", default="tiny", help="Model scale (tiny, small, medium, large)")
@click.option("--cycles", default=None, type=int, help="Maximum number of training cycles")
@click.option("--cycle-size", default=1024, type=int, help="Training examples per cycle")
@click.option("--from-scratch", is_flag=True, help="Train from scratch instead of fine-tuning")
@click.option("--enable-curriculum", is_flag=True, default=True, help="Enable curriculum learning")
@click.option("--enable-progressive", is_flag=True, default=True, help="Enable progressive resizing")
@click.option("--enable-multi-task", is_flag=True, default=True, help="Enable multi-task learning")
@click.option("--enable-self-supervised", is_flag=True, default=True, help="Enable self-supervised learning")
@click.option("--enable-auto-tuning", is_flag=True, default=True, help="Enable automatic hyperparameter tuning")
@click.option("--enable-meta", is_flag=True, default=False, help="Enable meta-learning")
@click.option("--enable-memory-augmented", is_flag=True, default=False, help="Enable memory-augmented networks")
@click.option("--enable-layerwise-lr", is_flag=True, default=False, help="Enable layerwise learning rates")
@click.option("--enable-gradient-penalty", is_flag=True, default=False, help="Enable gradient penalty")
@click.option("--enable-contrastive", is_flag=True, default=False, help="Enable contrastive learning")
@click.option("--enable-dynamic-batching", is_flag=True, default=True, help="Enable dynamic batching")
@click.option("--enable-flash-attention", is_flag=True, default=True, help="Enable flash attention")
@click.option("--enable-thermal", is_flag=True, default=True, help="Enable thermal management")
@click.option("--enable-power-optimization", is_flag=True, default=True, help="Enable power optimization")
@click.option("--anti-afk", is_flag=True, default=True, help="Prevent PC from sleeping during training")
@click.option("--target-domains", default=None, help="Comma-separated target domains")
@click.option("--quality-threshold", default=0.7, type=float, help="Quality threshold for data sampling")
@click.option("--resume", is_flag=True, default=True, help="Resume from previous training")
@click.option("--dataset", default=None, help="Specific dataset type to generate (or press Enter for auto)")
def start_command(
    model: str | None,
    output_dir: str,
    scale: str,
    cycles: int | None,
    cycle_size: int,
    from_scratch: bool,
    enable_curriculum: bool,
    enable_progressive: bool,
    enable_multi_task: bool,
    enable_self_supervised: bool,
    enable_auto_tuning: bool,
    enable_meta: bool,
    enable_memory_augmented: bool,
    enable_layerwise_lr: bool,
    enable_gradient_penalty: bool,
    enable_contrastive: bool,
    enable_dynamic_batching: bool,
    enable_flash_attention: bool,
    enable_thermal: bool,
    enable_power_optimization: bool,
    anti_afk: bool,
    target_domains: str | None,
    quality_threshold: float,
    resume: bool,
    dataset: str | None
):
    """Start continuous model improvement training with ultra-advanced features."""
    console.print("[bold dark_orange]◈ Nexara Forever Training - Ultra-Advanced Edition[/]")
    console.print("State-of-the-art continuous model improvement system\n")
    
                                   
    if dataset is None:
        console.print("[bold]Dataset Generation Selection[/]")
        console.print("What dataset generation do you want me to generate?")
        console.print("[dim](press Enter to let me generate my own)[/]")
        console.print("Options: general, code, instruction, dialogue, reasoning, technical, creative, or any custom domain")
        console.print()
        
        dataset_choice = input("Your choice: ").strip()
        if not dataset_choice:
            console.print("[dim]Auto-generation selected - I will generate diverse datasets during training[/]")
            dataset_choice = "auto"
    else:
        dataset_choice = dataset
    
    console.print(f"[green]✓[/] Dataset type: {dataset_choice}")
    
    config = ForeverConfig(
        model=model,
        output_dir=output_dir,
        scale=scale,
        cycle_size=cycle_size,
        max_cycles=cycles,
        from_scratch=from_scratch,
        enable_curriculum=enable_curriculum,
        enable_progressive_resizing=enable_progressive,
        enable_multi_task_learning=enable_multi_task,
        enable_self_supervised_learning=enable_self_supervised,
        enable_auto_hyperparameter_tuning=enable_auto_tuning,
        enable_meta_learning=enable_meta,
        enable_memory_augmented=enable_memory_augmented,
        enable_layerwise_learning_rates=enable_layerwise_lr,
        enable_gradient_penalty=enable_gradient_penalty,
        enable_contrastive_learning=enable_contrastive,
        enable_dynamic_batching=enable_dynamic_batching,
        enable_memory_efficient_attention=True,
        enable_flash_attention=enable_flash_attention,
        enable_thermal_management=enable_thermal,
        enable_power_optimization=enable_power_optimization,
        anti_afk=anti_afk,
        target_domains=target_domains.split(",") if target_domains else [dataset_choice] if dataset_choice != "auto" else None,
        quality_threshold=quality_threshold,
        resume=resume,
    )
    
    console.print("[bold]Training Configuration:[/]")
    console.print(f"  Scale: {scale}")
    console.print(f"  Cycle Size: {cycle_size}")
    console.print(f"  Max Cycles: {cycles or 'unlimited'}")
    console.print(f"  From Scratch: {from_scratch}")
    console.print(f"  Resume: {resume}")
    console.print(f"  Dataset Type: {dataset_choice}")
    
    console.print("\n[bold]Core Advanced Features:[/]")
    console.print(f"  Curriculum Learning: {enable_curriculum}")
    console.print(f"  Progressive Resizing: {enable_progressive}")
    console.print(f"  Multi-Task Learning: {enable_multi_task}")
    console.print(f"  Self-Supervised Learning: {enable_self_supervised}")
    console.print(f"  Auto Hyperparameter Tuning: {enable_auto_tuning}")
    
    console.print("\n[bold]Ultra-Advanced Features:[/]")
    console.print(f"  Meta-Learning: {enable_meta}")
    console.print(f"  Memory-Augmented Networks: {enable_memory_augmented}")
    console.print(f"  Layerwise Learning Rates: {enable_layerwise_lr}")
    console.print(f"  Gradient Penalty: {enable_gradient_penalty}")
    console.print(f"  Contrastive Learning: {enable_contrastive}")
    console.print(f"  Dynamic Batching: {enable_dynamic_batching}")
    console.print(f"  Flash Attention: {enable_flash_attention}")
    console.print(f"  Thermal Management: {enable_thermal}")
    console.print(f"  Power Optimization: {enable_power_optimization}")
    
    console.print("\n[bold]System Features:[/]")
    console.print(f"  Anti-AFK Protection: {anti_afk}")
    console.print(f"  PC Optimization: {enable_power_optimization}")
    console.print(f"  Virtual GPU System: {True}")
    console.print(f"  VM Auto-Management: {True}")
    console.print(f"  Quality Threshold: {quality_threshold}")
    
    if target_domains:
        console.print(f"  Target Domains: {target_domains}")
    
    console.print("\n[dim]Starting ultra-advanced continuous training...[/]")
    if anti_afk:
        console.print("[green]✓[/] Enhanced Anti-AFK protection enabled - PC will stay awake and optimized")
        console.print("[green]✓[/] PC optimization enabled - system will feel smooth and responsive")
    console.print("[green]✓[/] Virtual GPU System enabled - VMs will stay active during training")
    console.print("[green]✓[/] VM auto-management enabled - automatic VM creation and management")
    console.print("[dim]Press Ctrl+C to stop gracefully[/]\n")
    
    try:
        engine = NexaraEngine()
        hardware = engine.detect_hardware()
        
        console.print("[bold]Hardware Detected:[/]")
        console.print(f"  CPU Cores: {hardware['cpu_cores']}")
        console.print(f"  RAM: {hardware['ram']} GB")
        console.print(f"  GPU Available: {hardware['gpu_available']}")
        if hardware['gpu_available']:
            console.print(f"  GPU Memory: {hardware['gpu_memory']} GB")
            console.print(f"  GPU Count: {hardware['gpu_count']}")
        
        loop = ForeverRetrainLoop(config, hardware=hardware, engine=engine)
        
        def progress_callback(stage: str, progress: float, info: dict | None = None):
            if stage == "generate":
                console.print(f"[dim]Generating {dataset_choice} data... Cycle {info.get('cycle', '?')}[/]")
            elif stage == "train":
                console.print(f"[dim]Training... {info.get('train_n', '?')} examples[/]")
            elif stage == "complete":
                cycle = info.get('cycle', '?')
                loss = info.get('eval_loss', 'N/A')
                stage_info = info.get('stage', 'N/A')
                console.print(f"[green]✓[/] Cycle {cycle} complete - Loss: {loss} - Stage: {stage_info}")
        
        result = loop.run(progress=progress_callback)
        
        console.print("\n[bold dark_orange]Training Complete[/]")
        console.print(f"Status: {result['status']}")
        console.print(f"Total Cycles: {result['cycles']}")
        console.print(f"Best Loss: {result['best_loss']}")
        console.print(f"Final Model: {result['current_model']}")
        console.print(f"Total Time: {result['seconds']:.1f} seconds")
        
        if result.get('advanced_features'):
            console.print("\n[bold]Advanced Features Used:[/]")
            for feature, enabled in result['advanced_features'].items():
                status = "[green]✓[/]" if enabled else "[dim]✗[/]"
                console.print(f"  {status} {feature}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/]")
    except Exception as e:
        console.print(f"\n[red]Error during training: {e}[/]")
        import traceback
        traceback.print_exc()


@forever_group.command("status")
@click.option("--output-dir", default="./nexara_forever", help="Training output directory")
def status_command(output_dir: str):
    """Check the status of ongoing or completed training."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        console.print(f"[red]No training found at {output_dir}[/]")
        return
    
    console.print("[bold dark_orange]◈ Nexara Training Status[/]\n")
    
                       
    summary_file = output_path / "summary.json"
    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            
            console.print("[bold]Training Summary:[/]")
            console.print(f"  Status: {summary['status']}")
            console.print(f"  Total Cycles: {summary['cycles']}")
            console.print(f"  Best Loss: {summary['best_loss']}")
            console.print(f"  Current Model: {summary['current_model']}")
            console.print(f"  Total Time: {summary['seconds']:.1f} seconds")
            
            if summary.get('advanced_features'):
                console.print("\n[bold]Advanced Features:[/]")
                for feature, enabled in summary['advanced_features'].items():
                    status = "[green]✓[/]" if enabled else "[dim]✗[/]"
                    console.print(f"  {status} {feature}")
            
        except Exception as e:
            console.print(f"[red]Error reading summary: {e}[/]")
    
                             
    cycles = sorted(output_path.glob("cycle_*"))
    if cycles:
        console.print("\n[bold]Recent Cycles:[/]")
        table = Table(show_header=True, header_style="bold dark_orange")
        table.add_column("Cycle", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Loss", style="green")
        table.add_column("Examples", style="white")
        
        for cycle_dir in cycles[-5:]:                      
            metrics_file = cycle_dir / "cycle_metrics.json"
            if metrics_file.exists():
                try:
                    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                    table.add_row(
                        cycle_dir.name,
                        metrics.get("status", "unknown"),
                        f"{metrics.get('eval_loss', 'N/A'):.4f}" if metrics.get('eval_loss') else "N/A",
                        str(metrics.get('trained_on', 'N/A'))
                    )
                except:
                    table.add_row(cycle_dir.name, "error", "N/A", "N/A")
        
        console.print(table)
    
                          
    best_model_file = output_path / "best_model.json"
    if best_model_file.exists():
        try:
            best_model = json.loads(best_model_file.read_text(encoding="utf-8"))
            console.print("\n[bold]Best Model:[/]")
            console.print(f"  Cycle: {best_model.get('cycle', 'N/A')}")
            console.print(f"  Loss: {best_model.get('loss', 'N/A')}")
            console.print(f"  Path: {best_model.get('path', 'N/A')}")
        except Exception as e:
            console.print(f"[red]Error reading best model: {e}[/]")


@forever_group.command("list-recipes")
def list_recipes_command():
    """List available training recipes and scales."""
    from inferforge.nexara.scaling import list_recipes
    
    console.print("[bold dark_orange]◈ Available Training Recipes[/]\n")
    
    recipes = list_recipes()
    
    table = Table(show_header=True, header_style="bold dark_orange")
    table.add_column("Scale", style="cyan")
    table.add_column("Parameters", style="white")
    table.add_column("Recommended VRAM", style="green")
    table.add_column("Use Case", style="white")
    
    for recipe in recipes:
        table.add_row(
            recipe["scale"],
            f"{recipe['parameters']:,}",
            f"{recipe['vram_gb']} GB",
            recipe["use_case"]
        )
    
    console.print(table)
    console.print("\n[dim]Use --scale option to select recipe[/]")