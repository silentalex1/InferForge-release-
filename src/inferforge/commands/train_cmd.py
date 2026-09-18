from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from inferforge.core.config import get_training_config
from inferforge.core.premium import Tier, get_premium_manager
from inferforge.core.priority_queue import get_priority_queue
from inferforge.core.registry import ModelRecord, Registry
from inferforge.model.identity import INFERFORGE_BETA, resolve_base_model
from inferforge.nexara.auto_scaler import AutoScaler
from inferforge.nexara.engine import NexaraEngine
from inferforge.training.coding_dataset import build_coding_dataset
from inferforge.training.forge_trainer import ForgeTrainer

console = Console(force_terminal=True, stderr=True)


@click.command("train")
@click.argument("model", required=False, default=None)
@click.option("--data", default=None, help="Path to JSON array of {input, output} examples.")
@click.option("--max-examples", default=None, type=int, help="Cap on examples embedded.")
@click.option("--base", default=None, help="Base Ollama model (for new / beta builds).")
@click.option(
    "--curriculum",
    type=click.Choice(["none", "coding"], case_sensitive=False),
    default="coding",
    help="Built-in training curriculum to include.",
)
@click.option("--system", default=None, help="Override system prompt.")
@click.option("--rebuild-beta", is_flag=True, help="Force rebuild InferForge beta with coding curriculum.")
@click.option("--temperature", default=None, type=float, help="Sampling temperature baked into model.")
@click.option("--ctx", default=None, type=int, help="Context length (num_ctx).")
@click.option("--export-dataset", type=click.Path(path_type=Path), default=None, help="Write built-in dataset JSON and exit.")
@click.option("--epochs", default=1, type=int, help="Number of training epochs.")
@click.option("--learning-rate", default=None, type=float, help="Learning rate for training.")
@click.option("--batch-size", default=None, type=int, help="Batch size for training.")
@click.option("--checkpoint-dir", type=click.Path(path_type=Path), default=None, help="Directory for training checkpoints.")
@click.option("--resume", is_flag=True, help="Resume training from checkpoint.")
@click.option("--validation-split", default=0.1, type=float, help="Fraction of data for validation (0.0-1.0).")
@click.option("--lora", is_flag=True, help="Use LoRA fine-tuning for efficiency.")
@click.option("--lora-r", default=8, type=int, help="LoRA rank.")
@click.option("--lora-alpha", default=16, type=int, help="LoRA alpha.")
@click.option("--workers", default=1, type=int, help="Number of parallel workers for data processing.")
@click.option("--validate-data", is_flag=True, help="Validate training data quality before training.")
@click.option("--nexara", type=click.Path(path_type=Path), default=None, help="Path to Nexara code file for AI-native training.")
@click.option("--finetune", is_flag=True, help="Real weight fine-tuning (updates weights) via UniversalTrainer instead of Ollama prompt-baking.")
@click.option("--agent", is_flag=True, help="Agent preset: tool-calling SFT dataset + fine-tune (implies --finetune --lora).")
@click.option("--agent-examples", default=160, type=int, help="Number of synthetic agent traces for --agent mode.")
@click.option("--dpo", is_flag=True, help="Run a DPO alignment stage after SFT (agent prefs auto-generated with --agent, or use preference-format --data).")
@click.option("--scratch", is_flag=True, help="Train a new model from scratch (char-level if no tokenizer).")
@click.option("--scale", default="nano", type=str, help="Model scale for --scratch (nano|micro|tiny|small|base|…|gpt4o).")
@click.option("--seq-len", default=None, type=int, help="Sequence length for weight training.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Directory to write the trained model to.")
@click.option("--auto-scale", is_flag=True, help="Enable automatic scaling for premium users.")
@click.option("--cloud", is_flag=True, help="Use cloud training for premium users.")
@click.option("--hybrid", is_flag=True, help="Use hybrid local+cloud training for premium users.")
@click.option("--priority", is_flag=True, help="Use priority queue for faster training (premium).")
@click.option("--monitoring", is_flag=True, help="Enable advanced monitoring dashboard (premium).")
@click.option("--distributed-nodes", default=None, type=int, help="Number of distributed nodes (premium).")
def train_command(
    model: str | None,
    data: str | None,
    max_examples: int | None,
    base: str | None,
    curriculum: str,
    system: str | None,
    rebuild_beta: bool,
    temperature: float | None,
    ctx: int | None,
    export_dataset: Path | None,
    epochs: int,
    learning_rate: float | None,
    batch_size: int | None,
    checkpoint_dir: Path | None,
    resume: bool,
    validation_split: float,
    lora: bool,
    lora_r: int,
    lora_alpha: int,
    workers: int,
    validate_data: bool,
    nexara: Path | None,
    auto_scale: bool,
    cloud: bool,
    hybrid: bool,
    priority: bool,
    monitoring: bool,
    distributed_nodes: int | None,
    finetune: bool,
    agent: bool,
    agent_examples: int,
    dpo: bool,
    scratch: bool,
    scale: str,
    seq_len: int | None,
    output_dir: Path | None,
) -> None:
    """
    Train / customize a model with advanced fine-tuning options.

    Examples:

      forge train                          # rebuild InferForge beta (coding)

      forge train inferforge-beta --rebuild-beta

      forge train my-model --data examples.json --base qwen2.5-coder:7b

      forge train --export-dataset coding.json

      forge train --epochs 3 --learning-rate 0.0001 --batch-size 4

      forge train --lora --lora-r 16 --lora-alpha 32

      forge train --checkpoint-dir ./checkpoints --resume

      forge train --nexara model.nexara     # Train with Nexara AI-native code

      forge train my-agent --agent --base gpt2          # Agent: tool-calling SFT + synthetic traces
      forge train my-model --finetune --base Qwen/Qwen2.5-0.5B-Instruct --data sft.jsonl --lora
      forge train mini --scratch --scale nano --data corpus.txt   # From-scratch training

      forge train --auto-scale --priority   # Premium: auto-scaling with priority queue

      forge train --cloud --distributed-nodes 4  # Premium: cloud training with 4 nodes

      forge train --hybrid --monitoring     # Premium: hybrid training with advanced monitoring
    """
    # Initialize premium manager
    premium_manager = get_premium_manager()
    current_tier = premium_manager.get_current_tier()
    
    # Check premium feature availability
    if auto_scale and not premium_manager.has_feature("automatic_scaling"):
        console.print("[yellow]Automatic scaling is not available in your current tier.[/]")
        console.print(f"[dim]Current tier: {current_tier.value}[/]")
        console.print("[cyan]Upgrade to Premium for automatic scaling features.[/]")
        auto_scale = False
    
    if cloud and not premium_manager.has_feature("cloud_training"):
        console.print("[yellow]Cloud training is not available in your current tier.[/]")
        console.print(f"[dim]Current tier: {current_tier.value}[/]")
        console.print("[cyan]Upgrade to Premium for cloud training features.[/]")
        cloud = False
    
    if hybrid and not premium_manager.has_feature("hybrid_training"):
        console.print("[yellow]Hybrid training is not available in your current tier.[/]")
        console.print(f"[dim]Current tier: {current_tier.value}[/]")
        console.print("[cyan]Upgrade to Premium for hybrid training features.[/]")
        hybrid = False
    
    if priority and not premium_manager.has_feature("priority_support"):
        console.print("[yellow]Priority queue is not available in your current tier.[/]")
        console.print(f"[dim]Current tier: {current_tier.value}[/]")
        console.print("[cyan]Upgrade to Premium for priority queue features.[/]")
        priority = False
    
    if monitoring and not premium_manager.has_feature("advanced_monitoring"):
        console.print("[yellow]Advanced monitoring is not available in your current tier.[/]")
        console.print(f"[dim]Current tier: {current_tier.value}[/]")
        console.print("[cyan]Upgrade to Premium for advanced monitoring features.[/]")
        monitoring = False
    
    if distributed_nodes and not premium_manager.has_feature("distributed_training"):
        console.print("[yellow]Distributed training is not available in your current tier.[/]")
        console.print(f"[dim]Current tier: {current_tier.value}[/]")
        console.print("[cyan]Upgrade to Premium for distributed training features.[/]")
        distributed_nodes = None
    
    # Display tier information
    if current_tier != Tier.COMMUNITY:
        console.print(f"[bold cyan]◈ {current_tier.value.title()} Tier Active[/]")
        limits = premium_manager.get_limits()
        console.print(f"[dim]  Max concurrent trainings: {limits.max_concurrent_trainings}[/]")
        console.print(f"[dim]  Max GPU nodes: {limits.max_gpu_nodes}[/]")
        console.print(f"[dim]  Priority multiplier: {limits.priority_queue_multiplier}x[/]")
        console.print()
    
    # Setup priority queue if requested
    queue = None
    job_id = None
    if priority:
        queue = get_priority_queue()
        estimated_duration = 3600.0  # 1 hour default estimate
        try:
            job_id, estimated_wait = queue.submit_job(
                user_id="local_user",
                model_name=model or "inferforge-beta",
                estimated_duration=estimated_duration,
                metadata={"tier": current_tier.value}
            )
            console.print(f"[green]✓[/] Job queued with ID: {job_id}")
            console.print(f"[dim]Estimated wait time: {estimated_wait:.1f}s[/]")
            console.print()
        except ValueError as e:
            console.print(f"[red]Cannot queue job:[/] {e}")
            raise SystemExit(1) from e
    
    # Setup advanced monitoring if requested
    dashboard = None
    if monitoring:
        try:
            from inferforge.monitoring.advanced_dashboard import get_advanced_dashboard
            dashboard = get_advanced_dashboard()
            console.print("[green]✓[/] Advanced monitoring enabled")
            console.print()
        except ImportError:
            console.print("[yellow]Advanced monitoring dependencies not available[/]")
            monitoring = False
    
    # Setup auto-scaler if requested
    auto_scaler = None
    if auto_scale:
        auto_scaler = AutoScaler()
        console.print("[green]✓[/] Auto-scaling enabled")
        console.print()
    
    if nexara:
        if not nexara.exists():
            console.print(f"[red]Nexara file not found:[/] {nexara}")
            raise SystemExit(1)
        
        nexara_code = nexara.read_text(encoding="utf-8")
        engine = NexaraEngine()
        output_dir = Path.cwd() / "nexara_output"
        
        console.print("[bold dark_orange]◈ Nexara[/] AI-native compilation\n")
        result = engine.compile_and_train(nexara_code, output_dir)
        
        console.print(f"[green]✓[/] Hardware detected: {result['hardware']['cpu_cores']} cores, {result['hardware']['ram']}GB RAM")
        if result['hardware']['gpu_available']:
            console.print(f"[green]✓[/] GPU detected: {result['hardware']['gpu_memory']}GB VRAM")
        
        console.print(f"[green]✓[/] Compiled {len(result['compiled']['models'])} model(s)")
        for model_name in result['compiled']['models'].keys():
            console.print(f"  - {model_name}")
        
        engine.generate_training_script(result['compiled'], output_dir)
        console.print(f"[green]✓[/] Training script generated: {output_dir / 'train_nexara.py'}")
        console.print("\n[dim]Run training script to start training with Nexara optimizations.[/]")
        return

    if agent:
        finetune = True
        lora = True

    if finetune or agent or scratch:
        _run_weight_training(
            target=model, data=data, base=base, system=system,
            epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
            lora=lora, lora_r=lora_r, lora_alpha=lora_alpha,
            agent=agent, agent_examples=agent_examples, dpo=dpo,
            scratch=scratch, scale=scale, seq_len=seq_len,
            output_dir=output_dir, resume=resume, checkpoint_dir=checkpoint_dir,
            validation_split=validation_split, validate_data=validate_data,
            curriculum=curriculum,
        )
        return

    if export_dataset:
        payload = build_coding_dataset()
        export_dataset.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]✓[/] wrote {len(payload)} examples → [cyan]{export_dataset}[/]")
        return

    config = get_training_config()
    if not config["enabled"]:
        console.print("[red]Training is disabled in settings[/]")
        raise SystemExit(1)

    # Check training limits for premium users
    can_train, limit_message = premium_manager.check_training_limit(estimated_hours=1.0)
    if not can_train:
        console.print(f"[red]{limit_message}[/]")
        console.print("[cyan]Upgrade your tier for higher training limits.[/]")
        raise SystemExit(1)
    
    # Record training start
    premium_manager.start_training(estimated_hours=1.0)

    # Default target: InferForge beta
    target = (model or INFERFORGE_BETA).strip()
    if rebuild_beta or target in {INFERFORGE_BETA, "beta", "inferforge"}:
        target = INFERFORGE_BETA

    training_data: list[dict] = []
    if data:
        data_path = Path(data)
        if not data_path.exists():
            console.print(f"[red]Training data not found:[/] {data}")
            raise SystemExit(1)
        with data_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            console.print("[red]Training data must be a JSON array of {input, output} objects[/]")
            raise SystemExit(1)
        training_data = loaded

    # Apply premium tier limits
    limits = premium_manager.get_limits()
    cap = max_examples or int(config.get("max_examples") or 64)
    if target == INFERFORGE_BETA and max_examples is None:
        cap = max(cap, 64)
    
    # Enforce tier-specific limits
    cap = min(cap, limits.max_dataset_size_examples)

    params: dict = {}
    if temperature is not None:
        params["temperature"] = temperature
    if ctx is not None:
        params["num_ctx"] = ctx
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    if batch_size is not None:
        params["batch_size"] = batch_size
    if epochs is not None:
        params["epochs"] = epochs
    if lora:
        params["lora"] = True
        params["lora_r"] = lora_r
        params["lora_alpha"] = lora_alpha
    
    checkpoint_path = checkpoint_dir or Path.cwd() / "checkpoints" / target.replace(":", "-")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    use_coding = curriculum.lower() == "coding"
    base_model = resolve_base_model(base)

    console.print(
        f"[bold dark_orange]◈ InferForge[/] training [bold cyan]{target}[/]\n"
        f"  base: [cyan]{base_model}[/]  ·  curriculum: [cyan]{curriculum}[/]  ·  cap: {cap}\n"
        f"  epochs: {epochs}  ·  workers: {workers}"
    )
    if training_data:
        console.print(f"  extra examples: {len(training_data)}")
    if lora:
        console.print(f"  LoRA: rank={lora_r}, alpha={lora_alpha}")
    if validation_split > 0:
        console.print(f"  validation split: {validation_split:.1%}")
    if checkpoint_dir:
        console.print(f"  checkpoints: {checkpoint_path}")
    console.print()
    
    if validate_data and training_data:
        _validate_training_data(training_data, console)

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
            task = progress.add_task("training…", total=100)

            def on_prog(status: str, frac: float, eta: float | None = None) -> None:
                desc = f"[cyan]{status or 'training'}[/]"
                if eta:
                    desc += f" [dim]ETA: {eta:.0f}s[/]"
                progress.update(
                    task,
                    completed=max(1, min(99, int(frac * 100))),
                    description=desc,
                )

            if target == INFERFORGE_BETA:
                result = trainer.build_inferforge_beta(
                    base_model=base_model,
                    force=True,
                    progress=on_prog,
                    extra_data=training_data or None,
                    max_examples=cap,
                    checkpoint_dir=checkpoint_path if not resume else None,
                    resume_from=checkpoint_path if resume else None,
                    validation_split=validation_split,
                    workers=workers,
                )
                from inferforge.model.identity import register_inferforge_beta

                register_inferforge_beta(
                    base_model,
                    extra_meta={
                        "trained": True,
                        "examples_embedded": result.get("examples_embedded"),
                        "epochs": epochs,
                        "lora": lora,
                        "checkpoint_path": str(checkpoint_path),
                    },
                )
            else:
                reg = Registry()
                existing = reg.get(target)
                if existing and not training_data and not use_coding:
                    console.print(
                        "[yellow]No --data and curriculum=none — nothing to train. "
                        "Pass --data or --curriculum coding.[/]"
                    )
                    raise SystemExit(2)

                train_base = base or (existing.meta.get("base_model") if existing else None) or base_model
                result = trainer.train_model(
                    target,
                    train_base,
                    training_data=training_data or None,
                    system=system,
                    max_examples=cap,
                    params=params or None,
                    progress=on_prog,
                    use_builtin_coding=use_coding,
                    checkpoint_dir=checkpoint_path if not resume else None,
                    resume_from=checkpoint_path if resume else None,
                    validation_split=validation_split,
                    workers=workers,
                )
            progress.update(task, completed=100, description="done")
    except Exception as exc:
        console.print(f"[bold red]Training failed:[/] {exc}")
        raise SystemExit(1) from exc

    table = Table(title="Model created (prompt-based — weights unchanged)", show_header=True, header_style="bold")
    table.add_column("field", style="dim")
    table.add_column("value", style="cyan")
    table.add_row("model", target)
    table.add_row("method", "Ollama Modelfile (system prompt + few-shot)")
    table.add_row("examples", str(result.get("examples_embedded", "—")))
    table.add_row("epochs", str(epochs))
    table.add_row("status", str(result.get("status", "completed")))
    if lora:
        table.add_row("method", "LoRA")
    if result.get("validation_loss"):
        table.add_row("validation_loss", f"{result['validation_loss']:.4f}")
    if result.get("training_time"):
        table.add_row("training_time", f"{result['training_time']:.1f}s")
    if result.get("path"):
        table.add_row("path", str(result["path"]))
    console.print()
    console.print(table)
    console.print()
    console.print(f"[green]✓[/] Run with: [bold]forge chat[/]  or  [bold]forge run {target}[/]")
    console.print("[dim]Note: this mode embeds prompts and examples but does not update weights. "
                  "Use [bold]--finetune[/], [bold]--agent[/], or [bold]--scratch[/] for real weight training.[/]")
    
    # Record training completion with premium manager
    training_time = result.get("training_time", 0) / 3600.0  # Convert to hours
    premium_manager.end_training(training_time)
    
    # Complete priority queue job
    if queue and job_id:
        queue.complete_job(job_id, success=True)
        console.print(f"[green]✓[/] Priority queue job {job_id} completed")
    
    # Export monitoring data if enabled
    if dashboard and monitoring:
        try:
            export_path = Path.cwd() / "training_metrics.json"
            dashboard.export_metrics(export_path)
            console.print(f"[green]✓[/] Training metrics exported to {export_path}")
            
            # Generate performance report
            report = dashboard.get_performance_report()
            console.print("\n[bold]Performance Report:[/]")
            console.print(f"  Total training time: {report['training_summary']['total_time_hours']:.2f} hours")
            console.print(f"  Loss improvement: {report['training_summary']['loss_improvement_percent']:.1f}%")
            console.print(f"  Average GPU utilization: {report['resource_efficiency']['avg_gpu_utilization']:.1f}%")
            
            if report['recommendations']:
                console.print("\n[bold]Recommendations:[/]")
                for rec in report['recommendations'][:3]:
                    console.print(f"  • {rec}")
        except Exception as e:
            console.print(f"[yellow]Could not export metrics: {e}[/]")
    
    # Display scaling results if auto-scaling was used
    if auto_scale and auto_scaler:
        scaling_report = auto_scaler.get_scaling_report()
        if scaling_report.get("scaling_metrics", {}).get("decisions_made", 0) > 0:
            console.print("\n[bold]Auto-scaling Results:[/]")
            console.print(f"  Scaling decisions made: {scaling_report['scaling_metrics']['decisions_made']}")
            console.print(f"  Cost savings: ${scaling_report['scaling_metrics']['total_cost_savings']:.2f}")
            console.print(f"  Time savings: {scaling_report['scaling_metrics']['total_time_savings']:.1f}s")
            console.print(f"  Final nodes: {scaling_report['scaling_metrics']['final_nodes']}")
    
    sys.stdout.flush()


def _run_weight_training(
    *,
    target: str | None,
    data: str | None,
    base: str | None,
    system: str | None,
    epochs: int,
    learning_rate: float | None,
    batch_size: int | None,
    lora: bool,
    lora_r: int,
    lora_alpha: int,
    agent: bool,
    agent_examples: int,
    dpo: bool,
    scratch: bool,
    scale: str,
    seq_len: int | None,
    output_dir: Path | None,
    resume: bool,
    checkpoint_dir: Path | None,
    validation_split: float,
    validate_data: bool,
    curriculum: str,
) -> None:
    """Real gradient-based training via the UniversalTrainer (updates weights)."""
    try:
        from inferforge.nexara.universal_trainer import UniversalTrainConfig, UniversalTrainer
        import torch  # noqa: F401
    except ImportError as exc:
        console.print(f"[red]Weight training requires PyTorch + transformers:[/] {exc}")
        console.print("[dim]Install with: pip install torch transformers peft accelerate[/]")
        console.print("[dim]Or drop --finetune/--agent/--scratch to use Ollama prompt-baking.[/]")
        raise SystemExit(1) from exc

    from inferforge.core.config import trained_models_dir
    from inferforge.nexara.universal_trainer import load_raw_records

    name = (target or ("forge-agent" if agent else ("forge-scratch" if scratch else "forge-model"))).replace(":", "-")
    out_dir = output_dir or (trained_models_dir() / name)

    # --- assemble records ---
    records: list = []
    if agent:
        from inferforge.training.agent_dataset import build_agent_dataset, validate_agent_records
        records.extend(build_agent_dataset(agent_examples))
        console.print(f"[green]✓[/] {agent_examples} synthetic agent tool-call traces")
    if data:
        data_path = Path(data)
        if not data_path.exists():
            console.print(f"[red]Training data not found:[/] {data}")
            raise SystemExit(1)
        records.extend(load_raw_records(str(data_path)))
    if not records:
        if scratch:
            console.print("[yellow]No data — scratch run will use placeholder corpus.[/]")
        elif curriculum.lower() == "coding":
            records.extend(build_coding_dataset())
            console.print(f"[green]✓[/] built-in coding dataset ({len(records)} examples)")
        else:
            console.print("[red]No training data.[/] Pass --data, --agent, or --curriculum coding.")
            raise SystemExit(2)

    if agent:
        from inferforge.training.agent_dataset import validate_agent_records
        records, issues = validate_agent_records(records)
        if issues:
            console.print(f"[yellow]{len(issues)} invalid agent records skipped[/] (first: {issues[0]})")
    elif validate_data and data and Path(data).suffix.lower() == ".json":
        flat = [r for r in records if isinstance(r, dict) and "messages" not in r]
        if flat:
            _validate_training_data(flat, console)

    eval_records = None
    if validation_split > 0 and len(records) > 10:
        cut = int(len(records) * (1 - validation_split))
        eval_records = records[cut:]
        records = records[:cut]

    base_model = None
    if not scratch:
        base_model = base
        if not base_model:
            reg = Registry()
            rec = reg.get(target) if target else None
            base_model = (rec.meta.get("hf_model_id") if rec else None) or (rec.meta.get("base_model") if rec else None) or "gpt2"
            console.print(f"[dim]No --base given; using {base_model}[/]")

    cfg = UniversalTrainConfig(
        output_dir=str(out_dir),
        model=base_model,
        from_scratch=scratch,
        scale=scale,
        stage="auto",
        data=records,
        eval_data=eval_records,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seq_len=seq_len,
        peft_method="lora" if lora else "auto",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        resume=str(checkpoint_dir) if resume and checkpoint_dir else None,
    )

    console.print(
        f"[bold dark_orange]◈ InferForge[/] weight-training [bold cyan]{name}[/]\n"
        f"  mode: {'scratch' if scratch else 'fine-tune'}  ·  base: {base_model or 'none'}"
        f"  ·  records: {len(records)}  ·  epochs: {epochs}"
        + (f"  ·  LoRA r={lora_r} a={lora_alpha}" if lora else "")
    )

    try:
        trainer = UniversalTrainer(cfg)
        resolved = trainer.resolve()
        console.print(
            f"  resolved: stage={resolved['stage']} peft={resolved['peft']} "
            f"seq_len={resolved['seq_len']} batch={resolved['batch_size']}x{resolved['gradient_accumulation_steps']} "
            f"lr={resolved['learning_rate']:.2e} device={resolved['device']}"
        )
        with Progress(
            SpinnerColumn(style="dark_orange"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("training…", total=100)

            def on_prog(stage_name: str, frac: float, row: dict | None = None) -> None:
                desc = f"[cyan]{stage_name}[/] loss={row['loss']:.3f} lr={row['lr']:.1e}" if row else f"[cyan]{stage_name}[/]"
                progress.update(task, completed=max(1, min(99, int(frac * 100))), description=desc)

            result = trainer.train(progress=on_prog)
            progress.update(task, completed=100, description="done")
    except Exception as exc:
        console.print(f"[bold red]Training failed:[/] {exc}")
        raise SystemExit(1) from exc

    if dpo:
        # Alignment stage on preference pairs — auto-generate agent prefs or filter
        # preference-schema records from user data.
        from inferforge.training.agent_dataset import build_agent_preference_dataset
        from inferforge.nexara.universal_trainer import detect_record_schema

        prefs = [r for r in (eval_records or []) + records
                 if isinstance(r, dict) and detect_record_schema(r) == "preference"]
        if agent or not prefs:
            prefs = build_agent_preference_dataset(max(40, agent_examples // 2)) + prefs
        console.print(f"[bold dark_orange]◈[/] DPO alignment stage on {len(prefs)} preference pairs")

        dpo_cfg = UniversalTrainConfig(
            output_dir=str(out_dir / "dpo"),
            model=str(Path(result["output_dir"])),
            stage="dpo",
            data=prefs,
            epochs=1,
            learning_rate=learning_rate or 5e-5,
            batch_size=batch_size,
            seq_len=seq_len,
            peft_method="lora" if lora else "auto",
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        try:
            dpo_result = UniversalTrainer(dpo_cfg).train()
            result["dpo"] = {k: dpo_result.get(k) for k in ("status", "steps", "best_loss")}
            result["output_dir"] = dpo_result["output_dir"]
        except Exception as exc:
            console.print(f"[yellow]DPO stage failed (SFT model kept):[/] {exc}")

    # Register the merged model so `forge run <name>` / `forge chat` can load it.
    final_dir = Path(result["output_dir"])
    reg = Registry()
    size = sum(p.stat().st_size for p in final_dir.rglob("*") if p.is_file()) if final_dir.exists() else 0
    reg.upsert(ModelRecord(
        name=name,
        source="forge",
        backend="huggingface",
        size=size,
        format="safetensors",
        family=base_model or scale,
        parameter_size=scale,
        path=str(final_dir),
        meta={
            "hf_model_id": str(final_dir),
            "base_model": base_model,
            "trained": True,
            "agentic": agent,
            "peft": resolved.get("peft"),
            "completions_only": result.get("completions_only"),
        },
    ))

    table = Table(title="Weight training complete", show_header=True, header_style="bold")
    table.add_column("field", style="dim")
    table.add_column("value", style="cyan")
    for key in ("status", "steps", "micro_steps", "samples_seen", "tokens_seen"):
        if result.get(key) is not None:
            table.add_row(key, str(result[key]))
    table.add_row("eval_loss", f"{result['eval']['eval_loss']:.4f}")
    table.add_row("best_loss", f"{result['best_loss']:.4f}")
    if result.get("dpo"):
        table.add_row("dpo", f"{result['dpo']['status']} ({result['dpo'].get('steps', 0)} steps)")
    table.add_row("seconds", f"{result['seconds']:.1f}")
    table.add_row("output", str(final_dir))
    console.print()
    console.print(table)
    console.print()
    console.print(f"[green]✓[/] Run with: [bold]forge run {name}[/]")
    sys.stdout.flush()


def _validate_training_data(data: list[dict], console: Console) -> None:
    """Validate training data quality."""
    console.print("[dim]Validating training data...[/]")
    
    issues = []
    total = len(data)
    
    for i, example in enumerate(data):
        if not isinstance(example, dict):
            issues.append(f"Example {i}: not a dict")
            continue
        
        if "input" not in example:
            issues.append(f"Example {i}: missing 'input' field")
        elif not isinstance(example["input"], str):
            issues.append(f"Example {i}: 'input' not a string")
        elif len(example["input"]) < 3:
            issues.append(f"Example {i}: 'input' too short")
        
        if "output" not in example:
            issues.append(f"Example {i}: missing 'output' field")
        elif not isinstance(example["output"], str):
            issues.append(f"Example {i}: 'output' not a string")
        elif len(example["output"]) < 3:
            issues.append(f"Example {i}: 'output' too short")
    
    if issues:
        console.print(f"[yellow]Found {len(issues)} validation issues:[/]")
        for issue in issues[:10]:
            console.print(f"  [dim]- {issue}[/]")
        if len(issues) > 10:
            console.print(f"  [dim]... and {len(issues) - 10} more[/]")
    else:
        console.print(f"[green]✓[/] All {total} examples validated successfully")
