from __future__ import annotations

import atexit
import signal
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from inferforge.nexara.engine import NexaraEngine
from inferforge.nexara.forever_factory import CYCLE_SIZE
from inferforge.nexara.forever_loop import ForeverConfig, ForeverRetrainLoop
from inferforge.nexara.safe_io import StopFlag, cleanup_partials

console = Console(force_terminal=True, stderr=True)
_STOP = StopFlag()
_OUTPUT: Path | None = None


def _cleanup_partials() -> None:
    if _OUTPUT is not None:
        cleanup_partials(_OUTPUT)


def _request_stop(*_args) -> None:
    if _STOP.requested():
        _STOP.request(force=True)
        console.print("\n[yellow]Second stop — leaving completed datasets, dropping only .partial/.tmp files.[/]")
        _cleanup_partials()
        raise SystemExit(0)
    _STOP.request()
    console.print("\n[yellow]Stop requested — finishing the current safe boundary (shard/step), then exit.[/]")
    console.print("[dim]Completed cycles, shards, and checkpoints are kept. Nothing outside the run folder is touched.[/]")


@click.group("re-train")
def retrain_group():
    """Generate fresh datasets and continuously retrain the model."""


@retrain_group.command("forever")
@click.option("--model", default=None, help="Base model path or id. Omit to train from scratch.")
@click.option("--output", "-o", default="./nexara_forever", help="Isolated run directory (never writes outside it).")
@click.option("--scale", default="tiny", help="nano|micro|tiny|small|base|medium|large|…|gpt4o")
@click.option("--cycle-size", default=CYCLE_SIZE, type=int, help="Examples generated each cycle (default 250000).")
@click.option("--max-cycles", default=None, type=int, help="Stop after N cycles. Default: run forever.")
@click.option("--train-samples", default=None, type=int, help="Cap examples actually trained per cycle.")
@click.option("--peft", default="auto", help="auto|none|lora|dora|qlora")
@click.option("--epochs", default=1, type=int)
@click.option("--max-steps", default=None, type=int)
@click.option("--from-scratch", is_flag=True, help="Start from random weights.")
@click.option("--stop-on-plateau", is_flag=True, help="Stop if eval loss stops improving.")
@click.option("--no-nexara", is_flag=True, help="Skip Nexara compile path (still trains).")
@click.option("--no-resume", is_flag=True, help="Do not resume a previous run in --output.")
@click.option("--seed", default=7, type=int)
def forever_command(
    model: str | None,
    output: str,
    scale: str,
    cycle_size: int,
    max_cycles: int | None,
    train_samples: int | None,
    peft: str,
    epochs: int,
    max_steps: int | None,
    from_scratch: bool,
    stop_on_plateau: bool,
    no_nexara: bool,
    no_resume: bool,
    seed: int,
) -> None:
    """Keep generating 250K datasets and fine-tuning via Nexara, cycle after cycle."""
    global _OUTPUT
    _STOP._stop = False
    _STOP._force = False
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _OUTPUT = out
    atexit.register(_cleanup_partials)
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None), getattr(signal, "SIGBREAK", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _request_stop)
        except Exception:
            pass

    engine = NexaraEngine()
    hardware = engine.detect_hardware()
    cfg = ForeverConfig(
        model=model,
        output_dir=str(out),
        scale=scale,
        cycle_size=cycle_size,
        max_cycles=max_cycles,
        peft_method=peft,
        epochs_per_cycle=epochs,
        max_steps=max_steps,
        train_samples=train_samples,
        seed=seed,
        from_scratch=from_scratch or not model,
        stop_on_plateau=stop_on_plateau,
        use_nexara=not no_nexara,
        resume=not no_resume,
    )
    loop = ForeverRetrainLoop(cfg, hardware=hardware, stop=_STOP, engine=engine)

    console.print("[bold dark_orange]◈ forge re-train forever[/]")
    console.print("  250K-class datasets → Nexara compile + train/align → repeat")
    console.print(f"  scale={scale}  peft={peft}  model={model or 'scratch'}  out={out}")
    console.print("[dim]  Ctrl+C / process stop is safe: completed shards and models are kept; only .partial files are dropped.[/]\n")

    def on_progress(phase: str, frac: float, info: dict | None) -> None:
        info = info or {}
        if phase == "generate":
            console.print(f"[dim]cycle {info.get('cycle', 0)}[/] synthesizing {info.get('target', cycle_size):,} examples…")
        elif phase == "train":
            console.print(
                f"[dim]cycle {info.get('cycle', 0)}[/] Nexara training on {info.get('train_n', 0):,} "
                f"(generated {info.get('generated', 0):,})"
            )
        elif phase == "complete":
            loss = info.get("eval_loss")
            loss_s = f"{loss:.4f}" if isinstance(loss, (int, float)) else "—"
            tag = "nexara" if info.get("nexara") else "direct"
            console.print(
                f"[green]✓[/] cycle {info.get('cycle')} [{tag}/{info.get('stage', 'sft')}]  "
                f"gen={info.get('generated'):,}  train={info.get('trained_on'):,}  loss={loss_s}"
            )

    result = loop.run(progress=on_progress, should_stop=lambda: _STOP.requested())
    _cleanup_partials()
    table = Table(title="re-train forever")
    table.add_column("field", style="dim")
    table.add_column("value", style="cyan")
    table.add_row("status", str(result.get("status")))
    table.add_row("cycles", str(result.get("cycles")))
    table.add_row("best_loss", str(result.get("best_loss")))
    table.add_row("model", str(result.get("current_model")))
    table.add_row("seconds", f"{result.get('seconds', 0):.1f}")
    table.add_row("output", str(out))
    table.add_row("datasets_preserved", "yes")
    console.print()
    console.print(table)
    console.print("[dim]Resume later with the same --output. History: history.jsonl[/]")
