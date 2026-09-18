from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from inferforge.core.config import data_dir
from inferforge.core.registry import ModelRecord, Registry

console = Console()


@click.command("merge")
@click.argument("models", nargs=-1, required=False)
@click.option(
    "--strategy",
    "--method",
    "strategy",
    type=click.Choice([
        "slerp",
        "ties",
        "dare_ties",
        "dare_linear",
        "task_arithmetic",
        "passthrough",
        "moe",
        "linear",
        "simple_average",
    ]),
    default="ties",
    help="Merging strategy to use (ties, dare_ties, dare_linear, task_arithmetic, passthrough, slerp, etc.)",
)
@click.option("--model1", type=str, default=None, help="First model (alternative to positional args).")
@click.option("--model2", type=str, default=None, help="Second model (alternative to positional args).")
@click.option("--output", "output_name", type=str, default=None, help="Output model name (alias for --name).")
@click.option("--interpolation", type=float, default=0.5, help="Interpolation value for SLERP/linear (0.0-1.0)")
@click.option("--weights", "weights_str", type=str, default=None, help="Comma-separated weights (e.g. 0.6,0.4).")
@click.option("--base-model", "base_model", type=str, default=None, help="Base model for delta methods (TIES, DARE, Task Arithmetic).")
@click.option("--ties-k", type=float, default=0.2, help="TIES parameter k: keep top k of significant weights")
@click.option("--dare-drop-rate", type=float, default=0.2, help="DARE parameter: drop rate probability (0.0 - 0.99).")
@click.option("--dare-rescale/--no-dare-rescale", default=True, help="DARE parameter: rescale surviving weights by 1/(1-p).")
@click.option("--slices", type=str, default=None, help="Layer slice mapping for passthrough/frankenmerge.")
@click.option("--recipe", "--config", "recipe_file", type=click.Path(path_type=Path), default=None, help="YAML/JSON merge recipe file.")
@click.option("--resume", is_flag=True, help="Resume an interrupted layer-wise merge from checkpoint.")
@click.option("--quantize", "post_quantize", type=str, default=None, help="Auto-quantize output after merge (q4_k_m, q5_k_m, q8_0).")
@click.option(
    "--precision",
    type=click.Choice(["float32", "float16", "bfloat16"]),
    default="bfloat16",
    help="Target precision for merged model",
)
@click.option("--output-dir", type=click.Path(), default=None, help="Output directory for merged model")
@click.option("--force", is_flag=True, help="Force overwrite existing model")
@click.option("--name", type=str, default=None, help="Final model name (skip interactive prompt)")
@click.option("--enable-procrustes", is_flag=True, help="Enable Procrustes alignment")
@click.option("--enable-fisher", is_flag=True, help="Enable Fisher importance masking")
@click.option("--enable-evaluation", is_flag=True, help="Enable real-time merge evaluation")
@click.option("--enable-svd/--no-enable-svd", default=True, help="Handle dimension mismatches with SVD")
@click.option(
    "--per-layer-coeffs",
    type=str,
    default=None,
    help="Per-layer coefficient pattern (uniform, increasing, decreasing, attention-heavy)",
)
@click.option("--auto-optimize", is_flag=True, help="Enable automatic hyperparameter optimization")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
@click.option("--verify", is_flag=True, help="Load merged tensors and confirm they are finite")
# --- speed / performance flags ------------------------------------------------
@click.option("--use-gpu", "use_gpu", is_flag=True, help="Use CUDA for weight operations when available.")
@click.option("--lazy-load/--no-lazy-load", "lazy_load", default=True, help="Stream weights layer-by-layer to minimize RAM.")
@click.option("--parallel-layers", "parallel_layers", is_flag=True, help="Process independent layers in parallel where possible.")
@click.option("--num-workers", "num_workers", default=0, type=int, help="Worker processes/threads for parallel stage work.")
@click.option("--mmap/--no-mmap", "use_mmap", default=True, help="Memory-map weight files to eliminate RAM copies.")
@click.option("--cache-dir", "cache_dir", type=click.Path(path_type=Path), default=None, help="Directory for merge cache.")
@click.option("--use-cache", "use_cache", is_flag=True, help="Reuse cached validation digests from --cache-dir.")
@click.option("--clear-cache", "clear_cache", is_flag=True, help="Clear the merge cache and exit.")
@click.option("--skip-embeddings", "skip_embeddings", is_flag=True, help="Skip embedding layers during merge.")
@click.option("--skip-normalization", "skip_normalization", is_flag=True, help="Skip normalization weights during merge.")
@click.option("--layer-range", "layer_range", default=None, help="Only merge layers in range, e.g. 0-20.")
@click.option("--show-speed", "show_speed", is_flag=True, help="Print real-time speed metrics (MB/s) per stage.")
@click.option("--dry-run", "dry_run", is_flag=True, help="Validate models and print the merge plan without writing output.")
@click.option("--skip-validation", "skip_validation", is_flag=True, help="Skip expensive pre-merge validation.")
def merge_command(
    models: tuple,
    strategy: str,
    interpolation: float,
    weights_str: Optional[str],
    base_model: Optional[str],
    ties_k: float,
    dare_drop_rate: float,
    dare_rescale: bool,
    slices: Optional[str],
    recipe_file: Optional[Path],
    resume: bool,
    post_quantize: Optional[str],
    precision: str,
    output_dir: Optional[str],
    force: bool,
    name: Optional[str],
    output_name: Optional[str],
    model1: Optional[str],
    model2: Optional[str],
    enable_procrustes: bool,
    enable_fisher: bool,
    enable_evaluation: bool,
    enable_svd: bool,
    per_layer_coeffs: Optional[str],
    auto_optimize: bool,
    verbose: bool,
    verify: bool,
    use_gpu: bool,
    lazy_load: bool,
    parallel_layers: bool,
    num_workers: int,
    use_mmap: bool,
    cache_dir: Optional[Path],
    use_cache: bool,
    clear_cache: bool,
    skip_embeddings: bool,
    skip_normalization: bool,
    layer_range: Optional[str],
    show_speed: bool,
    dry_run: bool,
    skip_validation: bool,
):
    """Merge multiple AI models into one model using real weight operations.

    Example:
        forge merge llama3.1:8b qwen2.5-coder:7b --name fused-coder
        forge merge model1 model2 --strategy slerp --enable-fisher
    """
    if model1 and model2 and not models:
        models = (model1, model2)
    if output_name and not name:
        name = output_name

    # If 3 or more positional arguments given and no explicit output name,
    # check if the last argument was intended as the destination name (e.g. inferforge-beta)
    if len(models) >= 3 and not name:
        cand_name = models[-1]
        reg_check = Registry()
        last_rec = reg_check.get(cand_name)
        first_two_valid = all(reg_check.get(m) is not None for m in models[:-1])
        if first_two_valid and (last_rec is None or not getattr(last_rec, "path", "") or cand_name == "inferforge-beta"):
            name = cand_name
            models = models[:-1]

    if len(models) < 2:
        console.print("[red]Error:[/] at least 2 models are required for merging")
        console.print("[dim]Usage: forge merge <model-a> <model-b> [--name fused][/]")
        raise SystemExit(1)
    if not 0.0 <= interpolation <= 1.0:
        console.print("[red]Error:[/] --interpolation must be between 0.0 and 1.0")
        raise SystemExit(1)
    if not 0.0 < ties_k <= 1.0:
        console.print("[red]Error:[/] --ties-k must be in (0.0, 1.0]")
        raise SystemExit(1)

    console.print("[bold cyan]InferForge Model Merger[/]")
    console.print(f"[dim]Merging {len(models)} models with {strategy.upper()}[/]")

    # --- cache management ---------------------------------------------------
    cache_root = cache_dir or Path(data_dir()) / "merge_cache"
    if clear_cache:
        import shutil as _sh
        if cache_root.exists():
            _sh.rmtree(cache_root)
        console.print(f"[green]OK[/] merge cache cleared ({cache_root})")
        raise SystemExit(0)

    # --- GPU detection --------------------------------------------------------
    device = "cpu"
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
                console.print(f"[green]OK[/] GPU acceleration: [cyan]{torch.cuda.get_device_name(0)}[/]")
            else:
                console.print("[yellow]CUDA not available — merging on CPU.[/]")
        except ImportError:
            console.print("[yellow]torch not installed — merging on CPU.[/]")
    # --- Parse recipe file if provided -------------------------------------
    if recipe_file:
        import json
        p = Path(recipe_file)
        if not p.exists():
            console.print(f"[red]Recipe file not found:[/] {p}")
            raise SystemExit(1)
        try:
            if p.suffix in {".yaml", ".yml"}:
                import yaml
                recipe_data = yaml.safe_load(p.read_text(encoding="utf-8"))
            else:
                recipe_data = json.loads(p.read_text(encoding="utf-8"))
            if "merge_method" in recipe_data:
                strategy = recipe_data["merge_method"].lower()
            if "models" in recipe_data and not models:
                raw_models = recipe_data["models"]
                models = tuple(m["model"] if isinstance(m, dict) else str(m) for m in raw_models)
            if "base_model" in recipe_data and not base_model:
                base_model = recipe_data["base_model"]
        except Exception as exc:
            console.print(f"[yellow]Warning: Could not parse recipe file ({exc}), continuing with CLI arguments.[/]")

    # Parse weights
    weights_list = None
    if weights_str:
        try:
            weights_list = [float(x.strip()) for x in weights_str.split(",")]
        except ValueError:
            console.print("[red]Error:[/] --weights must be comma-separated numbers (e.g. 0.6,0.4)")
            raise SystemExit(1)

    dest_root = Path(output_dir) if output_dir else Path(data_dir()) / "merged_models"

    registry = Registry()
    model_records: list[ModelRecord] = []
    console.print("\n[bold]Step 1: Validating models[/]")
    for model_name in models:
        record = registry.get(model_name)
        if record is None:
            available = ", ".join(registry.names()[:12]) or "(none)"
            console.print(f"[red]Model not found:[/] {model_name}")
            console.print("[dim]Run 'forge list' to see registered models[/]")
            console.print(f"[dim]Known: {available}[/]")
            raise SystemExit(1)
        if not record.path and not record.ollama_name:
            console.print(
                f"[red]Model '{model_name}' has no local weights or Ollama tag.[/]\n"
                "[dim]Import it first: forge import ollama[/]"
            )
            raise SystemExit(1)
        model_records.append(record)
        if verbose:
            console.print(f"  [green]OK[/] {record.name}  backend={record.backend}  format={record.format or 'auto'}")
    console.print(f"[green]OK[/] {len(model_records)} models ready")

    # --- Pre-merge Resource Validation ----------------------------------------
    from inferforge.merger.core.loader import estimate_merge_memory
    mem_info = estimate_merge_memory(model_records, strategy=strategy, layer_wise=lazy_load, output_dir=dest_root)
    console.print("\n[bold]Step 1b: Resource & Memory Estimation[/]")
    console.print(f"  Available RAM:  [cyan]{mem_info['available_ram_gb']} GB[/] (Total: {mem_info['total_ram_gb']} GB)")
    console.print(f"  Streaming RAM:  [green]{mem_info['streaming_ram_gb']} GB[/] (Peak per layer)")
    console.print(f"  Estimated Size: [cyan]{mem_info['estimated_output_gb']} GB[/]")
    console.print(f"  Available Disk: [green]{mem_info['available_disk_gb']} GB[/] on {dest_root.drive or dest_root}")

    if not mem_info["disk_ok"]:
        if force:
            console.print(f"[yellow]Warning: Disk space low ({mem_info['available_disk_gb']} GB free, need ~{mem_info['estimated_output_gb']} GB), proceeding due to --force.[/]")
        else:
            console.print(f"[bold red]Error: Insufficient disk space.[/] Need ~{mem_info['estimated_output_gb']} GB, but only {mem_info['available_disk_gb']} GB free.")
            console.print("[dim]Pass --force to proceed anyway.[/]")
            raise SystemExit(1)
    if not mem_info["can_in_memory"]:
        if not lazy_load:
            console.print("[yellow]Notice:[/] System RAM is insufficient for in-memory merge. Auto-switching to streaming layer-wise mode.")
            lazy_load = True

    # --- cached validation digests ---------------------------------------------
    if use_cache and not skip_validation:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = cache_root / "digests.json"
        digests = {}
        if cache_file.exists():
            try:
                digests = json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                digests = {}
        import hashlib as _hl
        for rec in model_records:
            src = getattr(rec, "path", None)
            if src and Path(src).is_file():
                key = f"{rec.name}:{Path(src).stat().st_size}:{int(Path(src).stat().st_mtime)}"
                if key in digests:
                    console.print(f"[dim]cache hit:[/] {rec.name}")
                else:
                    h = _hl.sha256()
                    with open(src, "rb") as fh:
                        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                            h.update(block)
                    digests[key] = h.hexdigest()
                    cache_file.write_text(json.dumps(digests), encoding="utf-8")
                    console.print(f"[dim]cached digest:[/] {rec.name}")

    if dry_run:
        console.print("\n[bold yellow]Dry run — merge plan (nothing written):[/]")
        console.print(f"  strategy:   {strategy}")
        console.print(f"  precision:  {precision}")
        console.print(f"  weights:    {weights_list or interpolation}")
        console.print(f"  models:     {', '.join(m.name for m in model_records)}")
        console.print(f"  device:     {device}")
        console.print(f"  lazy_load:  {lazy_load}")
        console.print(f"  output dir: {dest_root}")
        total_bytes = sum(Path(m.path).stat().st_size for m in model_records if m.path and Path(m.path).is_file())
        console.print(f"  weight bytes to process: {total_bytes / (1024**3):.2f} GB")
        console.print("[green]OK[/] plan is valid — rerun without --dry-run to merge.")
        raise SystemExit(0)

    try:
        from inferforge.merger.core.weight_blender import MergeConfig, MergeStrategy
    except ImportError as exc:
        console.print(
            f"[red]Merge dependencies missing:[/] {exc}\n"
            "[dim]Install with: pip install 'inferforge[merging]'[/]"
        )
        raise SystemExit(1)

    merge_config = MergeConfig(
        strategy=MergeStrategy(strategy),
        interpolation=interpolation,
        weights=weights_list,
        ties_param_k=ties_k,
        dare_drop_rate=dare_drop_rate,
        dare_rescale=dare_rescale,
        precision=precision,
        normalize=False,
    )
    console.print("\n[bold]Step 2: Merge configuration[/]")
    console.print(f"  strategy={strategy}  precision={precision}  interpolation={interpolation}")
    if weights_list:
        console.print(f"  weights={weights_list}")
    if "dare" in strategy:
        console.print(f"  dare_drop_rate={dare_drop_rate}  dare_rescale={dare_rescale}")
    if "ties" in strategy:
        console.print(f"  ties_k={ties_k}")
    if base_model:
        console.print(f"  base_model={base_model}")

    dest_root.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    dest = dest_root / f"merged_{stamp}"

    console.print("\n[bold]Step 3: Loading and merging real weights[/]")
    merge_started = time.time()
    try:
        merged_path = perform_model_merge(
            model_records,
            merge_config,
            dest,
            verbose=verbose,
            enable_procrustes=enable_procrustes,
            enable_fisher=enable_fisher,
            enable_evaluation=enable_evaluation,
            enable_svd=enable_svd,
            per_layer_coeffs=per_layer_coeffs,
            auto_optimize=auto_optimize,
            device=device,
            lazy_load=lazy_load,
            parallel_layers=parallel_layers,
            num_workers=num_workers,
            use_mmap=use_mmap,
            resume=resume,
            slices=slices,
            base_model=base_model,
            skip_embeddings=skip_embeddings,
            skip_normalization=skip_normalization,
            layer_range=layer_range,
            show_speed=show_speed,
            skip_validation=skip_validation,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]Weight files not found:[/] {exc}")
        console.print("[dim]Pull or import the models so local weights exist, then retry.[/]")
        raise SystemExit(1)
    except TypeError:
        # older pipeline without the extended kwargs — retry with core args only
        merged_path = perform_model_merge(
            model_records,
            merge_config,
            dest,
            verbose=verbose,
            enable_procrustes=enable_procrustes,
            enable_fisher=enable_fisher,
            enable_evaluation=enable_evaluation,
            enable_svd=enable_svd,
            per_layer_coeffs=per_layer_coeffs,
            auto_optimize=auto_optimize,
        )
    except Exception as exc:
        console.print(f"[red]Merge failed:[/] {exc}")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise SystemExit(1)
    elapsed = time.time() - merge_started
    if show_speed:
        total_bytes = sum(Path(m.path).stat().st_size for m in model_records if m.path and Path(m.path).is_file())
        speed = (total_bytes / (1024**2)) / max(0.001, elapsed)
        console.print(f"[cyan]speed:[/] {speed:.1f} MB/s over {elapsed:.1f}s")

    console.print(f"[green]OK[/] Merged weights written to {merged_path}")
    if verify:
        try:
            report = verify_merged_weights(merged_path)
            console.print(
                f"[green]OK[/] verify tensors={report['tensors']} params={report['parameters']} finite=yes"
            )
        except Exception as exc:
            console.print(f"[red]Verify failed:[/] {exc}")
            raise SystemExit(1)

    console.print("\n[bold]Step 4: Name the merged model[/]")
    if name:
        final_name = name.strip()
    else:
        console.print("[yellow]What should the merged model be named?[/]")
        console.print("[dim]Press Enter for merged_model[/]")
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = ""
        final_name = user_input or "merged_model"

    cleaned = final_name.replace("-", "").replace("_", "").replace(":", "").replace(".", "")
    if not cleaned.isalnum():
        console.print("[red]Invalid model name. Use letters, numbers, hyphens, underscores, dots, or colons.[/]")
        raise SystemExit(1)
    existing = registry.get(final_name)
    is_premade_or_beta = bool(existing and (existing.meta.get("premade") or not existing.path or final_name == "inferforge-beta"))
    if existing is not None and not force and not is_premade_or_beta:
        console.print(f"[red]Model '{final_name}' already exists. Pass --force to overwrite.[/]")
        raise SystemExit(1)

    console.print(f"\n[bold]Step 5: Registering '{final_name}'[/]")
    try:
        register_merged_model(final_name, merged_path, model_records, strategy, precision)
    except Exception as exc:
        console.print(f"[red]Failed to register model:[/] {exc}")
        raise SystemExit(1)

    console.print(f"[green]OK[/] '{final_name}' is registered")

    if post_quantize:
        console.print(f"\n[bold]Step 6: Auto-quantizing merged model to {post_quantize}[/]")
        try:
            from inferforge.commands.pull_cmd import _quantize_model
            _quantize_model(merged_path, post_quantize)
        except Exception as q_exc:
            console.print(f"[yellow]Warning: Auto-quantization failed ({q_exc}). The full precision model is still usable.[/]")

    console.print("\n[bold cyan]Merge complete[/]")
    console.print("[dim]Run it with:[/]")
    console.print(f"[bold]  forge run {final_name}[/]")
    console.print(f"[bold]  run {final_name}[/]")


def _is_premium() -> bool:
    try:
        from inferforge.core.premium import get_premium_manager
        return get_premium_manager().get_current_tier().value != "community"
    except Exception:
        return False

def perform_model_merge(
    model_records: List,
    config,
    output_dir: Path,
    verbose: bool = False,
    enable_procrustes: bool = False,
    enable_fisher: bool = False,
    enable_evaluation: bool = False,
    enable_svd: bool = True,
    per_layer_coeffs: Optional[str] = None,
    auto_optimize: bool = False,
    device: str = "cpu",
    lazy_load: bool = True,
    parallel_layers: bool = False,
    num_workers: int = 0,
    use_mmap: bool = True,
    resume: bool = False,
    slices: Optional[str] = None,
    base_model: Optional[str] = None,
    skip_embeddings: bool = False,
    skip_normalization: bool = False,
    layer_range: Optional[str] = None,
    show_speed: bool = False,
    skip_validation: bool = False,
) -> Path:
    last_stage = {"text": "Starting"}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Merging weights...", total=100)

        def on_progress(stage: str, current: int, total: int, extra=None) -> None:
            last_stage["text"] = stage
            ratio = 0 if total <= 0 else min(1.0, current / total)
            progress.update(task, completed=int(ratio * 100), description=f"[cyan]{stage}")
            if verbose and extra:
                console.print(f"[dim]  {stage} {extra}[/]")

        from inferforge.merger.pipeline import MergePipeline

        def _layer_filter(name: str) -> bool:
            lowered = name.lower()
            if skip_embeddings and ("embed" in lowered or "token" in lowered):
                return False
            if skip_normalization and ("norm" in lowered or "ln_" in lowered or "layer_norm" in lowered):
                return False
            if layer_range:
                import re as _re
                m = _re.search(r"layers?\.(\d+)", lowered)
                if m:
                    lo, hi = (int(x) for x in layer_range.split("-"))
                    if not (lo <= int(m.group(1)) <= hi):
                        return False
            return True

        pipeline = MergePipeline(
            config=config,
            enable_procrustes=enable_procrustes,
            enable_fisher=enable_fisher,
            enable_evaluation=enable_evaluation,
            enable_svd=enable_svd,
            per_layer_coeffs=per_layer_coeffs,
            auto_optimize=auto_optimize,
            progress=on_progress,
            device=device,
            lazy_load=lazy_load,
            parallel_layers=parallel_layers,
            num_workers=num_workers,
            use_mmap=use_mmap,
            resume=resume,
            base_model=base_model,
            skip_validation=skip_validation,
        )
        # optional pipeline extensions (ignored gracefully when unsupported)
        for attr, value in (
            ("device", device),
            ("lazy_load", lazy_load),
            ("parallel_layers", parallel_layers),
            ("num_workers", num_workers),
            ("use_mmap", use_mmap),
            ("resume", resume),
            ("base_model", base_model),
            ("skip_validation", skip_validation),
        ):
            try:
                setattr(pipeline, attr, value)
            except Exception:
                pass

        original_run = pipeline.run

        def _filtered_run(records, out_dir):
            result = original_run(records, out_dir)
            return result

        try:
            pipeline.layer_filter = _layer_filter  # type: ignore[attr-defined]
        except Exception:
            pass

        result = _filtered_run(model_records, output_dir)
        progress.update(task, completed=100, description="[cyan]Saved merged model")
    if verbose:
        console.print(f"[dim]Last stage: {last_stage['text']}[/]")
    return result


def verify_merged_weights(model_path: Path) -> dict:
    from inferforge.merger.core.loader import load_model_weights

    weights = load_model_weights(model_path)
    if not weights:
        raise ValueError(f"No tensors found in {model_path}")
    import torch

    total = 0
    for name, tensor in weights.items():
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"Non-finite tensor: {name}")
        total += int(tensor.numel())
        if tensor.ndim >= 2:
            probe = tensor.float()[: min(8, tensor.shape[0]), : min(8, tensor.shape[1])]
            if not torch.isfinite(probe @ probe.T).all():
                raise ValueError(f"Unusable tensor: {name}")
    if total <= 0:
        raise ValueError("Merged model has zero parameters")
    return {"tensors": len(weights), "parameters": total}


def register_merged_model(
    model_name: str,
    model_path: Path,
    source_models: List,
    strategy: str,
    precision: str = "bfloat16",
) -> None:
    registry = Registry()
    base_model = source_models[0]
    weight_file = model_path / "model.safetensors"
    size = weight_file.stat().st_size if weight_file.exists() else 0
    if size == 0 and model_path.exists():
        size = sum(p.stat().st_size for p in model_path.rglob("*") if p.is_file())
    families = [m.family for m in source_models if getattr(m, "family", "")]
    family = families[0] if len(set(families)) == 1 else "merged"
    record = ModelRecord(
        name=model_name,
        digest=f"merged-{abs(hash((model_name, str(model_path)))):016x}",
        source="forge",
        backend="huggingface",
        family=family,
        parameter_size=base_model.parameter_size,
        quantization=precision,
        format="safetensors",
        context_length=max((getattr(m, "context_length", 0) or 0) for m in source_models) or 4096,
        size=int(size),
        path=str(model_path),
        ollama_name="",
        capabilities=sorted({cap for m in source_models for cap in (m.capabilities or [])}),
        imported_at=time.time(),
        meta={
            "merged": True,
            "merge_strategy": strategy,
            "source_models": [m.name for m in source_models],
            "merge_timestamp": int(time.time()),
            "merged_model_path": str(model_path),
            "precision": precision,
        },
    )
    registry.upsert(record)
