from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import click
from rich.console import Console

from inferforge.core.registry import Registry
from inferforge.engine import get_router

console = Console()


class ModelPreloader:
    def __init__(self):
        self.loaded_models: dict[str, any] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def preload_model(self, model_name: str) -> bool:
        try:
            reg = Registry()
            record = reg.get(model_name)
            
            if not record:
                console.print(f"[red]Model '{model_name}' not found[/]")
                return False
            
            router = get_router()
            engine = router.resolve(record)
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                lambda: engine.load_model()
            )
            
            self.loaded_models[model_name] = engine
            console.print(f"[green]✓[/] Preloaded: {model_name}")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to preload {model_name}:[/] {e}")
            return False
    
    async def preload_multiple(self, model_names: list[str]) -> dict[str, bool]:
        tasks = [self.preload_model(name) for name in model_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return dict(zip(model_names, [isinstance(r, bool) and r for r in results]))
    
    def unload_model(self, model_name: str) -> bool:
        if model_name in self.loaded_models:
            try:
                engine = self.loaded_models[model_name]
                engine.unload_model()
                del self.loaded_models[model_name]
                console.print(f"[green]✓[/] Unloaded: {model_name}")
                return True
            except Exception as e:
                console.print(f"[red]Failed to unload {model_name}:[/] {e}")
                return False
        return False
    
    def list_loaded(self) -> list[str]:
        return list(self.loaded_models.keys())
    
    def clear_all(self):
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)


_preloader: ModelPreloader | None = None


def get_preloader() -> ModelPreloader:
    global _preloader
    if _preloader is None:
        _preloader = ModelPreloader()
    return _preloader


@click.group("preload")
def preload_group():
    """Manage model preloading for faster context switching."""
    pass


@preload_group.command("add")
@click.argument("models", nargs=-1, required=True)
@click.option("--parallel", type=int, default=2, help="Number of parallel loads")
def preload_add(models: tuple[str, ...], parallel: int):
    """Preload models into memory for faster access."""
    preloader = get_preloader()
    
    console.print(f"[cyan]Preloading {len(models)} model(s)...[/]")
    
    async def load_models():
        preloader.executor = ThreadPoolExecutor(max_workers=parallel)
        results = await preloader.preload_multiple(list(models))
        
        successful = sum(1 for success in results.values() if success)
        console.print(f"\n[green]Successfully preloaded:[/] {successful}/{len(models)}")
        
        if successful < len(models):
            failed = [name for name, success in results.items() if not success]
            console.print(f"[red]Failed:[/] {', '.join(failed)}")
    
    asyncio.run(load_models())


@preload_group.command("list")
def preload_list():
    """List all currently preloaded models."""
    preloader = get_preloader()
    loaded = preloader.list_loaded()
    
    if not loaded:
        console.print("[yellow]No models preloaded[/]")
        return
    
    console.print(f"[green]Preloaded Models:[/] {len(loaded)}")
    for model in loaded:
        console.print(f"  • {model}")


@preload_group.command("remove")
@click.argument("model")
def preload_remove(model: str):
    """Unload a preloaded model from memory."""
    preloader = get_preloader()
    
    if preloader.unload_model(model):
        console.print(f"[green]✓[/] Removed {model} from memory")
    else:
        console.print(f"[yellow]Model '{model}' not preloaded[/]")


@preload_group.command("clear")
@click.option("--force", is_flag=True, help="Force clear without confirmation")
def preload_clear(force: bool):
    """Clear all preloaded models from memory."""
    preloader = get_preloader()
    loaded = preloader.list_loaded()
    
    if not loaded:
        console.print("[yellow]No models preloaded[/]")
        return
    
    if not force:
        if not click.confirm(f"Clear {len(loaded)} preloaded model(s)?"):
            return
    
    preloader.clear_all()
    console.print(f"[green]✓[/] Cleared {len(loaded)} model(s) from memory")


@preload_group.command("status")
def preload_status():
    """Show memory usage and preloading status."""
    preloader = get_preloader()
    loaded = preloader.list_loaded()
    
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    console.print("\n[bold cyan]Preloading Status[/]")
    console.print(f"[bold]Loaded Models:[/] {len(loaded)}")
    console.print(f"[bold]Memory Usage:[/] {memory_info.rss / (1024**3):.2f} GB")
    console.print(f"[bold]Available Memory:[/] {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    if loaded:
        console.print("\n[bold]Preloaded:[/]")
        for model in loaded:
            console.print(f"  • {model}")


preload_command = preload_group