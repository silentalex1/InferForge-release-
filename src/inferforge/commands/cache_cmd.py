from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import click
from platformdirs import user_cache_dir
from rich.console import Console
from rich.table import Table

console = Console()


class SmartCache:
    def __init__(self):
        self.cache_dir = Path(user_cache_dir("inferforge")) / "responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> dict:
        if self.index_file.exists():
            return json.loads(self.index_file.read_text())
        return {}
    
    def _save_index(self):
        self.index_file.write_text(json.dumps(self.index, indent=2))
    
    def _hash_key(self, model: str, prompt: str) -> str:
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, model: str, prompt: str) -> str | None:
        key = self._hash_key(model, prompt)
        if key in self.index:
            cache_file = self.cache_dir / f"{key}.txt"
            if cache_file.exists():
                entry = self.index[key]
                entry["hits"] = entry.get("hits", 0) + 1
                entry["last_hit"] = time.time()
                self._save_index()
                return cache_file.read_text()
        return None
    
    def set(self, model: str, prompt: str, response: str):
        key = self._hash_key(model, prompt)
        cache_file = self.cache_dir / f"{key}.txt"
        cache_file.write_text(response)
        
        self.index[key] = {
            "model": model,
            "prompt": prompt[:100],
            "created": time.time(),
            "hits": 0,
            "size": len(response)
        }
        self._save_index()
    
    def stats(self) -> dict:
        total_size = sum(entry["size"] for entry in self.index.values())
        total_hits = sum(entry.get("hits", 0) for entry in self.index.values())
        
        return {
            "entries": len(self.index),
            "total_size": total_size,
            "total_hits": total_hits,
            "cache_dir": str(self.cache_dir)
        }
    
    def clear(self):
        for cache_file in self.cache_dir.glob("*.txt"):
            cache_file.unlink()
        self.index = {}
        self._save_index()


@click.group("cache")
def cache_command():
    """Manage smart response cache."""
    pass


@cache_command.command("stats")
def stats_command():
    """Show cache statistics."""
    cache = SmartCache()
    stats = cache.stats()
    
    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Cached Entries", str(stats["entries"]))
    table.add_row("Total Size", f"{stats['total_size'] / 1024:.1f} KB")
    table.add_row("Total Hits", str(stats["total_hits"]))
    table.add_row("Cache Directory", stats["cache_dir"])
    
    console.print(table)


@cache_command.command("list")
def list_command():
    """List all cached entries."""
    cache = SmartCache()
    
    if not cache.index:
        console.print("[yellow]Cache is empty[/]")
        return
    
    table = Table(title="Cached Responses")
    table.add_column("Model", style="cyan")
    table.add_column("Prompt", style="white")
    table.add_column("Hits", style="green")
    table.add_column("Size", style="yellow")
    
    for key, entry in cache.index.items():
        table.add_row(
            entry["model"],
            entry["prompt"][:50] + "...",
            str(entry.get("hits", 0)),
            f"{entry['size'] / 1024:.1f} KB"
        )
    
    console.print(table)


@cache_command.command("clear")
@click.option("--embeddings", is_flag=True, help="Clear only embedding cache")
@click.option("--responses", is_flag=True, help="Clear only response cache")
@click.option("--kv", is_flag=True, help="Clear only KV cache")
def clear_command(embeddings: bool, responses: bool, kv: bool):
    """Clear cached data."""
    cache = SmartCache()
    
    if not any([embeddings, responses, kv]):
        if click.confirm("Clear all cached responses?"):
            cache.clear()
            console.print("[green]✓[/] Cache cleared")
    else:
        if responses:
            if click.confirm("Clear response cache?"):
                cache.clear()
                console.print("[green]✓[/] Response cache cleared")
        
        if embeddings:
            embed_cache_dir = cache.cache_dir.parent / "embeddings"
            if embed_cache_dir.exists():
                if click.confirm("Clear embedding cache?"):
                    for f in embed_cache_dir.glob("*.bin"):
                        f.unlink()
                    console.print("[green]✓[/] Embedding cache cleared")
        
        if kv:
            kv_cache_dir = cache.cache_dir.parent / "kv_cache"
            if kv_cache_dir.exists():
                if click.confirm("Clear KV cache?"):
                    for f in kv_cache_dir.glob("*.bin"):
                        f.unlink()
                    console.print("[green]✓[/] KV cache cleared")


@cache_command.command("optimize")
def optimize_command():
    """Optimize cache by removing stale entries."""
    cache = SmartCache()
    
    current_time = time.time()
    stale_threshold = 7 * 24 * 3600  # 7 days
    
    stale_keys = []
    for key, entry in cache.index.items():
        if current_time - entry.get("last_hit", entry["created"]) > stale_threshold:
            stale_keys.append(key)
    
    if stale_keys:
        for key in stale_keys:
            cache_file = cache.cache_dir / f"{key}.txt"
            if cache_file.exists():
                cache_file.unlink()
            del cache.index[key]
        
        cache._save_index()
        console.print(f"[green]✓[/] Removed {len(stale_keys)} stale cache entries")
    else:
        console.print("[yellow]No stale entries found[/]")
