from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import click
import httpx
from rich.console import Console

console = Console()
UPDATE_API = "https://hyperneural.cfd/api/updates/latest"
ADMIN_UPDATE_API = "https://hyperneural.cfd/admin-panel/api/updates"


def _disk_state(path: Path) -> tuple[str, int]:
    if path.is_file():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest(), path.stat().st_size
    if path.is_dir():
        entries = sorted(
            (str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in path.rglob("*") if p.is_file()
        )
        size = sum(e[1] for e in entries)
        digest = "sha256:" + hashlib.sha256(json.dumps(entries).encode()).hexdigest()
        return digest, size
    return "", 0


def _update_model(model: str) -> None:
    from inferforge.commands.embedd_cmd import _embed_sdk, _resolve_upstream, _sdk_owner
    from inferforge.core.config import load_settings
    from inferforge.core.registry import Registry
    from inferforge.embedded.publish import (
        HOSTED_MODEL,
        default_site,
        persona_from_ollama,
        publish_sdk,
        sdk_slug,
        sdk_url,
    )

    reg = Registry()
    record = reg.get(model)
    if not record:
        console.print(f"[red]Model not found:[/] {model}")
        console.print("Run [bold]forge list[/] to see registered models.")
        raise SystemExit(1)

    settings = load_settings()
    slug = sdk_slug(model)
    embed_key = (settings.get("embed_keys") or {}).get(model)
    publish_token = (settings.get("publish_tokens") or {}).get(slug)

    if not embed_key or not publish_token:
        console.print(f"[yellow]{model} has not been published yet. Publishing it now.[/]")
        _embed_sdk(model, None, None, None, True, False)
        return

    old_digest = record.digest
    weights = Path(record.path) if record.path else None
    if weights and weights.exists():
        record.digest, record.size = _disk_state(weights)
        record.imported_at = time.time()
        reg.upsert(record)
        if record.digest != old_digest:
            console.print(f"[green]OK[/] Weights changed on disk, now {record.display_size()}")
        else:
            console.print(f"[dim]Weights unchanged ({record.display_size()})[/]")
    else:
        console.print("[dim]No local weight path to refresh, updating the published copy only[/]")

    site_url = (settings.get("sdk_site") or default_site()).rstrip("/")
    upstream = _resolve_upstream(None, settings)
    persona = persona_from_ollama(record.ollama_name or model)

    with console.status(f"[cyan]Updating {model} on {site_url}...[/]"):
        result = publish_sdk(
            site=site_url,
            model=model,
            slug=slug,
            embed_key=embed_key,
            publish_token=publish_token,
            endpoint=upstream,
            fallback=upstream,
            owner=_sdk_owner(),
            system=persona,
            hosted_model=HOSTED_MODEL,
        )

    if not result.get("ok"):
        console.print(f"[red]Update failed:[/] {result.get('error')}")
        raise SystemExit(1)

    console.print(f"[green]OK[/] {model} updated on {site_url}.")
    if persona:
        console.print(f"  Persona refreshed: {len(persona)} chars from the Modelfile.")
    console.print(f"  Embed URL:  [cyan]{sdk_url(site_url, slug)}[/]  [dim](unchanged)[/]")
    console.print(f"  Embed key:  [cyan]{embed_key}[/]  [dim](unchanged)[/]")
    console.print("  [dim]Pages using this model pick up the update on their next request.[/]")


@click.command("update")
@click.argument("model", required=False, metavar="[MODEL]")
@click.option("--new", "is_new", is_flag=True, help="Install the latest update pushed from the admin panel.")
@click.option("--check", is_flag=True, help="Check for available updates without installing.")
def update_command(model: str | None, is_new: bool, check: bool) -> None:
    """Update a published AI model, or the InferForge CLI itself.

    forge update MODEL pushes your latest weights and persona to the site,
    keeping the same embed URL and key so pages using it need no change.
    """
    if model:
        _update_model(model)
        return
    if not is_new and not check:
        console.print("Use [cyan]forge update --new[/] to install the latest admin-pushed update.")
        console.print("Use [cyan]forge update --check[/] to check without installing.")
        return
    try:
        token = None
        try:
            token = (Path.home() / ".inferforge" / "admin_token").read_text().strip()
        except OSError:
            pass
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with console.status("[cyan]Checking for updates...[/]"):
            resp = httpx.get(UPDATE_API, timeout=15.0)
        if resp.status_code == 401 or not resp.is_success:
            resp = httpx.get(ADMIN_UPDATE_API, headers=headers, timeout=15.0)
        if not resp.is_success:
            console.print(f"[red]Could not check updates:[/] {resp.status_code}")
            return
        data = resp.json()
        updates = data.get("updates") or ([data.get("update")] if data.get("update") else [])
        if not updates:
            console.print("[yellow]No updates pushed yet.[/]")
            return
        latest = updates[0]
        version = latest.get("version", "unknown")
        notes = latest.get("notes", "")
        pushed = latest.get("pushedAt", "")
        console.print(f"[bold]Latest update:[/] [cyan]{version}[/]  [dim]{pushed}[/]")
        if notes:
            console.print(f"[dim]Notes:[/] {notes}")
        public_version = None
        try:
            with open(Path("public/version.json"), "r", encoding="utf-8") as f:
                public_version = json.load(f).get("version")
        except Exception:
            pass
        if public_version:
            console.print(f"[dim]Public version: {public_version}[/]")
        if check:
            return
        console.print("[cyan]Updating from HyperNeural index...[/]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "inferforge", "--index-url", "https://hyperneural.cfd/pypi/simple/", "--extra-index-url", "https://pypi.org/simple/"])
        console.print(f"[green]OK[/] Updated to {version}. Restart your shell if needed.")
    except Exception as e:
        console.print(f"[red]Update failed:[/] {e}")
        raise SystemExit(1)
