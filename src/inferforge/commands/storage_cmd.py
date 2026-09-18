from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from inferforge.core.config import (
    get_remote_config,
    get_storage_config,
    load_settings,
    save_settings,
)
from inferforge.core.registry import Registry
from inferforge.storage.s3_backend import S3StorageBackend

console = Console(force_terminal=True, stderr=True)


@click.group("storage")
def storage_command() -> None:
    pass


@storage_command.command("upload")
@click.argument("model")
@click.option("--key", default=None, help="Custom storage key.")
def upload_model(model: str, key: str | None) -> None:
    pass
    config = get_storage_config()
    if not config["enabled"]:
        console.print("[red]Storage is not enabled. Configure it first.[/]")
        raise SystemExit(1)

    reg = Registry()
    record = reg.get(model)
    if not record:
        console.print(f"[red]Model not found:[/] {model}")
        raise SystemExit(1)

    if not record.path:
        console.print(f"[red]Model has no local path:[/] {model}")
        raise SystemExit(1)

    local_path = Path(record.path)
    if not local_path.exists():
        console.print(f"[red]Local file not found:[/] {local_path}")
        raise SystemExit(1)

    storage_key = key or f"models/{model.replace(':', '-')}.gguf"

    console.print(f"[bold dark_orange]◈ InferForge[/] uploading [cyan]{model}[/] to storage…")

    try:
        backend = S3StorageBackend(
            endpoint_url=config["endpoint"] or None,
            access_key=config["access_key"] or None,
            secret_key=config["secret_key"] or None,
            bucket_name=config["bucket"],
            region=config["region"],
        )

        with Progress(
            SpinnerColumn(style="dark_orange"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            console=console,
        ) as progress:
            task = progress.add_task("uploading…", total=None)
            url = backend.upload_file(local_path, storage_key)
            progress.update(task, completed=1, total=1, description="uploaded")

        console.print(f"[green]✓[/] uploaded to: [cyan]{url}[/]")

        record.meta["storage_key"] = storage_key
        record.meta["storage_url"] = url
        reg.upsert(record)

    except Exception as e:
        console.print(f"[red]Upload failed:[/] {e}")
        raise SystemExit(1)


@storage_command.command("download")
@click.argument("model")
@click.option("--force", is_flag=True, help="Overwrite existing local file.")
def download_model(model: str, force: bool) -> None:
    pass
    config = get_storage_config()
    if not config["enabled"]:
        console.print("[red]Storage is not enabled. Configure it first.[/]")
        raise SystemExit(1)

    reg = Registry()
    record = reg.get(model)
    if not record:
        console.print(f"[red]Model not found:[/] {model}")
        raise SystemExit(1)

    storage_key = record.meta.get("storage_key")
    if not storage_key:
        console.print(f"[red]Model has no storage key:[/] {model}")
        raise SystemExit(1)

    from inferforge.core.config import models_dir

    local_path = models_dir() / f"{model.replace(':', '-')}.gguf"
    if local_path.exists() and not force:
        console.print(f"[yellow]Local file already exists:[/] {local_path}")
        console.print("[dim]Use --force to overwrite[/]")
        return

    console.print(f"[bold dark_orange]◈ InferForge[/] downloading [cyan]{model}[/] from storage…")

    try:
        backend = S3StorageBackend(
            endpoint_url=config["endpoint"] or None,
            access_key=config["access_key"] or None,
            secret_key=config["secret_key"] or None,
            bucket_name=config["bucket"],
            region=config["region"],
        )

        with Progress(
            SpinnerColumn(style="dark_orange"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            console=console,
        ) as progress:
            task = progress.add_task("downloading…", total=None)
            backend.download_file(storage_key, local_path)
            progress.update(task, completed=1, total=1, description="downloaded")

        console.print(f"[green]✓[/] downloaded to: [cyan]{local_path}[/]")

        record.path = str(local_path)
        reg.upsert(record)

    except Exception as e:
        console.print(f"[red]Download failed:[/] {e}")
        raise SystemExit(1)


@storage_command.command("list")
def list_storage() -> None:
    pass
    config = get_storage_config()
    if not config["enabled"]:
        console.print("[red]Storage is not enabled. Configure it first.[/]")
        raise SystemExit(1)

    console.print("[bold dark_orange]◈ InferForge[/] listing remote storage…")

    try:
        backend = S3StorageBackend(
            endpoint_url=config["endpoint"] or None,
            access_key=config["access_key"] or None,
            secret_key=config["secret_key"] or None,
            bucket_name=config["bucket"],
            region=config["region"],
        )

        files = backend.list_files(prefix="models/")

        if not files:
            console.print("[dim]No models in storage[/]")
            return

        table = Table(title="Remote Models", show_header=True, header_style="bold")
        table.add_column("key", style="cyan")
        table.add_column("size", style="green")

        for file_key in files:
            size = backend.get_file_size(file_key)
            size_gb = size / (1024**3)
            table.add_row(file_key, f"{size_gb:.2f} GB")

        console.print(table)

    except Exception as e:
        console.print(f"[red]List failed:[/] {e}")
        raise SystemExit(1)


@storage_command.command("configure")
@click.option("--endpoint", default=None, help="S3 endpoint URL.")
@click.option("--access-key", default=None, help="Access key.")
@click.option("--secret-key", default=None, help="Secret key.")
@click.option("--bucket", default=None, help="Bucket name.")
@click.option("--region", default=None, help="Region.")
def configure_storage(
    endpoint: str | None,
    access_key: str | None,
    secret_key: str | None,
    bucket: str | None,
    region: str | None,
) -> None:
    pass
    settings = load_settings()

    if endpoint is not None:
        settings["storage_endpoint"] = endpoint
    if access_key is not None:
        settings["storage_access_key"] = access_key
    if secret_key is not None:
        settings["storage_secret_key"] = secret_key
    if bucket is not None:
        settings["storage_bucket"] = bucket
    if region is not None:
        settings["storage_region"] = region

    settings["storage_enabled"] = True
    save_settings(settings)

    console.print("[green]✓[/] Storage configured successfully")
    console.print(f"[dim]Endpoint:[/] {settings.get('storage_endpoint')}")
    console.print(f"[dim]Bucket:[/] {settings.get('storage_bucket')}")


@click.group("remote")
def remote_command() -> None:
    pass


@remote_command.command("configure")
@click.option("--endpoint", default=None, help="Remote API endpoint.")
@click.option("--api-key", default=None, help="API key.")
@click.option("--timeout", default=None, type=float, help="Request timeout.")
@click.option("--prefer", is_flag=True, help="Prefer remote execution.")
def configure_remote(
    endpoint: str | None,
    api_key: str | None,
    timeout: float | None,
    prefer: bool,
) -> None:
    pass
    settings = load_settings()

    if endpoint is not None:
        settings["remote_endpoint"] = endpoint
    if api_key is not None:
        settings["remote_api_key"] = api_key
    if timeout is not None:
        settings["remote_timeout"] = timeout
    if prefer:
        settings["prefer_remote"] = True

    settings["remote_enabled"] = True
    save_settings(settings)

    console.print("[green]✓[/] Remote execution configured successfully")
    console.print(f"[dim]Endpoint:[/] {settings.get('remote_endpoint')}")
    console.print(f"[dim]Prefer remote:[/] {settings.get('prefer_remote')}")


@remote_command.command("status")
def remote_status() -> None:
    pass
    config = get_remote_config()
    if not config["enabled"]:
        console.print("[dim]Remote execution is not enabled[/]")
        return

    console.print("[bold dark_orange]◈ InferForge[/] remote status")
    console.print(f"[dim]Endpoint:[/] {config['endpoint']}")
    console.print(f"[dim]Prefer remote:[/] {config['prefer_remote']}")

    try:
        from inferforge.remote.http_backend import HTTPRemoteBackend

        backend = HTTPRemoteBackend(
            endpoint=config["endpoint"],
            api_key=config["api_key"],
            model_name="health-check",
        )
        if backend.health_check():
            console.print("[green]✓[/] Remote endpoint is healthy")
        else:
            console.print("[red]✗[/] Remote endpoint is not healthy")
    except Exception as e:
        console.print(f"[red]✗[/] Health check failed: {e}")
