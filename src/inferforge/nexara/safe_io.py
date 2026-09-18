from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class StopFlag:
    def __init__(self) -> None:
        self._stop = False
        self._force = False

    def request(self, force: bool = False) -> None:
        self._stop = True
        if force:
            self._force = True

    def requested(self) -> bool:
        return self._stop

    def forced(self) -> bool:
        return self._force

    def __call__(self) -> bool:
        return self._stop


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, default=str, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def promote_partial(partial: Path, final: Path) -> None:
    partial = Path(partial)
    final = Path(final)
    if not partial.exists():
        return
    os.replace(partial, final)


def cleanup_partials(root: Path) -> list[str]:
    removed: list[str] = []
    if not root.exists():
        return removed
    for pattern in ("*.partial", "*.tmp", ".*.tmp"):
        for p in root.rglob(pattern):
            if not p.is_file():
                continue
            try:
                p.unlink()
                removed.append(str(p))
            except OSError:
                pass
    return removed


def discard_incomplete_cycle(cycle_dir: Path, keep_complete_shards: bool = True) -> None:
    cycle_dir = Path(cycle_dir)
    if not cycle_dir.exists():
        return
    cleanup_partials(cycle_dir)
    manifest = cycle_dir / "manifest.json"
    if keep_complete_shards and manifest.exists():
        return
    if not manifest.exists():
        for p in cycle_dir.glob("shard_*.jsonl"):
            try:
                p.unlink()
            except OSError:
                pass
