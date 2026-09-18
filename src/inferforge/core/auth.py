from __future__ import annotations

import json
from pathlib import Path


def _auth_path() -> Path:
    return Path.home() / ".inferforge" / "auth.json"


def load_auth_state() -> dict:
    path = _auth_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"connected": False}


def save_auth_state(state: dict) -> None:
    path = _auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_connected_user() -> dict | None:
    state = load_auth_state()
    if state.get("connected"):
        return state
    return None