from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from typing import Any

import httpx

from inferforge.embedded.sdk import SITE_URL

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")

HOSTED_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def persona_from_ollama(name: str, timeout: float = 15.0) -> str:
    if not name or not shutil.which("ollama"):
        return ""
    try:
        out = subprocess.run(
            ["ollama", "show", name, "--system"],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return ""
    if out.returncode != 0:
        return ""
    text = (out.stdout or "").strip()
    if not text or text.lower().startswith("error"):
        return ""
    return text[:8000]


def sdk_slug(model: str) -> str:
    slug = _SLUG_RE.sub("-", str(model).strip().lower()).strip("-.")
    return slug or "model"


def new_embed_key() -> str:
    return "sk-embed-" + uuid.uuid4().hex[:24]


def new_publish_token() -> str:
    return "sk-pub-" + uuid.uuid4().hex


def sdk_url(site: str, slug: str, embed_key: str = "", mount: str = "") -> str:
    url = f"{site.rstrip('/')}/sdk/{slug}.js"
    parts = []
    if embed_key:
        parts.append(f"key={embed_key}")
    if mount:
        parts.append(f"mount={mount}")
    return url + ("?" + "&".join(parts) if parts else "")


def is_local_endpoint(endpoint: str) -> bool:
    lowered = (endpoint or "").lower()
    return any(host in lowered for host in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]", "::1"))


def publish_sdk(
    site: str,
    model: str,
    slug: str,
    embed_key: str,
    publish_token: str,
    endpoint: str,
    fallback: str = "",
    owner: str = "",
    system: str = "",
    hosted_model: str = "",
    timeout: float = 20.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "slug": slug,
        "key": embed_key,
        "endpoint": endpoint,
        "fallback": fallback,
        "owner": owner,
        "system": system,
        "hosted_model": hosted_model or HOSTED_MODEL,
    }
    try:
        resp = httpx.post(
            f"{site.rstrip('/')}/api/sdk/publish",
            json=payload,
            headers={"X-Forge-Token": publish_token},
            timeout=timeout,
        )
    except httpx.RequestError as exc:
        return {"ok": False, "error": f"could not reach {site}: {exc}"}
    if resp.status_code in (200, 201):
        body: dict[str, Any] = {}
        try:
            body = resp.json()
        except ValueError:
            pass
        return {"ok": True, **body}
    detail = resp.text.strip()
    try:
        body = resp.json()
        detail = body.get("message") or body.get("error") or detail
    except ValueError:
        pass
    return {"ok": False, "status": resp.status_code, "error": detail or f"HTTP {resp.status_code}"}


def unpublish_sdk(site: str, slug: str, publish_token: str, timeout: float = 20.0) -> dict[str, Any]:
    try:
        resp = httpx.request(
            "DELETE",
            f"{site.rstrip('/')}/api/sdk/publish",
            json={"slug": slug},
            headers={"X-Forge-Token": publish_token},
            timeout=timeout,
        )
    except httpx.RequestError as exc:
        return {"ok": False, "error": f"could not reach {site}: {exc}"}
    if resp.status_code == 200:
        return {"ok": True}
    return {"ok": False, "status": resp.status_code, "error": resp.text.strip() or f"HTTP {resp.status_code}"}


def default_site() -> str:
    return SITE_URL
