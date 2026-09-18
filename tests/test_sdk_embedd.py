from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferforge.core.registry import ModelRecord, Registry
from inferforge.embedded.publish import sdk_slug, sdk_url
from inferforge.embedded.sdk import (
    MISSING_JS,
    SDK_JS,
    SITE_URL,
    render_embed_html,
    render_missing_sdk,
    render_sdk,
)


def test_generated_worker_template_matches_python_source() -> None:
    generated = Path(__file__).resolve().parents[1] / "lib" / "sdk" / "template.ts"
    if not generated.exists():
        pytest.skip("run scripts/sync_sdk.py to generate the worker template")
    text = generated.read_text(encoding="utf-8")
    assert SDK_JS in text
    assert MISSING_JS in text


def test_render_sdk_substitutes_model_endpoint_and_fallback() -> None:
    js = render_sdk(
        model="my-model",
        endpoint="https://inferforge.org",
        api_key="sk-embed-abc",
        fallback="http://127.0.0.1:11435",
    )
    assert 'DEFAULT_ENDPOINT = "https://inferforge.org"' in js
    assert 'DEFAULT_MODEL = "my-model"' in js
    assert 'DEFAULT_API_KEY = "sk-embed-abc"' in js
    assert 'DEFAULT_FALLBACK = "http://127.0.0.1:11435"' in js
    assert "global.InferForge" in js


def test_render_sdk_blank_uses_origin_fallback() -> None:
    js = render_sdk()
    assert 'DEFAULT_ENDPOINT = ""' in js
    assert 'DEFAULT_MODEL = ""' in js
    assert "location.origin" in js


def test_render_sdk_has_no_template_literals() -> None:
    js = render_sdk(model="m", endpoint="https://inferforge.org")
    assert "`" not in js
    assert "${" not in js


def test_sdk_slug_normalises_model_names() -> None:
    assert sdk_slug("BatProx-AI") == "batprox-ai"
    assert sdk_slug("llama3:8b") == "llama3-8b"
    assert sdk_slug("org/Model Name") == "org-model-name"
    assert sdk_slug("   ") == "model"


def test_sdk_url_builds_hosted_link() -> None:
    url = sdk_url(SITE_URL, "batprox-ai", "sk-embed-abc", "%23inferforge-chat")
    assert url == "https://inferforge.org/sdk/batprox-ai.js?key=sk-embed-abc&mount=%23inferforge-chat"
    assert sdk_url(SITE_URL, "batprox-ai") == "https://inferforge.org/sdk/batprox-ai.js"


def test_render_embed_html_points_at_the_site() -> None:
    html = render_embed_html("m1", "https://inferforge.org", api_key="sk-embed-abc")
    assert 'src="https://inferforge.org/sdk/m1.js?key=sk-embed-abc&mount=%23inferforge-chat"' in html
    assert 'apiKey: "sk-embed-abc"' in html
    assert 'id="inferforge-chat"' in html


def test_render_embed_html_without_key() -> None:
    html = render_embed_html("m1", "https://inferforge.org")
    assert 'apiKey: ""' in html
    assert "key=" not in html


def test_render_missing_sdk_is_actionable() -> None:
    js = render_missing_sdk("ghost")
    assert "not published yet" in js
    assert 'var MODEL = "ghost"' in js
    assert "Run: forge embedd " in js


def test_sdk_route_serves_javascript() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from inferforge.server.api import app

    client = TestClient(app)
    res = client.get("/sdk/inferforge.js")
    assert res.status_code == 200
    assert "javascript" in res.headers["content-type"]
    assert "global.InferForge" in res.text


def test_sdk_model_route_bakes_model_and_key() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from inferforge.server.api import app

    client = TestClient(app)
    res = client.get("/sdk/my-model.js?key=sk-embed-xyz")
    assert res.status_code == 200
    assert 'DEFAULT_MODEL = "my-model"' in res.text
    assert 'DEFAULT_API_KEY = "sk-embed-xyz"' in res.text
    assert 'DEFAULT_ENDPOINT = "http://testserver"' in res.text


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> dict:
    saved: dict = {}
    monkeypatch.setattr("inferforge.core.config.load_settings", lambda: {})
    monkeypatch.setattr("inferforge.core.config.save_settings", lambda s: saved.update(s))
    return saved


def _patch_publish(monkeypatch: pytest.MonkeyPatch, result: dict | None = None) -> list[dict]:
    calls: list[dict] = []

    def fake_publish(**kwargs):
        calls.append(kwargs)
        return result if result is not None else {"ok": True}

    monkeypatch.setattr("inferforge.embedded.publish.publish_sdk", fake_publish)
    monkeypatch.setattr("inferforge.embedded.publish.persona_from_ollama", lambda name: "")
    return calls


def test_embed_sdk_publishes_and_writes_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferforge.commands import embedd_cmd

    monkeypatch.setattr(Registry, "get", lambda self, name: ModelRecord(name=name, backend="huggingface"))
    saved = _patch_settings(monkeypatch)
    calls = _patch_publish(monkeypatch)

    embedd_cmd._embed_sdk("agent-v2", str(tmp_path), "https://ai.example.com", None, True, False)

    sdk_dir = tmp_path / "sdk" / "agent-v2"
    js = (sdk_dir / "agent-v2.js").read_text(encoding="utf-8")
    html = (sdk_dir / "ai-embed.html").read_text(encoding="utf-8")
    config = json.loads((sdk_dir / "sdk-config.json").read_text(encoding="utf-8"))

    assert 'DEFAULT_MODEL = "agent-v2"' in js
    assert 'DEFAULT_ENDPOINT = "https://inferforge.org"' in js
    assert 'DEFAULT_FALLBACK = "https://ai.example.com"' in js
    assert "agent-v2" in html

    assert config["model"] == "agent-v2"
    assert config["slug"] == "agent-v2"
    assert config["site"] == "https://inferforge.org"
    assert config["upstream"] == "https://ai.example.com"
    assert config["hosted_sdk_url"] == "https://inferforge.org/sdk/agent-v2.js"
    assert config["connected_link"].startswith("https://inferforge.org/sdk/agent-v2.js?key=sk-embed-")
    assert config["published"] is True

    assert saved["always_on_models"] == ["agent-v2"]
    assert saved["embed_keys"]["agent-v2"].startswith("sk-embed-")
    assert saved["publish_tokens"]["agent-v2"].startswith("sk-pub-")

    assert len(calls) == 1
    assert calls[0]["slug"] == "agent-v2"
    assert calls[0]["endpoint"] == "https://ai.example.com"
    assert calls[0]["publish_token"] == saved["publish_tokens"]["agent-v2"]


def test_embed_sdk_slugifies_model_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferforge.commands import embedd_cmd

    monkeypatch.setattr(Registry, "get", lambda self, name: ModelRecord(name=name))
    _patch_settings(monkeypatch)
    _patch_publish(monkeypatch)

    embedd_cmd._embed_sdk("llama3:8b", str(tmp_path), "https://ai.example.com", None, True, False)

    config = json.loads((tmp_path / "sdk" / "llama3-8b" / "sdk-config.json").read_text(encoding="utf-8"))
    assert config["model"] == "llama3:8b"
    assert config["slug"] == "llama3-8b"
    assert config["hosted_sdk_url"] == "https://inferforge.org/sdk/llama3-8b.js"
    assert (tmp_path / "sdk" / "llama3-8b" / "llama3-8b.js").exists()


def test_embed_sdk_records_publish_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferforge.commands import embedd_cmd

    monkeypatch.setattr(Registry, "get", lambda self, name: ModelRecord(name=name))
    _patch_settings(monkeypatch)
    _patch_publish(monkeypatch, {"ok": False, "error": "name-taken"})

    embedd_cmd._embed_sdk("m", str(tmp_path), "https://ai.example.com", None, True, False)

    config = json.loads((tmp_path / "sdk" / "m" / "sdk-config.json").read_text(encoding="utf-8"))
    assert config["published"] is False
    assert config["publish_error"] == "name-taken"


def test_embed_sdk_no_publish_skips_the_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferforge.commands import embedd_cmd

    monkeypatch.setattr(Registry, "get", lambda self, name: ModelRecord(name=name))
    _patch_settings(monkeypatch)
    calls = _patch_publish(monkeypatch)

    embedd_cmd._embed_sdk("m", str(tmp_path), "https://ai.example.com", None, False, False)

    assert calls == []
    config = json.loads((tmp_path / "sdk" / "m" / "sdk-config.json").read_text(encoding="utf-8"))
    assert config["published"] is False


def test_embed_sdk_defaults_to_the_local_serve_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferforge.commands import embedd_cmd

    monkeypatch.setattr(Registry, "get", lambda self, name: ModelRecord(name=name))
    _patch_settings(monkeypatch)
    _patch_publish(monkeypatch)

    embedd_cmd._embed_sdk("m", str(tmp_path), None, None, True, False)

    config = json.loads((tmp_path / "sdk" / "m" / "sdk-config.json").read_text(encoding="utf-8"))
    assert config["upstream"] == "http://127.0.0.1:11435"


def test_embed_sdk_rotate_key_issues_a_new_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferforge.commands import embedd_cmd

    monkeypatch.setattr(Registry, "get", lambda self, name: ModelRecord(name=name))
    monkeypatch.setattr(
        "inferforge.core.config.load_settings",
        lambda: {"embed_keys": {"m": "sk-embed-old"}, "publish_tokens": {"m": "sk-pub-keep"}},
    )
    saved: dict = {}
    monkeypatch.setattr("inferforge.core.config.save_settings", lambda s: saved.update(s))
    _patch_publish(monkeypatch)

    embedd_cmd._embed_sdk("m", str(tmp_path), "https://ai.example.com", None, True, True)

    assert saved["embed_keys"]["m"] != "sk-embed-old"
    assert saved["publish_tokens"]["m"] == "sk-pub-keep"


def test_embed_key_authorizes_chat_for_its_model(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import inferforge.server.api as api

    monkeypatch.setattr("inferforge.core.config.load_settings", lambda: {"embed_keys": {"m1": "sk-embed-good"}})

    class FakeManager:
        def validate_key(self, key: str) -> bool:
            return False

    monkeypatch.setattr("inferforge.server.auth.get_api_key_manager", lambda: FakeManager())
    monkeypatch.setattr(Registry, "get", lambda self, name: ModelRecord(name=name))

    class FakeEngine:
        def chat(self, messages, options=None):
            return "hi"

        def close(self):
            pass

    monkeypatch.setattr(api, "get_router", lambda: type("R", (), {"resolve": lambda s, r: FakeEngine()})())

    client = TestClient(api.app)
    res = client.post(
        "/v1/chat/completions",
        json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer sk-embed-good"},
    )
    assert res.status_code == 200
    assert res.json()["choices"][0]["message"]["content"] == "hi"


def test_embed_key_rejected_for_other_model(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import inferforge.server.api as api

    monkeypatch.setattr("inferforge.core.config.load_settings", lambda: {"embed_keys": {"m1": "sk-embed-good"}})

    class FakeManager:
        def validate_key(self, key: str) -> bool:
            return False

    monkeypatch.setattr("inferforge.server.auth.get_api_key_manager", lambda: FakeManager())

    client = TestClient(api.app)
    res = client.post(
        "/v1/chat/completions",
        json={"model": "m2", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer sk-embed-good"},
    )
    assert res.status_code == 401


def test_embed_sdk_unknown_model_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    from inferforge.commands import embedd_cmd

    monkeypatch.setattr(Registry, "get", lambda self, name: None)
    with pytest.raises(SystemExit):
        embedd_cmd._embed_sdk("nope", None, None, None, True, False)
