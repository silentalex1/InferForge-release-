from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inferforge import __version__
from inferforge.core.registry import Registry
from inferforge.engine import ChatMessage, get_router
from inferforge.model.identity import INFERFORGE_BETA

try:
    from inferforge.server.auth import check_rate_limit, verify_api_key
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    async def verify_api_key(request: Request) -> str:
        return "dev-mode"

app = FastAPI(
    title="InferForge",
    version=__version__,
    description="Faster local LLM runtime — OpenAI-compatible API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessageIn(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessageIn]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


def _resolve_model_name(name: str) -> str:
    key = (name or "").strip().lower()
    if key in {"inferforge", "beta", "inferforge beta", "inferforge-beta"}:
        return INFERFORGE_BETA
    return name


def _authorize_chat(request: Request, model: str) -> None:
    """Accept a real API key, or a per-model embed key issued by `forge embedd --sdk`.

    Embed keys (``sk-embed-*``) only authorize chat on the model they were
    generated for — they cannot list models, embed, or hit other endpoints.
    """
    if not AUTH_AVAILABLE:
        return
    from inferforge.core.config import load_settings
    from inferforge.server.auth import get_api_key_manager

    key = ""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        key = auth[7:]
    key = key or request.headers.get("x-api-key", "") or request.headers.get("api-key", "")
    if key and get_api_key_manager().validate_key(key):
        return
    embed_keys = load_settings().get("embed_keys") or {}
    if key and embed_keys.get(model) == key:
        return
    raise HTTPException(status_code=401, detail="Missing or invalid API key")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "InferForge",
        "version": __version__,
        "status": "online",
        "channel": "beta",
        "docs": "/docs",
        "default_model": INFERFORGE_BETA,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/sdk/inferforge.js")
def sdk_js(request: Request):
    from fastapi.responses import Response

    from inferforge.embedded.sdk import render_sdk

    origin = str(request.base_url).rstrip("/")
    return Response(render_sdk(endpoint=origin), media_type="application/javascript")


@app.get("/sdk/{model_name}.js")
def sdk_js_for_model(model_name: str, request: Request, key: str = ""):
    from fastapi.responses import Response

    from inferforge.embedded.publish import sdk_slug
    from inferforge.embedded.sdk import render_sdk

    origin = str(request.base_url).rstrip("/")
    resolved = model_name
    wanted = sdk_slug(model_name)
    for record in Registry().list():
        if sdk_slug(record.name) == wanted:
            resolved = record.name
            break

    return Response(
        render_sdk(model=resolved, endpoint=origin, api_key=key),
        media_type="application/javascript",
        headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"},
    )


@app.get("/api/tags")
@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    models = Registry().list()
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "created": int(m.imported_at or time.time()),
                "owned_by": "inferforge",
                "size": m.size,
                "details": {
                    "family": m.family,
                    "parameter_size": m.parameter_size,
                    "quantization": m.quantization,
                    "format": m.format,
                },
            }
            for m in models
        ],
        "models": [
            {
                "name": m.name,
                "model": m.name,
                "size": m.size,
                "digest": m.digest,
                "details": {
                    "family": m.family,
                    "parameter_size": m.parameter_size,
                    "quantization_level": m.quantization,
                    "format": m.format,
                },
            }
            for m in models
        ],
    }


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[ChatMessageIn] = Field(default_factory=list)
    stream: bool = False
    system: str | None = None
    options: dict[str, Any] | None = None


class EmbeddingRequest(BaseModel):
    model: str = "text-embedding-ada-002"
    input: str | list[str]
    encoding_format: str = "float"


@app.post("/v1/embeddings")
async def create_embeddings(body: EmbeddingRequest, api_key: str = Depends(verify_api_key) if AUTH_AVAILABLE else None) -> dict[str, Any]:
    """Generate embeddings using sentence-transformers or model-specific embeddings."""
    try:
        # Try to use sentence-transformers for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Handle both string and list inputs
            texts = [body.input] if isinstance(body.input, str) else body.input
            
            # Generate embeddings
            embeddings = model.encode(texts, normalize_embeddings=True)
            
            # Format response
            data = []
            for idx, embedding in enumerate(embeddings):
                data.append({
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding.tolist()
                })
            
            return {
                "object": "list",
                "data": data,
                "model": "all-MiniLM-L6-v2",
                "usage": {
                    "prompt_tokens": sum(len(t.split()) for t in texts),
                    "total_tokens": sum(len(t.split()) for t in texts)
                }
            }
            
        except ImportError:
            # Fallback: Use simple TF-IDF or word averaging if sentence-transformers not available
            from collections import Counter

            import numpy as np
            
            texts = [body.input] if isinstance(body.input, str) else body.input
            
            # Simple word-based embedding (768 dimensions to match standard models)
            embeddings = []
            for text in texts:
                words = text.lower().split()
                # Create a simple hash-based embedding
                embedding = np.zeros(384)
                for word in words:
                    word_hash = hash(word) % 384
                    embedding[word_hash] += 1.0
                
                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                embeddings.append(embedding)
            
            data = []
            for idx, embedding in enumerate(embeddings):
                data.append({
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding.tolist()
                })
            
            return {
                "object": "list",
                "data": data,
                "model": "simple-embeddings",
                "usage": {
                    "prompt_tokens": sum(len(t.split()) for t in texts),
                    "total_tokens": sum(len(t.split()) for t in texts)
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


async def stream_chat_response(model_name: str, messages: list[ChatMessage], options: dict[str, Any] | None) -> AsyncIterator[str]:
    import json

    from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

    reg = Registry()
    record = reg.get(model_name)
    if record is None:
        yield f'data: {json.dumps({"error": {"message": f"model not found: {model_name}"}})}\n\n'
        yield "data: [DONE]\n\n"
        return

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    def frame(delta: dict[str, Any], finish: str | None = None) -> str:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": record.name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    try:
        engine = get_router().resolve(record)
        streamer = getattr(engine, "stream_chat", None)
        if callable(streamer):
            async for token in iterate_in_threadpool(streamer(messages, options=options)):
                if token:
                    yield frame({"content": token})
        else:
            full = await run_in_threadpool(engine.chat, messages, options=options)
            if full:
                yield frame({"content": full})
        yield frame({}, "stop")
    except Exception as exc:
        yield f'data: {json.dumps({"error": {"message": str(exc), "type": "upstream_error"}})}\n\n'

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(body: ChatCompletionRequest, request: Request):
    model_name = _resolve_model_name(body.model)
    _authorize_chat(request, model_name)
    if body.stream:
        messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
        options: dict[str, Any] = {}
        if body.temperature is not None:
            options["temperature"] = body.temperature
        if body.max_tokens is not None:
            options["max_tokens"] = body.max_tokens
        
        return StreamingResponse(
            stream_chat_response(model_name, messages, options or None),
            media_type="text/event-stream"
        )

    reg = Registry()
    record = reg.get(model_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"model not found: {body.model}")

    router = get_router()
    engine = router.resolve(record)
    try:
        messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
        options: dict[str, Any] = {}
        if body.temperature is not None:
            options["temperature"] = body.temperature
        if body.max_tokens is not None:
            options["max_tokens"] = body.max_tokens
        content = engine.chat(messages, options=options or None)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"model backend error: {exc}") from exc

    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": created,
        "model": record.name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.content.split()) for m in messages),
            "completion_tokens": len(content.split()),
            "total_tokens": sum(len(m.content.split()) for m in messages) + len(content.split()),
        },
    }


@app.post("/api/chat")
def ollama_chat(body: OllamaChatRequest, request: Request) -> dict[str, Any]:
    model_name = _resolve_model_name(body.model)
    _authorize_chat(request, model_name)
    reg = Registry()
    record = reg.get(model_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"model not found: {body.model}")

    router = get_router()
    engine = router.resolve(record)
    try:
        messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
        content = engine.chat(messages, system=body.system, options=body.options)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"model backend error: {exc}") from exc

    return {
        "model": record.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
