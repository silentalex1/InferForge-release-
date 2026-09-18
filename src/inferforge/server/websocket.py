"""WebSocket support for real-time streaming."""

from __future__ import annotations

from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from inferforge.core.registry import Registry
from inferforge.engine.base import ChatMessage, GenerationConfig
from inferforge.engine.unified_router import get_unified_router


class WebSocketManager:
    """Manages WebSocket connections for streaming inference."""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.router = get_unified_router()
        self.registry = Registry()
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect(self, client_id: str) -> None:
        """Remove WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def handle_message(self, websocket: WebSocket, client_id: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            while True:
                data = await websocket.receive_json()
                await self.process_request(websocket, data)
        except WebSocketDisconnect:
            self.disconnect(client_id)
    
    async def process_request(self, websocket: WebSocket, data: dict[str, Any]) -> None:
        """Process inference request and stream response."""
        try:
            model_name = data.get("model")
            messages_data = data.get("messages", [])
            config_data = data.get("config", {})
            
            # Get model
            model = self.registry.get(model_name)
            if not model:
                await websocket.send_json({"error": f"Model not found: {model_name}"})
                return
            
            # Get engine
            engine = self.router.get_engine(model)
            
            # Parse messages
            messages = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in messages_data
            ]
            
            # Parse config
            config = GenerationConfig(**config_data) if config_data else None
            
            # Stream response
            await websocket.send_json({"status": "started"})
            
            for chunk in engine.stream(messages, config):
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk
                })
            
            await websocket.send_json({"status": "completed"})
        
        except Exception as e:
            await websocket.send_json({"error": str(e)})


ws_manager = WebSocketManager()
