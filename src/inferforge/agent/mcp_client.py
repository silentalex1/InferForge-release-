from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPServer:
    name: str
    url: str
    tools: list[MCPTool]
    connected: bool = False


class MCPClient:
    def __init__(self):
        self.servers: dict[str, MCPServer] = {}
        self._client = None
    
    def add_server(self, name: str, url: str) -> bool:
        if not HTTPX_AVAILABLE:
            return False
        
        try:
            server = MCPServer(name=name, url=url, tools=[], connected=False)
            self.servers[name] = server
            return True
        except Exception:
            return False
    
    async def connect_server(self, name: str) -> bool:
        if not HTTPX_AVAILABLE:
            return False
        
        server = self.servers.get(name)
        if not server:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{server.url}/tools/list",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
                )
                if response.status_code == 200:
                    data = response.json()
                    if "result" in data and "tools" in data["result"]:
                        for tool_data in data["result"]["tools"]:
                            tool = MCPTool(
                                name=tool_data["name"],
                                description=tool_data.get("description", ""),
                                input_schema=tool_data.get("inputSchema", {})
                            )
                            server.tools.append(tool)
                        server.connected = True
                        return True
        except Exception:
            pass
        
        return False
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not HTTPX_AVAILABLE:
            return {"error": "httpx not available"}
        
        server = self.servers.get(server_name)
        if not server or not server.connected:
            return {"error": f"Server {server_name} not connected"}
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{server.url}/tools/call",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": arguments
                        }
                    }
                )
                if response.status_code == 200:
                    return response.json()
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_all_tools(self) -> list[tuple[str, MCPTool]]:
        tools = []
        for server_name, server in self.servers.items():
            for tool in server.tools:
                tools.append((server_name, tool))
        return tools
    
    def get_tool(self, tool_name: str) -> tuple[str, MCPTool] | None:
        for server_name, server in self.servers.items():
            for tool in server.tools:
                if tool.name == tool_name:
                    return (server_name, tool)
        return None


_global_mcp_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    global _global_mcp_client
    if _global_mcp_client is None:
        _global_mcp_client = MCPClient()
    return _global_mcp_client


def reset_mcp_client() -> None:
    global _global_mcp_client
    _global_mcp_client = None
