from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inferforge.agent.security import (
    AuditEntry,
    OperationType,
    RiskLevel,
    SecurityConfig,
    SecurityManager,
    get_security_manager,
)

TOOL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
JSON_FENCE_RE = re.compile(
    r"```(?:json|tool|tool_call)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)
TOOL_NAMES = {
    "create_file",
    "edit_file",
    "delete_file",
    "read_file",
    "open_file",
    "list_dir",
    "run_command",
    "web_request",
    "check_storage",
    "write_file",
    "create",
    "edit",
    "delete",
    "read",
    "open",
    "list",
    "web",
    "fetch",
    "http",
    "storage",
    "disk",
}

NAME_ALIASES = {
    "write_file": "create_file",
    "write": "create_file",
    "create": "create_file",
    "edit": "edit_file",
    "delete": "delete_file",
    "read": "read_file",
    "open": "open_file",
    "list": "list_dir",
    "ls": "list_dir",
    "exec": "run_command",
    "shell": "run_command",
    "bash": "run_command",
    "cmd": "run_command",
    "web": "web_request",
    "fetch": "web_request",
    "http": "web_request",
    "request": "web_request",
    "storage": "check_storage",
    "disk": "check_storage",
    "df": "check_storage",
}

_ALL_TOOL_NAMES = "|".join(
    re.escape(n) for n in sorted(TOOL_NAMES | set(NAME_ALIASES), key=len, reverse=True)
)
BARE_TOOL_JSON_RE = re.compile(
    r"\{[^{}]*\"(?:name|tool|action)\"\s*:\s*\"(?:" + _ALL_TOOL_NAMES + r")\"[^{}]*\}",
    re.DOTALL | re.IGNORECASE,
)

FORBIDDEN_PREFIXES = (
    Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve(),
    Path(os.environ.get("WINDIR", r"C:\Windows")).resolve(),
)
FORBIDDEN_NAMES = {"/etc/passwd", "/etc/shadow", "C:\\Windows\\System32"}

DANGEROUS_CMD = re.compile(
    r"(rm\s+-rf\s+[\\/]|format\s+|del\s+/[sf]|Remove-Item\s+-Recurse\s+-Force\s+[A-Z]:\\|"
    r"shutdown|mkfs|dd\s+if=|:\(\)\s*\{)",
    re.IGNORECASE,
)


@dataclass
class ToolResult:
    name: str
    ok: bool
    message: str
    data: Any = None


def _normalize_call(obj: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name") or obj.get("tool") or obj.get("action") or "").strip()
    if not name:
        return None
    name = NAME_ALIASES.get(name, name)
    if name not in TOOL_NAMES and name not in NAME_ALIASES.values():
        return None
    name = NAME_ALIASES.get(name, name)
    out = dict(obj)
    out["name"] = name
    if "filepath" in out and "path" not in out:
        out["path"] = out["filepath"]
    if "file" in out and "path" not in out:
        out["path"] = out["file"]
    if "filename" in out and "path" not in out:
        out["path"] = out["filename"]
    if "text" in out and "content" not in out:
        out["content"] = out["text"]
    if "body" in out and "content" not in out:
        out["content"] = out["body"]
    if "cmd" in out and "command" not in out:
        out["command"] = out["cmd"]
    return out


def _try_parse_json(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return _normalize_call(obj) if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        obj = json.loads(raw.replace("\\", "\\\\"))
        return _normalize_call(obj) if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(obj: dict[str, Any] | None) -> None:
        if not obj:
            return
        key = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        if key in seen:
            return
        seen.add(key)
        calls.append(obj)

    for match in TOOL_BLOCK_RE.finditer(text or ""):
        add(_try_parse_json(match.group(1)))

    for match in JSON_FENCE_RE.finditer(text or ""):
        add(_try_parse_json(match.group(1)))

    for match in BARE_TOOL_JSON_RE.finditer(text or ""):
        add(_try_parse_json(match.group(0)))

    return calls


def strip_tool_calls(text: str) -> str:
    cleaned = text or ""
    cleaned = TOOL_BLOCK_RE.sub("", cleaned)
    cleaned = JSON_FENCE_RE.sub("", cleaned)

    def _drop_bare(m: re.Match[str]) -> str:
        obj = _try_parse_json(m.group(0))
        return "" if obj else m.group(0)

    cleaned = BARE_TOOL_JSON_RE.sub(_drop_bare, cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _safe_path(path_str: str, workspace: Path, security: SecurityManager | None = None) -> Path:
    raw = Path(path_str).expanduser()
    if not raw.is_absolute():
        raw = (workspace / raw).resolve()
    else:
        raw = raw.resolve()

    # Check workspace access if security manager is provided
    if security and not security.check_workspace_access(raw):
        raise PermissionError(
            f"Path outside allowed workspaces: {raw}\n"
            f"Allowed workspaces: {[str(w) for w in security.config.allowed_workspaces]}"
        )

    for forbidden in FORBIDDEN_PREFIXES:
        try:
            raw.relative_to(forbidden)
            raise PermissionError(f"Refusing to access system path: {raw}")
        except ValueError:
            pass
    if str(raw) in FORBIDDEN_NAMES:
        raise PermissionError(f"Refusing to access protected path: {raw}")
    return raw


def _create_file(path: Path, content: str, security: SecurityManager | None = None) -> ToolResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content is not None else "", encoding="utf-8")
    
    if security:
        security.log_audit(
            AuditEntry(
                timestamp=time.time(),
                operation=OperationType.CREATE_FILE,
                risk_level=RiskLevel.LOW,
                path=str(path),
                success=True,
                metadata={"size_bytes": path.stat().st_size},
            )
        )
    
    return ToolResult(
        "create_file",
        True,
        f"Created {path}",
        {"path": str(path), "bytes": path.stat().st_size},
    )


def _edit_file(path: Path, old: str, new: str, security: SecurityManager | None = None) -> ToolResult:
    if not path.exists():
        return ToolResult("edit_file", False, f"File not found: {path}")
    
    # Create backup if security manager is enabled
    backup_path = None
    if security:
        backup_path = security.create_backup(path)
    
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if not old:
            path.write_text(new, encoding="utf-8")
            if security:
                security.log_audit(
                    AuditEntry(
                        timestamp=time.time(),
                        operation=OperationType.EDIT_FILE,
                        risk_level=RiskLevel.LOW,
                        path=str(path),
                        success=True,
                        metadata={"backup": str(backup_path) if backup_path else None},
                    )
                )
            return ToolResult("edit_file", True, f"Replaced contents of {path}", {"path": str(path), "backup": str(backup_path) if backup_path else None})
        return ToolResult("edit_file", False, f"Old text not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    
    if security:
        security.log_audit(
            AuditEntry(
                timestamp=time.time(),
                operation=OperationType.EDIT_FILE,
                risk_level=RiskLevel.LOW,
                path=str(path),
                success=True,
                metadata={"backup": str(backup_path) if backup_path else None},
            )
        )
    
    return ToolResult("edit_file", True, f"Edited {path}", {"path": str(path), "backup": str(backup_path) if backup_path else None})


def _delete_file(path: Path, security: SecurityManager | None = None) -> ToolResult:
    if not path.exists():
        return ToolResult("delete_file", False, f"Not found: {path}")
    
    # Create backup before deletion if security manager is enabled
    backup_path = None
    if security:
        backup_path = security.create_backup(path)
    
    if path.is_dir():
        try:
            path.rmdir()
        except OSError:
            return ToolResult(
                "delete_file",
                False,
                f"Directory not empty (refusing recursive delete): {path}",
            )
    else:
        path.unlink()
    
    if security:
        security.log_audit(
            AuditEntry(
                timestamp=time.time(),
                operation=OperationType.DELETE_FILE,
                risk_level=RiskLevel.HIGH,
                path=str(path),
                success=True,
                metadata={"backup": str(backup_path) if backup_path else None},
            )
        )
    
    return ToolResult("delete_file", True, f"Deleted {path}", {"path": str(path), "backup": str(backup_path) if backup_path else None})


def _read_file(path: Path, max_chars: int = 80_000, security: SecurityManager | None = None) -> ToolResult:
    if not path.exists():
        return ToolResult("read_file", False, f"Not found: {path}")
    if path.is_dir():
        return ToolResult("read_file", False, f"Is a directory: {path}")
    data = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(data) > max_chars
    if truncated:
        data = data[:max_chars] + "\n… [truncated]"
    
    if security:
        security.log_audit(
            AuditEntry(
                timestamp=time.time(),
                operation=OperationType.READ_FILE,
                risk_level=RiskLevel.SAFE,
                path=str(path),
                success=True,
                metadata={"size_bytes": len(data), "truncated": truncated},
            )
        )
    
    return ToolResult(
        "read_file",
        True,
        f"Read {path}",
        {"path": str(path), "content": data, "truncated": truncated},
    )


def _open_file(path: Path) -> ToolResult:
    if not path.exists():
        return ToolResult("open_file", False, f"Not found: {path}")
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        preview = ""
        if path.is_file():
            try:
                preview = path.read_text(encoding="utf-8", errors="replace")[:4000]
            except Exception:
                preview = ""
        return ToolResult(
            "open_file",
            True,
            f"Opened {path}",
            {"path": str(path), "content": preview},
        )
    except Exception as exc:
        return ToolResult("open_file", False, f"Could not open {path}: {exc}")


def _list_dir(path: Path) -> ToolResult:
    if not path.exists():
        return ToolResult("list_dir", False, f"Not found: {path}")
    if not path.is_dir():
        return ToolResult("list_dir", False, f"Not a directory: {path}")
    entries = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        kind = "dir" if child.is_dir() else "file"
        size = child.stat().st_size if child.is_file() else 0
        entries.append({"name": child.name, "kind": kind, "size": size})
    return ToolResult(
        "list_dir",
        True,
        f"Listed {path} ({len(entries)} entries)",
        {"path": str(path), "entries": entries},
    )


def _web_request(url: str, method: str = "GET", headers: dict[str, str] | None = None, body: str | None = None, security: SecurityManager | None = None) -> ToolResult:
    """Execute a web request with rate limiting and domain filtering."""
    if security and not security.config.allow_web_access:
        return ToolResult("web_request", False, "Web access is disabled")
    
    # Check rate limit
    if security and not security.check_web_rate_limit():
        return ToolResult("web_request", False, "Rate limit exceeded. Please wait before making more requests.")
    
    # Check domain whitelist
    if security and security.config.allowed_web_domains:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not any(domain == allowed or domain.endswith(f".{allowed}") for allowed in security.config.allowed_web_domains):
            return ToolResult(
                "web_request", False,
                f"Domain not in whitelist: {domain}\nAllowed: {security.config.allowed_web_domains}"
            )
    
    try:
        import httpx
        
        method = method.upper()
        allowed_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
        if method not in allowed_methods:
            return ToolResult("web_request", False, f"Method not allowed: {method}")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.request(
                method=method,
                url=url,
                headers=headers or {},
                content=body.encode() if body else None,
            )
        
        result_data = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }
        
        # Include response body if it's text and not too large
        try:
            content = response.text
            if len(content) <= 10_000:
                result_data["body"] = content
            else:
                result_data["body"] = content[:10_000] + "\n… [truncated]"
                result_data["truncated"] = True
        except Exception:
            result_data["body"] = "<binary content>"
        
        if security:
            security.log_audit(
                AuditEntry(
                    timestamp=time.time(),
                    operation=OperationType.WEB_REQUEST,
                    risk_level=RiskLevel.MEDIUM,
                    success=response.status_code < 400,
                    metadata={
                        "url": url,
                        "method": method,
                        "status_code": response.status_code,
                        "response_size": len(response.content),
                    },
                )
            )
        
        return ToolResult(
            "web_request",
            response.status_code < 400,
            f"HTTP {response.status_code}",
            result_data,
        )
    except ImportError:
        return ToolResult("web_request", False, "httpx not installed. Install with: pip install httpx")
    except Exception as exc:
        if security:
            security.log_audit(
                AuditEntry(
                    timestamp=time.time(),
                    operation=OperationType.WEB_REQUEST,
                    risk_level=RiskLevel.MEDIUM,
                    success=False,
                    error=str(exc),
                    metadata={"url": url, "method": method},
                )
            )
        return ToolResult("web_request", False, f"Request failed: {exc}")


def _check_storage(security: SecurityManager | None = None) -> ToolResult:
    """Check available storage on the system."""
    import platform
    import shutil
    
    try:
        system = platform.system()
        result_data = {}
        
        if system == "Windows":
            try:
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                total_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p("C:"), 
                    ctypes.byref(free_bytes), 
                    ctypes.byref(total_bytes), 
                    None
                )
                result_data = {
                    "drive": "C:",
                    "total_gb": total_bytes.value / (1024**3),
                    "free_gb": free_bytes.value / (1024**3),
                    "used_gb": (total_bytes.value - free_bytes.value) / (1024**3),
                    "percent_used": ((total_bytes.value - free_bytes.value) / total_bytes.value * 100) if total_bytes.value > 0 else 0,
                }
            except Exception:
                # Fallback to shutil
                usage = shutil.disk_usage("C:")
                result_data = {
                    "drive": "C:",
                    "total_gb": usage.total / (1024**3),
                    "free_gb": usage.free / (1024**3),
                    "used_gb": usage.used / (1024**3),
                    "percent_used": (usage.used / usage.total * 100) if usage.total > 0 else 0,
                }
        else:
            # Unix-like systems
            usage = shutil.disk_usage("/")
            result_data = {
                "drive": "/",
                "total_gb": usage.total / (1024**3),
                "free_gb": usage.free / (1024**3),
                "used_gb": usage.used / (1024**3),
                "percent_used": (usage.used / usage.total * 100) if usage.total > 0 else 0,
            }
        
        if security:
            security.log_audit(
                AuditEntry(
                    timestamp=time.time(),
                    operation=OperationType.RUN_COMMAND,
                    risk_level=RiskLevel.SAFE,
                    success=True,
                    metadata={"action": "check_storage", "result": result_data},
                )
            )
        
        summary = f"{result_data['drive']}: {result_data['free_gb']:.1f}GB free of {result_data['total_gb']:.1f}GB ({result_data['percent_used']:.1f}% used)"
        return ToolResult("check_storage", True, summary, result_data)
    except Exception as exc:
        if security:
            security.log_audit(
                AuditEntry(
                    timestamp=time.time(),
                    operation=OperationType.RUN_COMMAND,
                    risk_level=RiskLevel.SAFE,
                    success=False,
                    error=str(exc),
                    metadata={"action": "check_storage"},
                )
            )
        return ToolResult("check_storage", False, f"Failed to check storage: {exc}")


def _run_command(command: str, workspace: Path, timeout: float = 60.0, security: SecurityManager | None = None) -> ToolResult:
    if DANGEROUS_CMD.search(command or ""):
        if security:
            security.log_audit(
                AuditEntry(
                    timestamp=time.time(),
                    operation=OperationType.RUN_COMMAND,
                    risk_level=RiskLevel.HIGH,
                    command=command,
                    success=False,
                    error="Dangerous command blocked",
                )
            )
        return ToolResult("run_command", False, "Refused dangerous command")
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")
        if len(out) > 40_000:
            out = out[:40_000] + "\n… [truncated]"
        ok = completed.returncode == 0
        
        if security:
            security.log_audit(
                AuditEntry(
                    timestamp=time.time(),
                    operation=OperationType.RUN_COMMAND,
                    risk_level=RiskLevel.MEDIUM,
                    command=command,
                    success=ok,
                    metadata={"returncode": completed.returncode, "output_length": len(out)},
                )
            )
        
        return ToolResult(
            "run_command",
            ok,
            f"exit {completed.returncode}",
            {"stdout": out, "returncode": completed.returncode},
        )
    except subprocess.TimeoutExpired:
        if security:
            security.log_audit(
                AuditEntry(
                    timestamp=time.time(),
                    operation=OperationType.RUN_COMMAND,
                    risk_level=RiskLevel.MEDIUM,
                    command=command,
                    success=False,
                    error=f"Timed out after {timeout}s",
                )
            )
        return ToolResult("run_command", False, f"Timed out after {timeout}s")
    except Exception as exc:
        if security:
            security.log_audit(
                AuditEntry(
                    timestamp=time.time(),
                    operation=OperationType.RUN_COMMAND,
                    risk_level=RiskLevel.MEDIUM,
                    command=command,
                    success=False,
                    error=str(exc),
                )
            )
        return ToolResult("run_command", False, str(exc))


def execute_tool_call(call: dict[str, Any], workspace: Path | None = None, security: SecurityManager | None = None) -> ToolResult:
    workspace = (workspace or Path.cwd()).resolve()
    norm = _normalize_call(call) or call
    name = str(norm.get("name") or "").strip()
    
    # Get security manager if not provided
    if security is None:
        security = get_security_manager()
    
    # Determine operation type and risk level
    operation_map = {
        "create_file": OperationType.CREATE_FILE,
        "edit_file": OperationType.EDIT_FILE,
        "delete_file": OperationType.DELETE_FILE,
        "read_file": OperationType.READ_FILE,
        "open_file": OperationType.OPEN_FILE,
        "list_dir": OperationType.LIST_DIR,
        "run_command": OperationType.RUN_COMMAND,
        "web_request": OperationType.WEB_REQUEST,
        "check_storage": OperationType.RUN_COMMAND,
    }
    operation = operation_map.get(name)
    
    # Request consent if required
    if operation and security.should_require_consent(operation):
        details = norm.get("path") or norm.get("command") or ""
        if not security.request_consent(operation, details):
            return ToolResult(name, False, "Operation declined by user")
    
    try:
        if name == "create_file":
            path = _safe_path(str(norm.get("path") or ""), workspace, security)
            return _create_file(path, str(norm.get("content") if norm.get("content") is not None else ""), security)
        if name == "edit_file":
            path = _safe_path(str(norm.get("path") or ""), workspace, security)
            return _edit_file(path, str(norm.get("old") or ""), str(norm.get("new") or ""), security)
        if name == "delete_file":
            path = _safe_path(str(norm.get("path") or ""), workspace, security)
            return _delete_file(path, security)
        if name == "read_file":
            path = _safe_path(str(norm.get("path") or ""), workspace, security)
            return _read_file(path, security=security)
        if name == "open_file":
            path = _safe_path(str(norm.get("path") or ""), workspace, security)
            return _open_file(path)
        if name == "list_dir":
            path = _safe_path(str(norm.get("path") or "."), workspace, security)
            return _list_dir(path)
        if name == "run_command":
            return _run_command(str(norm.get("command") or ""), workspace, security=security)
        if name == "web_request":
            return _web_request(
                str(norm.get("url") or ""),
                str(norm.get("method") or "GET"),
                norm.get("headers"),
                norm.get("body"),
                security,
            )
        if name == "check_storage":
            return _check_storage(security)
        return ToolResult(name or "unknown", False, f"Unknown tool: {name}")
    except PermissionError as exc:
        if security and operation:
            security.log_audit(
                AuditEntry(
                    timestamp=time.time(),
                    operation=operation,
                    risk_level=RiskLevel.HIGH,
                    path=norm.get("path"),
                    command=norm.get("command"),
                    success=False,
                    error=str(exc),
                )
            )
        return ToolResult(name, False, str(exc))
    except Exception as exc:
        if security and operation:
            security.log_audit(
                AuditEntry(
                    timestamp=time.time(),
                    operation=operation,
                    risk_level=RiskLevel.MEDIUM,
                    path=norm.get("path"),
                    command=norm.get("command"),
                    success=False,
                    error=str(exc),
                )
            )
        return ToolResult(name, False, f"Tool error: {exc}")


def execute_tool_calls(text: str, workspace: Path | None = None, security: SecurityManager | None = None) -> list[ToolResult]:
    if security is None:
        if workspace is not None:
            # An explicitly requested workspace is the sandbox root for this batch.
            security = SecurityManager(SecurityConfig(allowed_workspaces=[Path(workspace).resolve()]))
        else:
            security = get_security_manager()
    return [execute_tool_call(c, workspace, security) for c in parse_tool_calls(text)]


def format_tool_results(results: list[ToolResult]) -> str:
    parts: list[str] = []
    for r in results:
        status = "ok" if r.ok else "error"
        body = r.message
        if r.data and isinstance(r.data, dict):
            if "content" in r.data and r.data["content"]:
                body += f"\n{r.data['content']}"
            elif "entries" in r.data:
                lines = []
                for e in r.data["entries"][:200]:
                    mark = "/" if e.get("kind") == "dir" else ""
                    lines.append(f"  {e['name']}{mark}")
                body += "\n" + "\n".join(lines)
            elif "stdout" in r.data:
                body += f"\n{r.data['stdout']}"
        parts.append(f'<tool_result name="{r.name}" status="{status}">\n{body}\n</tool_result>')
    return "\n".join(parts)
