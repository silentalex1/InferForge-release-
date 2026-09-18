"""Security module for responsible AI operations with scoped access and audit logging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console(force_terminal=True, stderr=True)


class OperationType(Enum):
    """Types of operations for audit logging."""
    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    DELETE_FILE = "delete_file"
    READ_FILE = "read_file"
    OPEN_FILE = "open_file"
    LIST_DIR = "list_dir"
    RUN_COMMAND = "run_command"
    WEB_REQUEST = "web_request"


class RiskLevel(Enum):
    """Risk levels for operations."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Security configuration for scoped access."""
    allowed_workspaces: list[Path] = field(default_factory=list)
    allow_web_access: bool = False
    allowed_web_domains: set[str] = field(default_factory=set)
    web_rate_limit: int = 60  # requests per minute
    require_consent_for_delete: bool = True
    require_consent_for_edit: bool = False
    require_consent_for_command: bool = True
    enable_audit_log: bool = True
    audit_log_path: Path = field(default_factory=lambda: Path.home() / ".inferforge" / "audit.log")
    enable_backups: bool = True
    backup_dir: Path = field(default_factory=lambda: Path.home() / ".inferforge" / "backups")
    max_backup_size_mb: int = 100
    unrestricted_access: bool = False  # Allow access to any directory on system


@dataclass
class AuditEntry:
    """Audit log entry for tracking operations."""
    timestamp: float
    operation: OperationType
    risk_level: RiskLevel
    path: str | None = None
    command: str | None = None
    success: bool = True
    error: str | None = None
    user_consent: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "operation": self.operation.value,
            "risk_level": self.risk_level.value,
            "path": self.path,
            "command": self.command,
            "success": self.success,
            "error": self.error,
            "user_consent": self.user_consent,
            "metadata": self.metadata,
        }


class SecurityManager:
    """Manages security policies, audit logging, and access control."""
    
    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self._audit_log: list[AuditEntry] = []
        self._web_request_times: list[float] = []
        self._backup_count = 0
        
        # Initialize directories
        self.config.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Add current directory as default workspace if none specified
        if not self.config.allowed_workspaces:
            self.config.allowed_workspaces = [Path.cwd()]
    
    def get_risk_level(self, operation: OperationType, path: str | None = None) -> RiskLevel:
        """Determine risk level for an operation."""
        if operation == OperationType.DELETE_FILE:
            return RiskLevel.HIGH
        if operation == OperationType.RUN_COMMAND:
            return RiskLevel.MEDIUM
        if operation == OperationType.EDIT_FILE:
            return RiskLevel.LOW
        if operation in {OperationType.READ_FILE, OperationType.LIST_DIR, OperationType.OPEN_FILE}:
            return RiskLevel.SAFE
        if operation == OperationType.CREATE_FILE:
            return RiskLevel.LOW
        if operation == OperationType.WEB_REQUEST:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    
    def check_workspace_access(self, path: Path) -> bool:
        """Check if path is within allowed workspaces."""
        if self.config.unrestricted_access:
            return True
        resolved = path.resolve()
        for workspace in self.config.allowed_workspaces:
            try:
                resolved.relative_to(workspace.resolve())
                return True
            except ValueError:
                continue
        return False
    
    def check_web_rate_limit(self) -> bool:
        """Check if web request is within rate limit."""
        now = time.time()
        # Remove requests older than 1 minute
        self._web_request_times = [t for t in self._web_request_times if now - t < 60]
        
        if len(self._web_request_times) >= self.config.web_rate_limit:
            return False
        
        self._web_request_times.append(now)
        return True
    
    def should_require_consent(self, operation: OperationType) -> bool:
        """Check if operation requires user consent."""
        if operation == OperationType.DELETE_FILE and self.config.require_consent_for_delete:
            return True
        if operation == OperationType.EDIT_FILE and self.config.require_consent_for_edit:
            return True
        if operation == OperationType.RUN_COMMAND and self.config.require_consent_for_command:
            return True
        return False
    
    def request_consent(self, operation: OperationType, details: str) -> bool:
        """Request user consent for an operation."""
        risk = self.get_risk_level(operation)
        console.print()
        console.print(
            f"[bold yellow]⚠ Consent Required[/] "
            f"[dim]({risk.value.upper()})[/]"
        )
        console.print(f"[cyan]{operation.value}:[/] {details}")
        
        try:
            response = console.input("[bold]Approve? [y/N]: [/]").strip().lower()
            return response in {"y", "yes"}
        except (KeyboardInterrupt, EOFError):
            return False
    
    def log_audit(self, entry: AuditEntry) -> None:
        """Log operation to audit trail."""
        if not self.config.enable_audit_log:
            return
        
        self._audit_log.append(entry)
        
        # Write to file
        try:
            with open(self.config.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            console.print(f"[red]Failed to write audit log:[/] {e}")
    
    def create_backup(self, path: Path) -> Path | None:
        """Create backup of a file before modification."""
        if not self.config.enable_backups:
            return None
        
        try:
            if not path.exists() or path.is_dir():
                return None
            
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.max_backup_size_mb:
                console.print(f"[yellow]Skipping backup (file too large):[/] {size_mb:.1f}MB")
                return None
            
            self._backup_count += 1
            backup_name = f"{path.name}.bak.{self._backup_count}.{int(time.time())}"
            backup_path = self.config.backup_dir / backup_name
            
            backup_path.write_bytes(path.read_bytes())
            return backup_path
        except Exception as e:
            console.print(f"[yellow]Backup failed:[/] {e}")
            return None
    
    def restore_backup(self, backup_path: Path, original_path: Path) -> bool:
        """Restore a file from backup."""
        try:
            if not backup_path.exists():
                return False
            
            original_path.parent.mkdir(parents=True, exist_ok=True)
            original_path.write_bytes(backup_path.read_bytes())
            return True
        except Exception as e:
            console.print(f"[red]Restore failed:[/] {e}")
            return False
    
    def get_audit_summary(self, last_n: int = 50) -> list[dict[str, Any]]:
        """Get summary of recent audit entries."""
        return [entry.to_dict() for entry in self._audit_log[-last_n:]]


# Global security manager instance
_security_manager: SecurityManager | None = None


def get_security_manager(config: SecurityConfig | None = None) -> SecurityManager:
    """Get or create the global security manager."""
    global _security_manager
    if _security_manager is None or config is not None:
        _security_manager = SecurityManager(config)
    return _security_manager


def reset_security_manager() -> None:
    """Reset the global security manager (mainly for testing)."""
    global _security_manager
    _security_manager = None
