"""Authentication and rate limiting for InferForge API."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from platformdirs import user_config_dir


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.rate = requests_per_minute / 60.0  # requests per second
        self.burst = burst
        self.tokens: dict[str, float] = defaultdict(lambda: self.burst)
        self.last_update: dict[str, float] = defaultdict(lambda: time.time())
    
    def allow_request(self, key: str) -> tuple[bool, dict[str, Any]]:
        """Check if request is allowed. Returns (allowed, headers)."""
        now = time.time()
        
        # Refill tokens based on time elapsed
        elapsed = now - self.last_update[key]
        self.tokens[key] = min(self.burst, self.tokens[key] + elapsed * self.rate)
        self.last_update[key] = now
        
        # Check if we have tokens
        if self.tokens[key] >= 1.0:
            self.tokens[key] -= 1.0
            remaining = int(self.tokens[key])
            reset_time = int(now + (self.burst - self.tokens[key]) / self.rate)
            
            return True, {
                "X-RateLimit-Limit": str(int(self.rate * 60)),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_time)
            }
        else:
            retry_after = int((1.0 - self.tokens[key]) / self.rate)
            return False, {
                "X-RateLimit-Limit": str(int(self.rate * 60)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(now + retry_after)),
                "Retry-After": str(retry_after)
            }


class APIKeyManager:
    """Manage API keys for authentication."""
    
    def __init__(self):
        self.config_dir = Path(user_config_dir("inferforge"))
        self.keys_file = self.config_dir / "api_keys.txt"
        self.keys: set[str] = set()
        self._load_keys()
    
    def _load_keys(self) -> None:
        """Load API keys from config file."""
        if self.keys_file.exists():
            with open(self.keys_file, 'r') as f:
                for line in f:
                    key = line.strip()
                    if key and not key.startswith('#'):
                        self.keys.add(key)
        else:
            # Create default key on first run
            self._create_default_key()
    
    def _create_default_key(self) -> None:
        """Create a default API key."""
        import secrets
        default_key = f"sk-{secrets.token_urlsafe(32)}"
        self.keys.add(default_key)
        self._save_keys()
        print(f"Created default API key: {default_key}")
        print(f"Save this key! It's stored in: {self.keys_file}")
    
    def _save_keys(self) -> None:
        """Save API keys to config file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.keys_file, 'w') as f:
            f.write("# InferForge API Keys\n")
            f.write("# One key per line. Lines starting with # are ignored.\n")
            f.write("# Generate new keys with: forge api-key create\n\n")
            for key in sorted(self.keys):
                f.write(f"{key}\n")
    
    def validate_key(self, key: str) -> bool:
        """Validate an API key."""
        if not self.keys:
            # If no keys configured, allow all requests (development mode)
            return True
        return key in self.keys
    
    def create_key(self, name: str = "") -> str:
        """Create a new API key."""
        import secrets
        key = f"sk-{secrets.token_urlsafe(32)}"
        self.keys.add(key)
        self._save_keys()
        return key
    
    def revoke_key(self, key: str) -> bool:
        """Revoke an API key."""
        if key in self.keys:
            self.keys.remove(key)
            self._save_keys()
            return True
        return False
    
    def list_keys(self) -> list[str]:
        """List all API keys (truncated for security)."""
        return [f"{key[:20]}...{key[-4:]}" for key in sorted(self.keys)]


# Global instances
_rate_limiter = RateLimiter(requests_per_minute=60, burst=10)
_api_key_manager = APIKeyManager()
security = HTTPBearer(auto_error=False)


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    return _rate_limiter


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance."""
    return _api_key_manager


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(security)
) -> str:
    """Verify API key from Authorization header."""
    
    # Extract API key
    api_key = None
    
    if credentials:
        api_key = credentials.credentials
    elif "x-api-key" in request.headers:
        api_key = request.headers["x-api-key"]
    elif "api-key" in request.headers:
        api_key = request.headers["api-key"]
    
    # Validate key
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include Authorization: Bearer <key> or X-API-Key: <key> header"
        )
    
    manager = get_api_key_manager()
    if not manager.validate_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key


async def check_rate_limit(request: Request) -> None:
    """Check rate limit for request."""
    # Use IP address as rate limit key
    client_ip = request.client.host if request.client else "unknown"
    
    # Also consider API key if present
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        rate_key = auth_header[7:20]  # Use first part of API key
    else:
        rate_key = client_ip
    
    limiter = get_rate_limiter()
    allowed, headers = limiter.allow_request(rate_key)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers=headers
        )
    
    # Add rate limit headers to response (FastAPI will handle this)
    request.state.rate_limit_headers = headers
