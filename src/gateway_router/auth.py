"""Bearer-token authentication and SSRF hardening for gateway-router.

Two-tier model:
  - `/internal/*` routes — require `INTERNAL_AUTH_TOKEN` bearer.
  - Public routes (`/v1/*`) — require any non-empty bearer in `PUBLIC_API_TOKENS`
    (comma-separated env var). For dev/test we degrade to "no public auth" when
    `OROGEN_ENV != production` AND no tokens are configured.

In production (`OROGEN_ENV=production`), the service refuses to start without
`INTERNAL_AUTH_TOKEN`.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_BEARER = HTTPBearer(auto_error=False)


def _is_production() -> bool:
    return os.environ.get("OROGEN_ENV", "").lower() == "production"


def require_internal_token() -> str:
    """Read `INTERNAL_AUTH_TOKEN` from env; refuse to start in production if missing."""
    tok = os.environ.get("INTERNAL_AUTH_TOKEN", "").strip()
    if not tok:
        if _is_production():
            raise RuntimeError(
                "INTERNAL_AUTH_TOKEN must be set in production "
                "(OROGEN_ENV=production)"
            )
    return tok


def _public_tokens() -> set[str]:
    raw = os.environ.get("PUBLIC_API_TOKENS", "")
    return {t.strip() for t in raw.split(",") if t.strip()}


async def require_internal_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_BEARER),
) -> None:
    """Dependency: gate /internal/* routes by a shared bearer token."""
    expected = require_internal_token()
    if not expected:
        # Dev/test default — explicitly opt-out by leaving env unset.
        return
    if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="internal auth required",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_public_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_BEARER),
) -> None:
    """Dependency: gate public /v1/* routes by a bearer in PUBLIC_API_TOKENS."""
    tokens = _public_tokens()
    if not tokens:
        # No public tokens configured.
        if _is_production():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="public api token not configured",
            )
        return
    if creds is None or creds.scheme.lower() != "bearer" or creds.credentials not in tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# SSRF defence
# ---------------------------------------------------------------------------

_FORBIDDEN_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "metadata.azure.com",
    "metadata",
    "169.254.169.254",  # AWS / GCP / DO metadata
    "100.100.100.200",  # Alibaba metadata
}


def _ip_is_forbidden(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_endpoint_url(url: str, *, allow_http_localhost: bool = False) -> None:
    """Raise ValueError if `url` is unsafe to dereference from this service.

    Defense:
      - Scheme must be https; http is only allowed for explicit 127.0.0.1 when
        `allow_http_localhost=True` (tests use this).
      - Hostname must not match common metadata names.
      - All resolved IP addresses must not be loopback/private/link-local/etc.
      - No credentials in the URL.
    """
    if not url:
        raise ValueError("empty url")
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"invalid url: {exc!r}") from exc
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"scheme not allowed: {parsed.scheme!r}")
    if parsed.username or parsed.password:
        raise ValueError("url credentials not allowed")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("url has no host")
    if parsed.scheme == "http":
        if not (allow_http_localhost and host in ("127.0.0.1", "::1", "localhost")):
            raise ValueError("only https:// urls are allowed")
    if host in _FORBIDDEN_HOSTNAMES:
        raise ValueError(f"forbidden hostname: {host!r}")

    # Resolve DNS and reject any forbidden IP.
    candidates: list[ipaddress._BaseAddress] = []
    # Direct literal IP?
    try:
        candidates.append(ipaddress.ip_address(host))
    except ValueError:
        try:
            for _fam, *_rest, sockaddr in socket.getaddrinfo(host, None):
                addr = sockaddr[0]
                try:
                    candidates.append(ipaddress.ip_address(addr))
                except ValueError:
                    continue
        except socket.gaierror as exc:
            raise ValueError(f"hostname did not resolve: {exc!r}") from exc
    if not candidates:
        raise ValueError("hostname resolved to no addresses")
    for ip in candidates:
        if _ip_is_forbidden(ip):
            # Allow loopback specifically for http://127.0.0.1 dev path.
            if (
                allow_http_localhost
                and parsed.scheme == "http"
                and ip.is_loopback
            ):
                continue
            raise ValueError(f"forbidden ip address: {ip}")


def safe_httpx_client(timeout_s: float) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient that won't follow redirects and has bounded pools."""
    return httpx.AsyncClient(
        timeout=timeout_s,
        follow_redirects=False,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )
