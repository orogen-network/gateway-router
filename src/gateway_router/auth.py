"""Bearer-token authentication and SSRF hardening for gateway-router.

Two-tier model:
  - `/internal/*` routes — require `INTERNAL_AUTH_TOKEN` bearer.
  - Public routes (`/v1/*`) — require any non-empty bearer in `PUBLIC_API_TOKENS`
    (comma-separated env var) OR a self-issued testnet key minted by
    `POST /v1/keys`. For dev/test we degrade to "no public auth" when
    `OROGEN_ENV != production` AND no tokens are configured.

Self-serve testnet keys (`POST /v1/keys`) are stored in Redis when the gateway
is configured with one (keyed under `gateway:testnet_keys`), else in an
in-process set. They are clearly testnet-scoped (`orogen-testnet-<random>`).

In production (`OROGEN_ENV=production`), the service refuses to start without
`INTERNAL_AUTH_TOKEN`.
"""

from __future__ import annotations

import ipaddress
import os
import secrets
import socket
import time
from urllib.parse import urlparse

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_BEARER = HTTPBearer(auto_error=False)

# Prefix for self-serve testnet API keys so they are unmistakably testnet-scoped.
TESTNET_KEY_PREFIX = "orogen-testnet-"

# Redis key (a set) holding self-issued testnet API keys when a Redis backend is
# configured. In-process keys live in `_INPROC_TESTNET_KEYS` instead.
_REDIS_TESTNET_KEYS = "gateway:testnet_keys"

# Fallback in-process store for self-issued testnet keys (used when no Redis is
# configured). Process-local, lost on restart — fine for a testnet bring-up.
_INPROC_TESTNET_KEYS: set[str] = set()


def _redis_url() -> str:
    return (
        os.environ.get("GATEWAY_REDIS_URL", "")
        or os.environ.get("REDIS_URL", "")
    ).strip()


def _testnet_key_store():
    """Return a Redis client if one is configured, else None (in-process store).

    Returning None signals callers to fall back to `_INPROC_TESTNET_KEYS`.
    """
    url = _redis_url()
    if not url:
        return None
    try:
        import redis  # local import: only needed when Redis is configured
    except ImportError:
        return None
    return redis.Redis.from_url(url)


def mint_testnet_key() -> str:
    """Mint, store, and return a fresh self-serve testnet API key."""
    key = f"{TESTNET_KEY_PREFIX}{secrets.token_urlsafe(24)}"
    client = _testnet_key_store()
    if client is not None:
        client.sadd(_REDIS_TESTNET_KEYS, key)
    else:
        _INPROC_TESTNET_KEYS.add(key)
    return key


def _is_self_issued_key(token: str) -> bool:
    if not token or not token.startswith(TESTNET_KEY_PREFIX):
        return False
    client = _testnet_key_store()
    if client is not None:
        try:
            return bool(client.sismember(_REDIS_TESTNET_KEYS, token))
        except Exception:
            return False
    return token in _INPROC_TESTNET_KEYS


# ---------------------------------------------------------------------------
# Self-serve key issuance rate limiting (per source IP, in-process).
# ---------------------------------------------------------------------------

# Simple fixed-window in-process limiter for the unauthenticated /v1/keys mint
# endpoint. Bounds how many keys a single source IP can mint per window so the
# open faucet can't be trivially abused. Testnet-scoped; not a distributed limit.
_KEY_MINT_WINDOW_S = 3600.0
_KEY_MINT_MAX_PER_WINDOW = 20
_key_mint_hits: dict[str, list[float]] = {}


def key_mint_rate_limit_ok(source_ip: str) -> bool:
    """Return True if `source_ip` may mint another key in the current window."""
    now = time.monotonic()
    hits = [t for t in _key_mint_hits.get(source_ip, []) if now - t < _KEY_MINT_WINDOW_S]
    if len(hits) >= _KEY_MINT_MAX_PER_WINDOW:
        _key_mint_hits[source_ip] = hits
        return False
    hits.append(now)
    _key_mint_hits[source_ip] = hits
    return True


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
    """Dependency: gate public /v1/* routes.

    Accepts either a static bearer in `PUBLIC_API_TOKENS` OR a self-issued
    testnet key minted via `POST /v1/keys`.
    """
    tokens = _public_tokens()
    presented = creds.credentials if (creds and creds.scheme.lower() == "bearer") else None
    if presented is not None and (presented in tokens or _is_self_issued_key(presented)):
        return
    if not tokens:
        # No static public tokens configured. A self-issued key (checked above)
        # is still always accepted; otherwise fall back to env policy.
        if _is_production():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="public api token not configured",
            )
        return
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


def validate_endpoint_url(
    url: str,
    *,
    allow_http_localhost: bool = False,
    allow_insecure_http: bool = False,
) -> None:
    """Raise ValueError if `url` is unsafe to dereference from this service.

    Defense:
      - Scheme must be https; http is only allowed for explicit 127.0.0.1 when
        `allow_http_localhost=True` (tests use this), OR for any non-forbidden
        host when `allow_insecure_http=True` (the testnet
        `GATEWAY_ALLOW_INSECURE_OPERATOR_ENDPOINTS` flag — still SSRF-checked).
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
        is_localhost = allow_http_localhost and host in ("127.0.0.1", "::1", "localhost")
        if not (is_localhost or allow_insecure_http):
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


def allow_insecure_operator_endpoints() -> bool:
    """Testnet flag: route to plain-http operator workers without TLS verification.

    Default false. When true (`GATEWAY_ALLOW_INSECURE_OPERATOR_ENDPOINTS`), the
    gateway's outbound client to operator workers permits http:// endpoints and
    does NOT verify TLS. Intended ONLY for testnet operators serving plain http.
    """
    return os.environ.get("GATEWAY_ALLOW_INSECURE_OPERATOR_ENDPOINTS", "").lower() in {
        "1",
        "true",
        "yes",
    }


def safe_httpx_client(timeout_s: float) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient that won't follow redirects and has bounded pools.

    When the testnet `GATEWAY_ALLOW_INSECURE_OPERATOR_ENDPOINTS` flag is set, the
    client disables TLS certificate verification so the gateway can reach a
    testnet operator worker serving http/self-signed TLS. Default verifies TLS.
    """
    return httpx.AsyncClient(
        timeout=timeout_s,
        follow_redirects=False,
        verify=not allow_insecure_operator_endpoints(),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )
