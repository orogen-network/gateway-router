"""Gateway FastAPI app.

Endpoints:
- `POST /v1/chat/completions` — OpenAI-compatible; routes to a worker.
- `POST /v1/nonces` — issue a fresh nonce challenge.
- `POST /internal/heartbeat` — operator heartbeat ingest.
- `GET  /internal/catalog` — current operator catalog.
- `POST /internal/seal_batch` — produce a `SettlementBatch` from buffered receipts.
- `GET  /healthz`

Security model:
- `/internal/*` routes are gated by a shared bearer (`INTERNAL_AUTH_TOKEN` env).
- `/v1/*` routes are gated by `PUBLIC_API_TOKENS` (csv env) when configured.
- Heartbeats are verified Ed25519-signed by the operator (pubkey from
  `OperatorRegistry`).
- Receipts proxied back by upstreams are verified against the routed operator's
  pubkey before being added to the settlement batch.
- Operator `endpoint_url` is validated against an SSRF allow-list before being
  stored or dereferenced.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from mining_types import OffChainHeartbeat, Receipt, verify_ed25519
from pydantic import BaseModel

from gateway_router.auth import (
    require_internal_auth,
    require_internal_token,
    require_public_auth,
    safe_httpx_client,
    validate_endpoint_url,
)
from gateway_router.batcher import BatchBuilder
from gateway_router.catalog import OperatorCatalog
from gateway_router.config import GatewayConfig
from gateway_router.nonces import NonceStore, NonceVault, RedisNonceVault
from gateway_router.registry import OperatorRegistry

# Cap on the size of the upstream JSON response we will buffer before rejecting
# (MED-SVC-011 — malicious upstream returning multi-GB body).
MAX_UPSTREAM_BYTES = 2 * 1024 * 1024  # 2 MiB


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 64
    seed: int = 0
    customer_nonce: str | None = None
    max_price: int | None = None
    region: str | None = None


def _allow_http_localhost() -> bool:
    """Permit http://127.0.0.1 endpoint URLs unless in production."""
    return os.environ.get("OROGEN_ENV", "").lower() != "production"


def _worker_auth_headers() -> dict[str, str]:
    token = (
        os.environ.get("WORKER_API_TOKEN", "")
        or os.environ.get("INTERNAL_AUTH_TOKEN", "")
    ).strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def build_app(config: GatewayConfig) -> FastAPI:
    # Fail-closed startup check for internal token in production.
    require_internal_token()
    env = os.environ.get("OROGEN_ENV", "").lower()
    nonce_backend = os.environ.get("GATEWAY_NONCE_BACKEND", "").lower()
    redis_url = (
        os.environ.get("GATEWAY_REDIS_URL", "")
        or os.environ.get("REDIS_URL", "")
    ).strip()
    if env == "production" and nonce_backend != "redis":
        raise RuntimeError("production gateway requires GATEWAY_NONCE_BACKEND=redis")
    if nonce_backend == "redis" and not redis_url:
        raise RuntimeError("GATEWAY_REDIS_URL or REDIS_URL is required for Redis nonce backend")

    app = FastAPI(title="gateway-router", version="0.1.0")
    # Trust only configured hosts; "*" is fine for dev, locked in prod.
    allowed_hosts = [
        h.strip()
        for h in os.environ.get("ALLOWED_HOSTS", "*").split(",")
        if h.strip()
    ] or ["*"]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    catalog = OperatorCatalog()
    vault: NonceStore
    if nonce_backend == "redis":
        vault = RedisNonceVault(
            gateway_id=config.gateway_id,
            ttl_ms=config.nonce_ttl_ms,
            redis_url=redis_url,
        )
    else:
        vault = NonceVault(gateway_id=config.gateway_id, ttl_ms=config.nonce_ttl_ms)
    batcher = BatchBuilder(
        gateway_id=config.gateway_id,
        gateway_private_key_hex=config.gateway_private_key(),
        epoch_number=config.epoch_number,
    )
    registry = OperatorRegistry.from_env()
    # Stash so tests can inspect / register pubkeys.
    app.state.catalog = catalog
    app.state.vault = vault
    app.state.batcher = batcher
    app.state.config = config
    app.state.registry = registry
    app.state.max_upstream_bytes = MAX_UPSTREAM_BYTES

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "gateway_id": config.gateway_id,
            "operators": len(catalog.all()),
            "batch_size": batcher.size,
        }

    @app.post("/internal/heartbeat", dependencies=[Depends(require_internal_auth)])
    async def heartbeat(hb: OffChainHeartbeat) -> dict[str, Any]:
        # Look up operator pubkey from registry.
        pubkey = registry.get(hb.operator_id)
        if pubkey is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"unknown operator {hb.operator_id!r}",
            )
        if not verify_ed25519(pubkey, hb.signing_payload(), hb.signature):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid operator signature",
            )
        # SSRF defence: validate the endpoint_url before storing it.
        if hb.endpoint_url:
            try:
                validate_endpoint_url(
                    hb.endpoint_url,
                    allow_http_localhost=_allow_http_localhost(),
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"endpoint_url rejected: {exc}",
                ) from exc
        catalog.upsert(hb)
        return {"ok": True, "operators": len(catalog.all())}

    @app.get("/internal/catalog", dependencies=[Depends(require_internal_auth)])
    async def get_catalog() -> dict[str, Any]:
        return {
            "operators": [
                {
                    "operator_id": r.operator_id,
                    "endpoint_url": r.endpoint_url,
                    "base_models": sorted(r.base_models),
                    "price_per_million": r.price_per_million,
                    "region": r.region,
                    "last_seen_ms": r.last_seen_ms,
                }
                for r in catalog.all()
            ]
        }

    @app.post("/v1/nonces", dependencies=[Depends(require_public_auth)])
    async def issue_nonce(request: Request) -> dict[str, Any]:
        # MED-SVC-010 hardening: reject oversized request bodies on a route
        # an attacker could flood to exhaust the nonce vault.
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > 1024:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="nonce request body too large",
                    )
            except ValueError:
                pass
        return vault.issue().model_dump(mode="json")

    @app.post("/internal/seal_batch", dependencies=[Depends(require_internal_auth)])
    async def seal_batch() -> dict[str, Any]:
        if batcher.size == 0:
            raise HTTPException(status_code=400, detail="no receipts buffered")
        sealed = batcher.seal()
        batcher.reset()
        return sealed.model_dump(mode="json")

    @app.post("/v1/chat/completions", dependencies=[Depends(require_public_auth)])
    async def chat_completions(req: ChatRequest) -> dict[str, Any]:
        rec = catalog.find(
            model_id=req.model, max_price=req.max_price, region=req.region,
        )
        if rec is None:
            raise HTTPException(
                status_code=503,
                detail=f"no operator advertising model {req.model!r}",
            )
        nonce = req.customer_nonce
        if nonce is None:
            nonce = vault.issue().nonce
        else:
            # If the customer brought a nonce, accept only if it came from us
            # and is still unused/unexpired.
            if not vault.is_known(nonce):
                raise HTTPException(
                    status_code=400,
                    detail="unknown, expired, or consumed nonce",
                )
        if not vault.claim(nonce):
            raise HTTPException(
                status_code=409,
                detail="nonce already consumed or expired",
            )

        upstream = {
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
            "max_tokens": req.max_tokens,
            "seed": req.seed,
            "customer_nonce": nonce,
        }
        if not rec.endpoint_url:
            raise HTTPException(status_code=503, detail="operator has no endpoint")
        # Re-validate at call-time (TOCTOU mitigation — the catalog row may have
        # been mutated by a later heartbeat).
        try:
            validate_endpoint_url(
                rec.endpoint_url,
                allow_http_localhost=_allow_http_localhost(),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=502, detail=f"operator endpoint rejected: {exc}",
            ) from exc

        async with safe_httpx_client(config.request_timeout_s) as client:
            try:
                upstream_resp = await client.post(
                    f"{rec.endpoint_url}/v1/chat/completions",
                    json=upstream,
                    headers=_worker_auth_headers(),
                )
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502, detail=f"operator unreachable: {exc!r}",
                ) from exc

        if upstream_resp.status_code != 200:
            raise HTTPException(
                status_code=502, detail=f"operator error {upstream_resp.status_code}",
            )
        # MED-SVC-011: cap upstream body size before parsing.
        raw = upstream_resp.content
        if len(raw) > app.state.max_upstream_bytes:
            raise HTTPException(
                status_code=502, detail="upstream response too large",
            )
        try:
            body = upstream_resp.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502, detail=f"upstream returned non-JSON: {exc!r}",
            ) from exc
        try:
            receipt = Receipt.model_validate(body["receipt"])
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"upstream receipt malformed: {exc!r}",
            ) from exc

        # MED-SVC-011 / CRIT-SVC-002: verify the receipt's operator_id matches
        # the operator we routed to AND that the receipt is signed by them.
        receipt_mismatch: dict[str, str] = {}
        if receipt.operator_id != rec.operator_id:
            receipt_mismatch["operator_id"] = (
                f"routed={rec.operator_id!r} got={receipt.operator_id!r}"
            )
        if receipt.customer_nonce != nonce:
            receipt_mismatch["customer_nonce"] = (
                f"expected={nonce!r} got={receipt.customer_nonce!r}"
            )
        if receipt.model_id != req.model:
            receipt_mismatch["model_id"] = (
                f"expected={req.model!r} got={receipt.model_id!r}"
            )
        if receipt.gateway_id != config.gateway_id:
            receipt_mismatch["gateway_id"] = (
                f"expected={config.gateway_id!r} got={receipt.gateway_id!r}"
            )
        if receipt_mismatch:
            raise HTTPException(
                status_code=502,
                detail={"receipt_mismatch": receipt_mismatch},
            )
        op_pubkey = registry.get(receipt.operator_id)
        if op_pubkey is None:
            raise HTTPException(
                status_code=502,
                detail=f"unknown operator {receipt.operator_id!r} in receipt",
            )
        if not verify_ed25519(
            op_pubkey, receipt.signing_payload(), receipt.operator_signature,
        ):
            raise HTTPException(
                status_code=502,
                detail="upstream receipt has invalid operator signature",
            )
        batcher.add(receipt, operator_pubkey=op_pubkey)
        body["gateway_id"] = config.gateway_id
        body["nonce_used"] = nonce
        return body

    return app
