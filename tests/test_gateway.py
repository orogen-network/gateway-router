"""Gateway router tests."""

from __future__ import annotations

import time

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from mining_types import (
    AttestationFreshness,
    Capability,
    LoadSnapshot,
    OffChainHeartbeat,
    Quantization,
    Receipt,
    WatchdogState,
    generate_keypair,
)

from gateway_router import GatewayConfig, build_app
from gateway_router.batcher import BatchBuilder
from gateway_router.catalog import OperatorCatalog
from gateway_router.nonces import NonceVault

INTERNAL_TOKEN = "test-internal-token"
PUBLIC_TOKEN = "test-public-token"


@pytest.fixture(autouse=True)
def auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERNAL_AUTH_TOKEN", INTERNAL_TOKEN)
    monkeypatch.setenv("PUBLIC_API_TOKENS", PUBLIC_TOKEN)
    # Make sure we're not in production mode (allows http:// localhost endpoints).
    monkeypatch.delenv("OROGEN_ENV", raising=False)


def _internal_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {INTERNAL_TOKEN}"}


def _public_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {PUBLIC_TOKEN}"}


@pytest.fixture
def config() -> GatewayConfig:
    priv, _ = generate_keypair()
    return GatewayConfig(gateway_id="gw-test", gateway_private_key_hex=priv)


def _hb_with_keys(
    operator_id: str,
    endpoint: str,
    model: str = "mock-model-7b",
) -> tuple[OffChainHeartbeat, str]:
    """Build + sign a heartbeat. Returns (heartbeat, operator_pubkey_hex)."""
    priv, pub = generate_keypair()
    hb = OffChainHeartbeat(
        operator_id=operator_id,
        capabilities=[Capability(base_model_id=model, quantization=Quantization.FP16)],
        current_load=LoadSnapshot(),
        attestation_freshness=AttestationFreshness(
            last_attested_at_ms=int(time.time() * 1000),
            expires_at_ms=int(time.time() * 1000) + 86400000,
            current_report_hash="ab" * 32,
        ),
        watchdog_state=WatchdogState(),
        endpoint_url=endpoint,
        price_per_million_tokens=1000,
        geo_region="US",
    ).sign(priv)
    return hb, pub


def _hb(operator_id: str, endpoint: str, model: str = "mock-model-7b") -> OffChainHeartbeat:
    hb, _pub = _hb_with_keys(operator_id, endpoint, model)
    return hb


def test_catalog_filters() -> None:
    c = OperatorCatalog()
    c.upsert(_hb("op-1", "http://a", "mock-model-7b"))
    c.upsert(_hb("op-2", "http://b", "other-model"))
    r = c.find(model_id="mock-model-7b")
    assert r and r.operator_id == "op-1"
    assert c.find(model_id="missing") is None


def test_nonce_vault_replay_protection() -> None:
    v = NonceVault("gw-1")
    n = v.issue()
    assert v.claim(n.nonce)
    assert not v.claim(n.nonce)  # double-claim rejected


def test_batch_builder_signs_and_resets() -> None:
    priv, pub = generate_keypair()
    b = BatchBuilder("gw", priv, 1)
    for i in range(3):
        unsigned = Receipt(
            job_id=str(i), operator_id="op", model_id="m",
            model_weight_hash="w", customer_nonce="n",
            request_hash="rq", response_hash="rs",
            kernel_pack_hash="k", attestation_report_hash="a",
            timestamp_ms=i, gateway_id="gw",
        )
        # Sign each receipt with a key; pass pubkey so the batcher verifies.
        op_priv, op_pub = generate_keypair()
        signed = unsigned.sign(op_priv)
        b.add(signed, operator_pubkey=op_pub)
    sealed = b.seal()
    assert sealed.receipt_count == 3
    assert sealed.gateway_signature
    assert len(sealed.per_operator_summary) == 1


def test_batch_builder_rejects_bad_signature() -> None:
    priv, _pub = generate_keypair()
    b = BatchBuilder("gw", priv, 1)
    unsigned = Receipt(
        job_id="x", operator_id="op", model_id="m",
        model_weight_hash="w", customer_nonce="n",
        request_hash="rq", response_hash="rs",
        kernel_pack_hash="k", attestation_report_hash="a",
        timestamp_ms=1, gateway_id="gw",
        operator_signature="00" * 64,  # not a real signature
    )
    _op_priv, op_pub = generate_keypair()
    with pytest.raises(ValueError, match="invalid operator signature"):
        b.add(unsigned, operator_pubkey=op_pub)


def test_healthz(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["gateway_id"] == "gw-test"


def test_internal_routes_require_auth(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post("/internal/heartbeat", json={})
        assert r.status_code == 401
        r2 = client.get("/internal/catalog")
        assert r2.status_code == 401


def test_public_routes_require_auth(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        # public token configured → reject anonymous.
        r = client.post("/v1/nonces")
        assert r.status_code == 401


def test_heartbeat_then_catalog(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        hb, pub = _hb_with_keys("op-1", "http://127.0.0.1:65535", "mock-model-7b")
        app.state.registry.register("op-1", pub)
        r = client.post(
            "/internal/heartbeat",
            json=hb.model_dump(mode="json"),
            headers=_internal_headers(),
        )
        assert r.status_code == 200, r.text
        r2 = client.get("/internal/catalog", headers=_internal_headers())
        assert r2.status_code == 200
        ops = r2.json()["operators"]
        assert len(ops) == 1
        assert ops[0]["operator_id"] == "op-1"


def test_heartbeat_rejects_unknown_operator(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        hb, _ = _hb_with_keys("rogue", "http://127.0.0.1:65535")
        r = client.post(
            "/internal/heartbeat",
            json=hb.model_dump(mode="json"),
            headers=_internal_headers(),
        )
        assert r.status_code == 401


def test_heartbeat_rejects_bad_signature(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        hb, pub = _hb_with_keys("op-1", "http://127.0.0.1:65535")
        # Register a *different* pubkey so the signature fails.
        _, wrong_pub = generate_keypair()
        app.state.registry.register("op-1", wrong_pub)
        r = client.post(
            "/internal/heartbeat",
            json=hb.model_dump(mode="json"),
            headers=_internal_headers(),
        )
        assert r.status_code == 401


def test_heartbeat_rejects_ssrf_endpoint(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        hb, pub = _hb_with_keys("op-1", "http://169.254.169.254/latest/meta-data")
        app.state.registry.register("op-1", pub)
        r = client.post(
            "/internal/heartbeat",
            json=hb.model_dump(mode="json"),
            headers=_internal_headers(),
        )
        assert r.status_code == 400
        assert "endpoint_url" in r.text


def test_chat_routes_to_mocked_worker(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        # register an operator + pubkey
        priv, pub = generate_keypair()
        hb = OffChainHeartbeat(
            operator_id="op-1",
            capabilities=[Capability(base_model_id="mock-model-7b")],
            current_load=LoadSnapshot(),
            attestation_freshness=AttestationFreshness(
                last_attested_at_ms=int(time.time() * 1000),
                expires_at_ms=int(time.time() * 1000) + 86400000,
                current_report_hash="ab" * 32,
            ),
            watchdog_state=WatchdogState(),
            endpoint_url="http://127.0.0.1:65535",
            price_per_million_tokens=1000,
            geo_region="US",
        ).sign(priv)
        app.state.registry.register("op-1", pub)
        client.post(
            "/internal/heartbeat",
            json=hb.model_dump(mode="json"),
            headers=_internal_headers(),
        )
        # Build a properly-signed receipt that the upstream will return.
        signed_receipt = Receipt(
            version=1,
            job_id="j-1",
            operator_id="op-1",
            model_id="mock-model-7b",
            model_weight_hash="w",
            customer_nonce="n",
            request_hash="rq",
            response_hash="rs",
            log_probs_sample=[-0.1, -0.2],
            kernel_pack_hash="k",
            gpu_model="mock-H100",
            driver_version="550.54",
            cuda_version="12.4",
            attestation_report_hash="a",
            batch_invariant_proof=None,
            timestamp_ms=1,
            gateway_id="gw-test",
        ).sign(priv)
        upstream_body = {
            "id": "j-1",
            "object": "chat.completion",
            "model": "mock-model-7b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "receipt": signed_receipt.model_dump(mode="json"),
        }
        with respx.mock(assert_all_called=True) as mocker:
            mocker.post("http://127.0.0.1:65535/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=upstream_body),
            )
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model-7b",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers=_public_headers(),
            )
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["message"]["content"] == "hello"
        # batch should now have 1 receipt
        r2 = client.post("/internal/seal_batch", headers=_internal_headers())
        assert r2.status_code == 200
        assert r2.json()["receipt_count"] == 1
