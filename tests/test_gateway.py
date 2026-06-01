"""Gateway router tests."""

from __future__ import annotations

import hashlib
import json
import os
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

import gateway_router.app as gateway_app
from gateway_router import GatewayConfig, build_app
from gateway_router.batcher import BatchBuilder
from gateway_router.catalog import OperatorCatalog
from gateway_router.nonces import NonceVault
from gateway_router.operator_auth import (
    DOMAIN_HEARTBEAT,
    decode_ss58,
    encode_ss58,
    verify_heartbeat_signature,
)

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


def test_production_requires_redis_nonce_backend(
    monkeypatch: pytest.MonkeyPatch, config: GatewayConfig,
) -> None:
    monkeypatch.setenv("OROGEN_ENV", "production")
    with pytest.raises(RuntimeError, match="GATEWAY_NONCE_BACKEND=redis"):
        build_app(config)


def test_production_wires_redis_nonce_backend(
    monkeypatch: pytest.MonkeyPatch, config: GatewayConfig,
) -> None:
    class FakeRedisNonceVault(NonceVault):
        seen: list[tuple[str, str, int]] = []

        def __init__(self, gateway_id: str, redis_url: str, ttl_ms: int) -> None:
            self.seen.append((gateway_id, redis_url, ttl_ms))
            super().__init__(gateway_id=gateway_id, ttl_ms=ttl_ms)

    monkeypatch.setenv("OROGEN_ENV", "production")
    monkeypatch.setenv("GATEWAY_NONCE_BACKEND", "redis")
    monkeypatch.setenv("GATEWAY_REDIS_URL", "redis://redis.test:6379/0")
    monkeypatch.setattr(gateway_app, "RedisNonceVault", FakeRedisNonceVault)
    app = build_app(config)
    assert isinstance(app.state.vault, FakeRedisNonceVault)
    assert FakeRedisNonceVault.seen == [
        ("gw-test", "redis://redis.test:6379/0", config.nonce_ttl_ms)
    ]


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
        def _upstream(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == f"Bearer {INTERNAL_TOKEN}"
            upstream_req = json.loads(request.content)
            signed_receipt = Receipt(
                version=1,
                job_id="j-1",
                operator_id="op-1",
                model_id=upstream_req["model"],
                model_weight_hash="w",
                customer_nonce=upstream_req["customer_nonce"],
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
            return httpx.Response(200, json=upstream_body)

        with respx.mock(assert_all_called=True) as mocker:
            mocker.post("http://127.0.0.1:65535/v1/chat/completions").mock(
                side_effect=_upstream,
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


def test_chat_rejects_consumed_nonce_before_routing(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
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
        issued = client.post("/v1/nonces", headers=_public_headers()).json()["nonce"]
        assert app.state.vault.claim(issued)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model-7b",
                "messages": [{"role": "user", "content": "hi"}],
                "customer_nonce": issued,
            },
            headers=_public_headers(),
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Per-operator sr25519 heartbeat (public, secret-free path)
# ---------------------------------------------------------------------------

# Cross-language test vector generated by wallet-sdk-core (the source of truth):
# mnemonic "abandon abandon ... about", DOMAIN_HEARTBEAT, sr25519 over
# blake2b-512(domain || body). Guards against any drift in the verify scheme.
_RUST_PUBKEY_HEX = "66933bd1f37070ef87bd1198af3dacceb095237f803f3d32b173e6b425ed7972"
_RUST_SS58 = "5EPCUjPxiHAcNooYipQFWr9NmmXJKpNG5RhcntXwbtUySrgH"
_RUST_BODY = (
    '{"version":1,"operator_ss58":"5EPCUjPxiHAcNooYipQFWr9NmmXJKpNG5RhcntXwbtUySrgH",'
    '"timestamp_ms":1700000000000,"gpu_model":"H100","free_kv_blocks":42}'
)
_RUST_SIG_HEX = (
    "321622403f6e9a2c4022509182caef63ea58163214f67cc4066046adff35d34d"
    "0887709ecae917966f625b6225637bebffd3762c2657dafaa96d2913e29d748a"
)


def _new_operator() -> tuple[bytes, bytes, str]:
    """Return a fresh sr25519 (pub, priv) pair and its ss58 address."""
    import sr25519

    pub, priv = sr25519.pair_from_seed(os.urandom(32))
    return pub, priv, encode_ss58(pub)


def _sign_heartbeat_body(pub: bytes, priv: bytes, body: str) -> str:
    import sr25519

    digest = hashlib.blake2b(DOMAIN_HEARTBEAT + body.encode(), digest_size=64).digest()
    return "0x" + sr25519.sign((pub, priv), digest).hex()


def test_ss58_roundtrip_and_rust_vector() -> None:
    # The encoder/decoder agree with the wallet-sdk-core ss58 (prefix 42).
    pub = bytes.fromhex(_RUST_PUBKEY_HEX)
    assert encode_ss58(pub) == _RUST_SS58
    assert decode_ss58(_RUST_SS58) == pub


def test_verify_heartbeat_signature_matches_rust() -> None:
    pub = verify_heartbeat_signature(_RUST_SS58, _RUST_BODY.encode(), _RUST_SIG_HEX)
    assert pub.hex() == _RUST_PUBKEY_HEX


def _operator_body(ss58: str, **extra: object) -> str:
    payload = {"version": 1, "operator_ss58": ss58}
    payload.update(extra)
    return json.dumps(payload)


def test_operator_heartbeat_enters_catalog(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        pub, priv, ss58 = _new_operator()
        body = _operator_body(
            ss58,
            endpoint_url="http://127.0.0.1:65535",
            models=["mock-model-7b"],
            price_per_million_tokens=1500,
            geo_region="US",
        )
        sig = _sign_heartbeat_body(pub, priv, body)
        r = client.post(
            "/v1/operator/heartbeat",
            json={"heartbeat_json": body, "signature": sig},
        )
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["ok"] is True
        assert out["operator_id"] == ss58
        assert out["advertised_models"] == ["mock-model-7b"]
        # No shared internal token was used — and the catalog now has the operator.
        cat = client.get("/internal/catalog", headers=_internal_headers()).json()
        assert any(o["operator_id"] == ss58 for o in cat["operators"])


def test_operator_heartbeat_rejects_bad_signature(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        _pub, _priv, ss58 = _new_operator()
        body = _operator_body(ss58)
        r = client.post(
            "/v1/operator/heartbeat",
            json={"heartbeat_json": body, "signature": "0x" + "00" * 64},
        )
        assert r.status_code == 401


def test_operator_heartbeat_rejects_tampered_body(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        pub, priv, ss58 = _new_operator()
        signed_body = _operator_body(ss58, models=["a"])
        sig = _sign_heartbeat_body(pub, priv, signed_body)
        # Submit a DIFFERENT body than the one signed.
        tampered = _operator_body(ss58, models=["b"])
        r = client.post(
            "/v1/operator/heartbeat",
            json={"heartbeat_json": tampered, "signature": sig},
        )
        assert r.status_code == 401


def test_operator_heartbeat_no_shared_secret_needed(config: GatewayConfig) -> None:
    """The public operator path must NOT require the foundation token."""
    app = build_app(config)
    with TestClient(app) as client:
        pub, priv, ss58 = _new_operator()
        body = _operator_body(ss58, models=["m"])
        sig = _sign_heartbeat_body(pub, priv, body)
        # No Authorization header at all.
        r = client.post(
            "/v1/operator/heartbeat",
            json={"heartbeat_json": body, "signature": sig},
        )
        assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Self-serve testnet key faucet (POST /v1/keys)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_key_state() -> None:
    """Clear the in-process key store + rate-limiter between tests."""
    import gateway_router.auth as auth_mod

    auth_mod._INPROC_TESTNET_KEYS.clear()
    auth_mod._key_mint_hits.clear()
    yield
    auth_mod._INPROC_TESTNET_KEYS.clear()
    auth_mod._key_mint_hits.clear()


def test_issue_key_mints_testnet_scoped_key(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post("/v1/keys")
        assert r.status_code == 200, r.text
        body = r.json()
        key = body["api_key"]
        assert key.startswith("orogen-testnet-")
        assert "note" in body and body["note"]


def test_self_issued_key_authorizes_v1(config: GatewayConfig) -> None:
    """A freshly minted key must authorize a public /v1 route.

    With PUBLIC_API_TOKENS set, the wrong/no bearer is rejected (401) but the
    self-issued key is accepted.
    """
    app = build_app(config)
    with TestClient(app) as client:
        key = client.post("/v1/keys").json()["api_key"]
        # The wrong bearer is still rejected.
        bad = client.post(
            "/v1/nonces", headers={"Authorization": "Bearer not-a-real-key"}
        )
        assert bad.status_code == 401
        # The minted key authorizes the call.
        ok = client.post("/v1/nonces", headers={"Authorization": f"Bearer {key}"})
        assert ok.status_code == 200, ok.text
        assert ok.json()["nonce"]


def test_key_mint_rate_limit_triggers(
    config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gateway_router.auth as auth_mod

    monkeypatch.setattr(auth_mod, "_KEY_MINT_MAX_PER_WINDOW", 3)
    app = build_app(config)
    with TestClient(app) as client:
        for _ in range(3):
            assert client.post("/v1/keys").status_code == 200
        # The 4th request from the same source IP is rate-limited.
        r = client.post("/v1/keys")
        assert r.status_code == 429, r.text


def test_allow_insecure_operator_endpoints_flag_plumbing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The testnet flag flips TLS verification + http endpoint acceptance."""
    from gateway_router.auth import (
        allow_insecure_operator_endpoints,
        safe_httpx_client,
        validate_endpoint_url,
    )

    # Default false: rejects plain http to a public host, builds a TLS-verifying
    # client.
    monkeypatch.delenv("GATEWAY_ALLOW_INSECURE_OPERATOR_ENDPOINTS", raising=False)
    assert allow_insecure_operator_endpoints() is False
    # Client builds without error in the default (verifying) configuration.
    _client = safe_httpx_client(5.0)
    assert _client is not None
    with pytest.raises(ValueError, match="https"):
        validate_endpoint_url("http://example.com:8100")

    # Flag on: plain http to a public host is allowed (still SSRF-checked) and
    # the outbound client disables TLS verification.
    monkeypatch.setenv("GATEWAY_ALLOW_INSECURE_OPERATOR_ENDPOINTS", "true")
    assert allow_insecure_operator_endpoints() is True
    # Public host over http now passes validation.
    validate_endpoint_url(
        "http://example.com:8100", allow_insecure_http=True
    )
    # But SSRF targets are still rejected even with the flag.
    with pytest.raises(ValueError, match="forbidden"):
        validate_endpoint_url(
            "http://169.254.169.254/latest", allow_insecure_http=True
        )


def test_chat_rejects_receipt_nonce_mismatch(config: GatewayConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
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
        bad_receipt = Receipt(
            version=1,
            job_id="j-1",
            operator_id="op-1",
            model_id="mock-model-7b",
            model_weight_hash="w",
            customer_nonce="0x" + "ff" * 32,
            request_hash="rq",
            response_hash="rs",
            log_probs_sample=[-0.1],
            kernel_pack_hash="k",
            attestation_report_hash="a",
            timestamp_ms=1,
            gateway_id="gw-test",
        ).sign(priv)
        with respx.mock(assert_all_called=True) as mocker:
            mocker.post("http://127.0.0.1:65535/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "j-1",
                        "object": "chat.completion",
                        "model": "mock-model-7b",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
                        "receipt": bad_receipt.model_dump(mode="json"),
                    },
                ),
            )
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model-7b",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers=_public_headers(),
            )
        assert r.status_code == 502
        assert "customer_nonce" in r.text
