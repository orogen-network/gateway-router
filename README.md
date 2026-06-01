# gateway-router

OpenAI-compatible HTTP front-of-house. Maintains a live catalog of operators (fed by
RFC-0003 heartbeats), enforces RFC-0007-shape nonces, routes each chat completion
to a capable operator, and aggregates returned receipts into RFC-0004 settlement
batches.

## Self-serve testnet API key

On the Forge testnet, customers mint their own API key — no signup, no shared
secret:

```
POST https://gateway.orogen.network/v1/keys
-> { "api_key": "orogen-testnet-…", "note": "…" }
```

The returned key is a testnet-scoped bearer accepted on all `/v1/*` routes
(alongside any statically configured `PUBLIC_API_TOKENS`). The mint endpoint is
unauthenticated and rate-limited per source IP. Keys are stored in Redis when
the gateway is configured with one, else in-process (reset on restart).

`GATEWAY_ALLOW_INSECURE_OPERATOR_ENDPOINTS=true` (default false, testnet only)
lets the gateway route to a plain-http operator worker without verifying TLS;
the SSRF allow-list still applies.

## Operator heartbeat (no shared secret)

External operators enter the routing catalog by signing a heartbeat with the
sr25519 key behind their ss58 hotkey — no foundation token required. The shared
`INTERNAL_AUTH_TOKEN` is reserved for foundation/admin endpoints
(`/internal/catalog`, `/internal/seal_batch`, the legacy `/internal/heartbeat`).

Endpoint:

```
POST https://gateway.orogen.network/v1/operator/heartbeat
Content-Type: application/json

{
  "heartbeat_json": "{\"version\":1,\"operator_ss58\":\"5...\",\"endpoint_url\":\"https://your-host\",\"models\":[\"mock-model-7b\"],\"price_per_million_tokens\":1500,\"geo_region\":\"US\"}",
  "signature": "0x<128-hex sr25519 signature>"
}
```

`signature` is a domain-prefixed sr25519 signature over the EXACT
`heartbeat_json` bytes:

```
hash = BLAKE2b-512( b"orogen.heartbeat.v1\x00" || heartbeat_json )
sig  = sr25519_sign( substrate_context("substrate"), hash )
```

This matches `wallet-sdk-core` / `wallet-cli heartbeat-test` (`DOMAIN_HEARTBEAT`).
Always submit the exact bytes you signed; the gateway verifies the signature
against the raw body, so any re-serialization (key reordering, whitespace)
invalidates it.

Set `GATEWAY_REQUIRE_ONCHAIN_OPERATOR=true` to additionally require that the
operator is registered/staked on-chain (`OperatorStake.Operators`) before
catalog entry; off by default during bring-up. When enabled, install the
`onchain` extra and point `GATEWAY_CHAIN_RPC_URL` at the Forge RPC.
