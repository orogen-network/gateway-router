"""Per-operator sr25519 heartbeat authentication.

External operators must be able to enter the routing catalog WITHOUT the shared
foundation `INTERNAL_AUTH_TOKEN`. Instead they prove ownership of their on-chain
hotkey (the ss58-encoded sr25519 account) by signing their heartbeat payload.

Signature scheme (matches `wallet-sdk-core` / `wallet-cli`)
-----------------------------------------------------------
`wallet-cli heartbeat-test` produces, for the gateway `/v1/operator/heartbeat`
ingest, a domain-separated sr25519 signature:

    hash = BLAKE2b-512( DOMAIN_HEARTBEAT || body )
    sig  = sr25519_sign( substrate_context("substrate"), hash )

where `DOMAIN_HEARTBEAT = b"orogen.heartbeat.v1\\x00"` and `body` is the exact
UTF-8 bytes of the heartbeat JSON the operator signed (field order preserved).
The verifier therefore operates on the *raw* body bytes the operator submitted,
never a re-serialized copy, so JSON key ordering can never desync sign/verify.

The signer's public key is the 32-byte sr25519 account behind the operator's
ss58 hotkey (ss58 prefix 42 on Forge). We decode the ss58 string, validate the
BLAKE2b checksum, and verify the signature against the recovered public key.

Optionally (behind `GATEWAY_REQUIRE_ONCHAIN_OPERATOR`) we confirm the operator
is actually registered on-chain by reading `OperatorStake.Operators(account)` at
the configured Forge RPC endpoint. This is OFF by default for bring-up so the
network can populate its catalog before every operator has finished staking.
"""

from __future__ import annotations

import hashlib
import os

# Domain tag for RFC-0003 operator heartbeats. MUST byte-match
# `wallet_sdk_core::signing::DOMAIN_HEARTBEAT`.
DOMAIN_HEARTBEAT = b"orogen.heartbeat.v1\x00"

# Substrate SS58 address prefix used on Forge (generic substrate prefix 42).
SS58_PREFIX = 42

# Substrate sr25519 signing context label (schnorrkel `signing_context`).
_SS58PRE = b"SS58PRE"

_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


class OperatorAuthError(ValueError):
    """Raised when an operator heartbeat fails signature / identity checks."""


def _b58decode(s: str) -> bytes:
    """Minimal base58 (Bitcoin alphabet) decode — no external dependency.

    `py-sr25519-bindings` is the only crypto dependency we add; base58 is small
    enough to keep in-tree so ss58 decoding never pulls a second library.
    """
    num = 0
    for ch in s:
        idx = _B58_ALPHABET.find(ch.encode())
        if idx == -1:
            raise OperatorAuthError(f"invalid base58 character: {ch!r}")
        num = num * 58 + idx
    # Convert to bytes.
    full = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
    # Restore leading-zero bytes (encoded as leading '1's).
    pad = 0
    for ch in s:
        if ch == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + full


def _b58encode(data: bytes) -> str:
    num = int.from_bytes(data, "big")
    out = bytearray()
    while num > 0:
        num, rem = divmod(num, 58)
        out.append(_B58_ALPHABET[rem])
    pad = 0
    for b in data:
        if b == 0:
            pad += 1
        else:
            break
    return ("1" * pad) + out[::-1].decode()


def encode_ss58(pubkey: bytes, prefix: int = SS58_PREFIX) -> str:
    """Encode a raw 32-byte sr25519 public key as an ss58 address.

    Inverse of `decode_ss58`; used by operator tooling and tests.
    """
    if len(pubkey) != 32:
        raise OperatorAuthError("ss58 public key must be 32 bytes")
    if prefix >= 64:
        raise OperatorAuthError("only simple ss58 prefixes (<64) are supported")
    checksum = hashlib.blake2b(
        _SS58PRE + bytes([prefix]) + pubkey, digest_size=64
    ).digest()[:2]
    return _b58encode(bytes([prefix]) + pubkey + checksum)


def decode_ss58(address: str) -> bytes:
    """Decode an ss58 address into its raw 32-byte sr25519 public key.

    Validates the simple (single-byte) prefix and the 2-byte BLAKE2b checksum,
    matching `wallet_sdk_core::addresses::decode_ss58`.
    """
    if not address:
        raise OperatorAuthError("empty ss58 address")
    raw = _b58decode(address)
    if len(raw) != 1 + 32 + 2:
        raise OperatorAuthError(
            "unexpected ss58 length (expected 35 bytes for a simple prefix)"
        )
    prefix = raw[0]
    if prefix >= 64:
        raise OperatorAuthError("only simple ss58 prefixes (<64) are supported")
    pubkey = raw[1:33]
    want = hashlib.blake2b(_SS58PRE + bytes([prefix]) + pubkey, digest_size=64).digest()[:2]
    if want != raw[33:35]:
        raise OperatorAuthError("ss58 checksum mismatch")
    return pubkey


def _heartbeat_hash(body: bytes) -> bytes:
    return hashlib.blake2b(DOMAIN_HEARTBEAT + body, digest_size=64).digest()


def verify_heartbeat_signature(
    operator_ss58: str,
    body: bytes,
    signature_hex: str,
) -> bytes:
    """Verify a domain-prefixed sr25519 heartbeat signature.

    Returns the recovered 32-byte public key on success; raises
    `OperatorAuthError` on any malformed input or verification failure.
    """
    import sr25519  # local import: only operator-heartbeat path needs it

    pubkey = decode_ss58(operator_ss58)
    sig_hex = signature_hex[2:] if signature_hex.startswith("0x") else signature_hex
    try:
        sig = bytes.fromhex(sig_hex)
    except ValueError as exc:
        raise OperatorAuthError("signature is not valid hex") from exc
    if len(sig) != 64:
        raise OperatorAuthError("sr25519 signature must be 64 bytes (128 hex chars)")
    digest = _heartbeat_hash(body)
    try:
        ok = sr25519.verify(sig, digest, pubkey)
    except ValueError as exc:
        raise OperatorAuthError(f"signature structurally invalid: {exc}") from exc
    if not ok:
        raise OperatorAuthError("invalid operator signature")
    return pubkey


def require_onchain_operator() -> bool:
    """Whether catalog entry requires an on-chain `OperatorStake` registration."""
    return os.environ.get("GATEWAY_REQUIRE_ONCHAIN_OPERATOR", "").lower() in {
        "1",
        "true",
        "yes",
    }


def onchain_rpc_url() -> str:
    return (
        os.environ.get("GATEWAY_CHAIN_RPC_URL", "")
        or os.environ.get("OROGEN_RPC_URL", "")
        or "wss://forge-rpc.orogen.network"
    ).strip()


def is_registered_onchain(operator_ss58: str, *, rpc_url: str | None = None) -> bool:
    """Query `OperatorStake.Operators(account)` at the Forge RPC.

    Returns True iff the storage item exists (operator has registered + staked).
    Requires `substrate-interface`; raises `OperatorAuthError` if it is missing
    so the operator gets a clear error rather than a silent pass.
    """
    url = (rpc_url or onchain_rpc_url())
    try:
        from substrateinterface import SubstrateInterface  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when flag is on
        raise OperatorAuthError(
            "GATEWAY_REQUIRE_ONCHAIN_OPERATOR is set but substrate-interface is "
            "not installed; install it or disable the on-chain check"
        ) from exc
    try:
        substrate = SubstrateInterface(url=url)
        result = substrate.query("OperatorStake", "Operators", [operator_ss58])
    except Exception as exc:  # pragma: no cover - network dependent
        raise OperatorAuthError(f"on-chain operator lookup failed: {exc}") from exc
    return result is not None and getattr(result, "value", None) is not None
