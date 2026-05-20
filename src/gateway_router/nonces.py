"""RFC-0007-shape nonce vaults.

Issues fresh nonces per gateway and consumes them on completion. Replays are
rejected. `NonceVault` is the in-process dev/test implementation. Production
uses `RedisNonceVault`, which stores issued and consumed markers in a shared
Redis backend and claims nonces with an atomic Lua script.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from mining_types import NonceChallenge


class NonceStore(Protocol):
    def issue(self) -> NonceChallenge: ...

    def claim(self, nonce: str) -> bool: ...

    def is_known(self, nonce: str) -> bool: ...


@dataclass(slots=True)
class _Entry:
    challenge: NonceChallenge
    consumed: bool = False


class NonceVault:
    def __init__(self, gateway_id: str, ttl_ms: int = 120_000) -> None:
        self.gateway_id = gateway_id
        self.ttl_ms = ttl_ms
        self._store: dict[str, _Entry] = {}

    def _sweep(self) -> None:
        now = int(time.time() * 1000)
        dead = [
            k for k, e in self._store.items()
            if now - e.challenge.issued_at_ms > e.challenge.ttl_ms
        ]
        for k in dead:
            del self._store[k]

    def issue(self) -> NonceChallenge:
        self._sweep()
        ch = NonceChallenge(gateway_id=self.gateway_id, ttl_ms=self.ttl_ms)
        self._store[ch.nonce] = _Entry(challenge=ch)
        return ch

    def claim(self, nonce: str) -> bool:
        """Consume; return True if the nonce was valid+unused."""
        e = self._store.get(nonce)
        if e is None or e.consumed:
            return False
        now = int(time.time() * 1000)
        if now - e.challenge.issued_at_ms > e.challenge.ttl_ms:
            return False
        e.consumed = True
        return True

    def is_known(self, nonce: str) -> bool:
        self._sweep()
        e = self._store.get(nonce)
        return e is not None and not e.consumed


class RedisNonceVault:
    """Durable nonce store for multi-replica production gateways.

    Redis keys:
    - `{namespace}:issued:{nonce}` exists while the nonce is valid.
    - `{namespace}:consumed:{nonce}` exists once a valid nonce has been claimed.

    `claim()` uses a Lua script to atomically check issued+unconsumed and set
    the consumed marker with the remaining TTL of the issued marker.
    """

    _CLAIM_SCRIPT = """
local issued = KEYS[1]
local consumed = KEYS[2]
if redis.call('EXISTS', issued) == 0 then
  return 0
end
if redis.call('EXISTS', consumed) == 1 then
  return 0
end
local ttl = redis.call('PTTL', issued)
if ttl <= 0 then
  return 0
end
redis.call('PSETEX', consumed, ttl, '1')
return 1
"""

    def __init__(
        self,
        gateway_id: str,
        redis_url: str,
        ttl_ms: int = 120_000,
        namespace: str = "orogen:gateway:nonce",
    ) -> None:
        if not redis_url:
            raise ValueError("redis_url is required")
        self.gateway_id = gateway_id
        self.ttl_ms = ttl_ms
        self.namespace = namespace.rstrip(":")
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - dependency packaging guard
            raise RuntimeError("redis package is required for RedisNonceVault") from exc
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._claim = self._redis.register_script(self._CLAIM_SCRIPT)

    def _issued_key(self, nonce: str) -> str:
        return f"{self.namespace}:{self.gateway_id}:issued:{nonce}"

    def _consumed_key(self, nonce: str) -> str:
        return f"{self.namespace}:{self.gateway_id}:consumed:{nonce}"

    def issue(self) -> NonceChallenge:
        ch = NonceChallenge(gateway_id=self.gateway_id, ttl_ms=self.ttl_ms)
        ok = self._redis.set(
            self._issued_key(ch.nonce),
            str(ch.issued_at_ms),
            px=ch.ttl_ms,
            nx=True,
        )
        if not ok:
            # NonceChallenge uses strong randomness; a collision is effectively
            # impossible, but refuse instead of silently returning an untracked nonce.
            raise RuntimeError("nonce collision while issuing challenge")
        return ch

    def claim(self, nonce: str) -> bool:
        return bool(
            self._claim(
                keys=[self._issued_key(nonce), self._consumed_key(nonce)],
                args=[],
            )
        )

    def is_known(self, nonce: str) -> bool:
        pipe = self._redis.pipeline()
        pipe.exists(self._issued_key(nonce))
        pipe.exists(self._consumed_key(nonce))
        issued, consumed = pipe.execute()
        return bool(issued) and not bool(consumed)
