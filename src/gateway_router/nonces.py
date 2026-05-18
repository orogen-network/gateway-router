"""RFC-0007-shape nonce vault.

Issues fresh nonces per (gateway, customer) and consumes them on completion. Replays
are rejected. TTL-cleaned every issue() call (cheap O(N) sweep is fine for skeletons).

MED-SVC-010 — the vault is intentionally in-memory in this build. Multi-replica
deployments and host restarts will lose the issued-nonce set, weakening replay
protection across boundaries. Production deployments MUST switch to a shared
Redis (or equivalent) backend with TTL eviction. The flood-protection / size
limit on `POST /v1/nonces` is enforced at the route layer in `app.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from mining_types import NonceChallenge


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
        return nonce in self._store
