"""In-memory operator catalog fed by heartbeats."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from mining_types import OffChainHeartbeat


@dataclass(slots=True)
class OperatorRecord:
    operator_id: str
    endpoint_url: str
    base_models: set[str] = field(default_factory=set)
    last_seen_ms: int = 0
    price_per_million: int = 0
    region: str = ""
    raw: OffChainHeartbeat | None = None


class OperatorCatalog:
    """Routing lookup. Filters: model + (optional) region + max staleness."""

    def __init__(self, stale_after_ms: int = 30_000) -> None:
        self._by_op: dict[str, OperatorRecord] = {}
        self.stale_after_ms = stale_after_ms

    def upsert(self, hb: OffChainHeartbeat) -> None:
        rec = self._by_op.get(hb.operator_id) or OperatorRecord(
            operator_id=hb.operator_id, endpoint_url=hb.endpoint_url,
        )
        rec.endpoint_url = hb.endpoint_url or rec.endpoint_url
        rec.base_models = {c.base_model_id for c in hb.capabilities}
        rec.last_seen_ms = int(time.time() * 1000)
        rec.price_per_million = hb.price_per_million_tokens
        rec.region = hb.geo_region
        rec.raw = hb
        self._by_op[hb.operator_id] = rec

    def find(
        self,
        *,
        model_id: str,
        max_price: int | None = None,
        region: str | None = None,
    ) -> OperatorRecord | None:
        now_ms = int(time.time() * 1000)
        candidates = [
            r
            for r in self._by_op.values()
            if model_id in r.base_models
            and now_ms - r.last_seen_ms < self.stale_after_ms
            and (max_price is None or r.price_per_million <= max_price)
            and (region is None or r.region == region)
        ]
        if not candidates:
            return None
        # cheapest first; ties → most recent
        candidates.sort(key=lambda r: (r.price_per_million, -r.last_seen_ms))
        return candidates[0]

    def all(self) -> list[OperatorRecord]:
        return list(self._by_op.values())
