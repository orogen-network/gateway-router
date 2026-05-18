"""Operator public-key registry.

In production this is sourced from `pallet-operator-registry` via a chain client.
For the skeleton + tests, we expose a small in-process dict that can be:
  - populated by tests via `OperatorRegistry.register(operator_id, pubkey_hex)`;
  - loaded from a JSON file path via `OPERATORS_REGISTRY_PATH`
    (mapping `{operator_id: public_key_hex}`).

The registry is the trust anchor for every signature verification in the gateway:
heartbeats, receipts, and (separately) the gateway's own batch signature.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class OperatorRegistry:
    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._by_op: dict[str, str] = dict(initial or {})

    @classmethod
    def from_env(cls) -> OperatorRegistry:
        path = os.environ.get("OPERATORS_REGISTRY_PATH", "").strip()
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"operators registry file {path!r} must be a JSON object")
        return cls({str(k): str(v) for k, v in data.items()})

    def register(self, operator_id: str, public_key_hex: str) -> None:
        self._by_op[operator_id] = public_key_hex

    def get(self, operator_id: str) -> str | None:
        return self._by_op.get(operator_id)

    def __contains__(self, operator_id: str) -> bool:
        return operator_id in self._by_op

    def __len__(self) -> int:
        return len(self._by_op)
