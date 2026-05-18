"""Gateway configuration.

The `gateway_private_key_hex` is wrapped in `pydantic.SecretStr` so it never
appears in log lines, `repr()`, or accidental JSON serializations of the
config (HIGH-SVC-009). Retrieve via `.gateway_private_key()` only inside the
signing call-sites.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr


def _as_secret(value: str | SecretStr) -> SecretStr:
    return value if isinstance(value, SecretStr) else SecretStr(value)


@dataclass
class GatewayConfig:
    gateway_id: str
    gateway_private_key_hex: SecretStr
    epoch_number: int = 1
    nonce_ttl_ms: int = 120_000
    request_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gateway_private_key_hex",
            _as_secret(self.gateway_private_key_hex),
        )

    def gateway_private_key(self) -> str:
        """Return the raw hex private key — only call from a signer."""
        return self.gateway_private_key_hex.get_secret_value()
