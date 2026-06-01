"""Gateway configuration.

The `gateway_private_key_hex` is wrapped in `pydantic.SecretStr` so it never
appears in log lines, `repr()`, or accidental JSON serializations of the
config (HIGH-SVC-009). Retrieve via `.gateway_private_key()` only inside the
signing call-sites.
"""

from __future__ import annotations

import os
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

    @classmethod
    def from_env(cls) -> GatewayConfig:
        """Build the runtime config from the process environment.

        Reads the required `GATEWAY_ID` / `GATEWAY_PRIVATE_KEY_HEX` and the
        optional tunables. `GATEWAY_REQUEST_TIMEOUT_S` overrides the upstream
        request timeout: the 10s default is fine for a warm GPU operator but
        far too low for a cold CPU-edge worker that reloads a multi-GB GGUF
        per request (which otherwise surfaces as `operator unreachable:
        ReadTimeout`). Optional knobs fall back to the dataclass defaults.
        """
        kwargs: dict[str, object] = {
            "gateway_id": os.environ["GATEWAY_ID"],
            "gateway_private_key_hex": os.environ["GATEWAY_PRIVATE_KEY_HEX"],
        }
        if (epoch := os.environ.get("GATEWAY_EPOCH_NUMBER", "").strip()):
            kwargs["epoch_number"] = int(epoch)
        if (ttl := os.environ.get("GATEWAY_NONCE_TTL_MS", "").strip()):
            kwargs["nonce_ttl_ms"] = int(ttl)
        if (timeout := os.environ.get("GATEWAY_REQUEST_TIMEOUT_S", "").strip()):
            kwargs["request_timeout_s"] = float(timeout)
        return cls(**kwargs)  # type: ignore[arg-type]
