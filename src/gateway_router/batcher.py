"""RFC-0004 batch builder.

Collects receipts as the gateway accumulates them, exposes a `seal()` method that
produces a signed `SettlementBatch` ready for chain submission.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict

from mining_types import OperatorSummary, Receipt, SettlementBatch, verify_ed25519


class BatchBuilder:
    def __init__(self, gateway_id: str, gateway_private_key_hex: str, epoch_number: int) -> None:
        self.gateway_id = gateway_id
        self.gateway_private_key_hex = gateway_private_key_hex
        self.epoch_number = epoch_number
        self._receipts: list[Receipt] = []

    def add(self, receipt: Receipt, *, operator_pubkey: str | None = None) -> None:
        """Append a receipt; if `operator_pubkey` is supplied, verify the signature.

        Callers SHOULD pass `operator_pubkey` so receipts can't be forged into the
        batch. CRIT-SVC-002.
        """
        if operator_pubkey is not None:
            if not verify_ed25519(
                operator_pubkey, receipt.signing_payload(), receipt.operator_signature,
            ):
                raise ValueError(
                    f"receipt {receipt.job_id!r} has invalid operator signature"
                )
        self._receipts.append(receipt)

    @property
    def size(self) -> int:
        return len(self._receipts)

    @property
    def receipts(self) -> list[Receipt]:
        return list(self._receipts)

    def seal(self) -> SettlementBatch:
        root = SettlementBatch.merkle_root_of(self._receipts)
        per_op: dict[str, list[Receipt]] = defaultdict(list)
        for r in self._receipts:
            per_op[r.operator_id].append(r)
        summaries: list[OperatorSummary] = []
        agg_mint = 0
        for op_id, rs in per_op.items():
            tokens = sum(max(1, len(r.log_probs_sample)) for r in rs)
            mint = tokens * 100  # placeholder mint maths
            agg_mint += mint
            summaries.append(
                OperatorSummary(
                    operator_id=op_id,
                    receipts_count=len(rs),
                    aggregate_tokens_served=tokens,
                    aggregate_mint_useful=mint,
                    merkle_subroot=SettlementBatch.merkle_root_of(rs),
                )
            )
        batch_id = hashlib.blake2b(
            (root + str(self.epoch_number) + self.gateway_id).encode(),
            digest_size=32,
        ).hexdigest()
        unsigned = SettlementBatch(
            batch_id=batch_id,
            epoch_number=self.epoch_number,
            gateway_id=self.gateway_id,
            receipt_count=len(self._receipts),
            merkle_root=root,
            aggregate_burn_cuc=agg_mint * 2,  # burn ≥ mint × ratio
            aggregate_mint_useful=agg_mint,
            per_operator_summary=summaries,
        )
        return unsigned.sign(self.gateway_private_key_hex)

    def reset(self) -> None:
        self._receipts.clear()
        self.epoch_number += 1

    def now_ms(self) -> int:
        return int(time.time() * 1000)
