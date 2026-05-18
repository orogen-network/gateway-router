# gateway-router

OpenAI-compatible HTTP front-of-house. Maintains a live catalog of operators (fed by
RFC-0003 heartbeats), enforces RFC-0007-shape nonces, routes each chat completion
to a capable operator, and aggregates returned receipts into RFC-0004 settlement
batches.
