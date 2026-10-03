# ADR 0004: A separate, network-enforced read-only verifier

**Status:** accepted

## Context
"Verification: does it determine whether the requested outcome was actually achieved?" A worker that says "done" is not evidence; it may have misread a confirmation, hit a silent failure (our ERP returns 504 *after* saving) or simply hallucinated.

## Decision
After `finish(status=done)`, `agent/verifier.py` runs a second agent with a **fresh context** (none of the worker's reasoning), a prompt telling it not to trust the claim, and a browser page whose every non-GET request is aborted at the network layer (except POSTs to the exact login paths). It shares the worker's cookies to avoid re-login. A failed verdict is fed back to the worker (≤ 2 rounds); runs end as `verified` or `unverified`.

## Alternatives
- **Self-check by the worker**: same context, same blind spots.
- **Deterministic checks per task** (query the DB): strongest, but task-specific; we use that only in the eval harness, where ground truth is available.
- **Prompt-only read-only verifier**: it could "fix" what it checks and report success.

## Consequences
+ A separate signal the eval harness can score for honesty (claimed `verified` vs ground truth).
+ The auditor's browser can't send state-changing (non-GET/HEAD) requests other than vault logins; tested by `test_read_only_blocks_pay_even_with_login_in_query`. It can still follow GET links such as logout.
− Extra LLM calls per run (typically a few).
− Same underlying model as the worker → correlated errors are possible.
