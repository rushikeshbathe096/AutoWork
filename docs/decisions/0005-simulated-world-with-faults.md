# ADR 0005: A simulated company with fault injection instead of real SaaS

**Status:** accepted

## Context
The agent needs realistic systems to act on: email, a vendor portal, an internal ERP. Real SaaS accounts are unavailable, non-resettable, and can't produce failures on demand.

## Decision
`simworld/` is a FastAPI + SQLite app serving server-rendered webmail, an Acme vendor portal (login) and an ERP (login, validation, duplicate detection, vendor edit, payments). It is deliberately messy: unsorted lists, European number formats, lookalike vendors, ambiguous invoices, phishing and prompt-injection emails. Faults are injectable per run: 503 on login, **504 after a successful save**, session expiry. `/admin/*` (admin-token protected) resets state and exposes ground truth to the eval harness.

## Alternatives
- **Mock tool calls** (`create_bill()` returning success): no real browser work, nothing can go wrong, so nothing is learned.
- **Real public sites**: not resettable, no ground truth, terms-of-service issues.

## Consequences
+ The agent does real browser work; outcomes are checked against the database, not the agent's words.
+ Reliability behaviour (retries, duplicate avoidance, re-login) is reproducible in tests and evals.
− Narrow: three apps. The agent's generality is only demonstrated within this world.
