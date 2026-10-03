# ADR 0003: Safety policy enforced in code, not in the prompt

**Status:** accepted

## Context
A finance agent must not pay, delete or redirect money without a human. The model reads untrusted emails and pages and can be wrong or manipulated, so a prompt instruction ("never pay without asking") is not a control.

## Decision
Two deterministic gates, both independent of the model:
1. `agent/policy.py` classifies the element about to be clicked by its label (pay/delete/transfer…) → requires human approval.
2. `agent/netpolicy.py` + the Playwright route guard classify every outgoing request by method + normalised path + body fields → high-risk requests are aborted unless approved, regardless of what triggered them.
Approvals are bound to tool, args and page fingerprint and are single-use.

## Alternatives
- **Prompt-only**: zero effort, zero guarantee.
- **LLM-as-judge for risk**: flexible, but a second model can be fooled the same way.
- **Per-app action manifests**: the right production answer (documented in SECURITY.md), more than this prototype needs.

## Consequences
+ The gates hold whatever the model decides, including under prompt injection, for every action they classify as high-risk. They are keyword-based (labels and paths), so an unusually named payment endpoint would not be classified; see SECURITY.md, known gaps.
+ Testable: `tests/test_security.py` proves payments behind an innocuous label are still stopped.
− Keyword lists can miss unusual endpoint names (`/settle`); see SECURITY.md known gaps.
