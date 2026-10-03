# Security model

AutoWork lets an LLM operate a browser on behalf of an employee. The model is **not trusted**: it can be wrong, confused, or manipulated by the content it reads. Security therefore comes from deterministic controls *around* the model, not from instructions *to* it.

Everything below marked ✅ is implemented and covered by a test in `tests/test_security.py` (37 tests), unless noted otherwise.

## Assets

| Asset | Why it matters |
|---|---|
| Money movement (marking bills paid) | Irreversible; the main harm a finance agent can do |
| Vendor master data (contact / bank details) | Changing it redirects future payments (classic invoice fraud) |
| Credentials for the ERP and vendor portal | Full access to the systems above |
| The control plane (`:8000`) | Whoever can call it can start runs and approve actions |
| Ground truth / admin endpoints of the simulated world | Lets someone reset or inspect state outside the agent's rules |

## Adversaries and failure sources

1. **Malicious content**: emails and web pages the agent reads (phishing, "SYSTEM: ignore previous instructions…").
2. **A mistaken model**: hallucinated values, wrong vendor, repeated actions, false "done".
3. **Other websites open in the user's browser**: they can send requests to `localhost` (CSRF, DNS rebinding).
4. **Anyone who can read logs**: event logs and reports get shared, so they must not contain secrets.

## Mitigations (implemented)

### Irreversible actions need a human, enforced twice
- ✅ **Button-label gate.** `agent/policy.py` runs before every click. Pay, delete, transfer, approve and similar buttons need approval in `balanced` and `supervised` modes. In `supervised` mode, every save or submit does too.
- ✅ **Network gate.** `agent/netpolicy.py:is_high_risk` and `BrowserSession._guard_route` cover any non-GET request to a pay, delete, transfer or refund path, or one carrying bank, IBAN or SWIFT fields. These are **aborted in the browser** unless approved, however they were triggered (a button with a harmless label such as "Settle", the Enter key, or page JavaScript). Test: `test_network_gate_catches_payment_the_label_rule_missed`.
- ✅ **Approvals bound to what the human saw.**
  - Each approval request carries the tool, the arguments and a page fingerprint. If the page changed while the human was deciding, the agent asks again (`Agent._approve_bound_action`). Test: `test_approval_rerequested_when_page_changed`.
  - A network approval allows exactly one request, matched by method and normalized path. If the action then sends a different high-risk request, it is blocked.
  - Approval question ids are 128-bit random values and **single-use**. Replays and guesses are rejected (`WebHuman.respond`).
- Known limit: when a *button* approval is granted, the one high-risk request that click triggers is allowed without matching its path, because the path isn't known before the click. The approval is still scoped to that single click on that page state.

### The agent can only reach what it should
- ✅ **Every** browser request, including redirects, subresources and `fetch`/XHR, is checked against an allowlist of origins (scheme, host, port) and a blocklist (`/admin`), applied to the **normalized** URL.
  - Normalization covers percent-decoding (repeated), dot segments, duplicate slashes, case, and loopback aliases. Non-http(s) schemes are rejected.
  - Playwright only intercepts the first URL of a redirect chain, so the guard fetches with `max_redirects=0` and checks `Location` itself.
  - The three bypasses that worked before this was built (`/%61dmin`, an open redirect through `/erp/login?next=/erp/../admin/state`, and page-JS `fetch`) are now regression tests.
- ✅ `/admin/*` in the simulated world also requires an `X-Admin-Token` header (`simworld/app.py:require_admin`). That token is used only by the eval harness and the world-reset endpoint, never by the agent. This is defense in depth: the agent's browser blocks `/admin` anyway.
- ✅ **File tools** are confined to the workspace using resolved real paths and `Path.is_relative_to`, which handles symlinks, `..` and absolute paths. Reads are capped at 200 KB per file (8,000 characters shown to the model) and writes at 100 KB (`agent/tools.py:confine`).

### The verifier can't change business data
- ✅ The auditor's page aborts every request that isn't GET or HEAD, except POSTs to the **exact** login paths from the vault (`BrowserSession._read_only_route`). Test: `test_read_only_blocks_pay_even_with_login_in_query`.
- This relies on the apps not changing data on GET. In the simulated apps the only state-changing GETs are `/acme/logout` and `/erp/logout`, which end a session; the auditor shares the worker's browser context, so it could log the worker out, but it can't change records.

### Credentials are kept out of prompts and logs
- ✅ Credentials live in a vault (`agent/vault.py`). The source is `config/vault.json` (git-ignored), `AUTOWORK_VAULT_FILE`, or the demo `config/vault.example.json` for the simulated apps.
- ✅ The model calls `login(site)`, and the browser fills the form itself. Nothing in the workspace contains passwords.
- ✅ `Redactor` scrubs known secret values from every event (`Agent.emit` is the single exit point), from `report.json`, and from every tool observation before it enters a prompt. Test: `test_no_secret_in_events_reports_or_prompts`, which logs into both sites and then searches `events.jsonl`, `report.json` and all prompts.
- Limits: redaction matches the vault's exact values only (see Known gaps), and screenshots are stored unredacted (password inputs are masked by the browser). The usage log (`llm_usage.jsonl`) stores sizes, not prompt content, and passes error bodies through the same `Redactor`; it is not covered by the test above.

### The control plane only obeys its own UI
- ✅ Both servers bind to `127.0.0.1` (`run.py`).
- ✅ `TrustedHostMiddleware` rejects any Host header other than `localhost` or `127.0.0.1`, which defeats DNS rebinding.
- ✅ Mutating `/api/*` calls need `X-AutoWork-Token`, a random per-process value embedded in the served UI page. Other origins can't read that page because no CORS middleware is installed.
- ✅ A foreign `Origin` header is rejected even when the token is present.
- ✅ Request bodies are validated (mode, step and fault ranges), run ids are validated before they are used in file paths, and `/files` can't traverse outside `runs/` (Starlette `StaticFiles`; regression-tested).

### Invented values are flagged, not blocked
- ✅ After every form fill, each value is checked against the user's task and everything observed this run (`agent/provenance.py`, `agent/tools.py:_t_browser_fill`). An unsourced value produces an `UNSOURCED VALUES` warning to the model and a failed step (counted by the error-streak escalation). **The value is not removed and the save is not blocked.** The agent's own notes and read-backs are not sources, so a guess can't be laundered through memory. Tests: `tests/test_provenance.py` (not part of the 37 security tests).

### Untrusted content is labelled, and the UI cannot be scripted by it
- ✅ Page text, emails and files reach the model inside `<<<UNTRUSTED_…>>>` blocks. The closing delimiter is neutralized inside the content, so a page can't fake the end of the block. The worker and verifier prompts say that instructions inside these blocks must never be followed.
  - **This is a mitigation, not a boundary.** A model can still be persuaded. The real defense is that the dangerous outcomes (payments, data changes outside the allowlist) are blocked by the code above, whatever the model decides.
- ✅ The UI (`server/static/index.html`) inserts all untrusted text with `textContent`. There is no `innerHTML` anywhere.

### Resource limits
- ✅ Per LLM call: a timeout (`LLM_TIMEOUT_S`, default 90 s), an output budget (`LLM_MAX_OUTPUT_TOKENS`, default 4,096; doubled up to 32,768 only when a reasoning model used it all before writing its tool call), at most 6 attempts for failures, and at most 5 minutes of waiting on rate limits.
- ✅ Per run: a step limit (`AUTOWORK_MAX_STEPS`, default 40), a token budget (`AUTOWORK_MAX_TOKENS`, default 400k) and an active wall-clock budget (`AUTOWORK_MAX_ACTIVE_SECONDS`, default 1800 s, excluding time spent waiting for a human). The auditor has at most 10 steps, and a human question times out after 15 minutes, defaulting to the safe answer (deny or stop).

### Supply chain
- ✅ Runtime dependencies are pinned in `requirements.txt`, and the full transitive set in `requirements.lock`, generated from a clean venv. `make audit` runs `pip-audit` over the installed environment. On 2026-10-04 it reported no known vulnerabilities in any project dependency, and 12 in `pip` 24.0 itself (the venv's installer, not imported by AutoWork; fixed in pip 26.2). CI installs from `requirements.lock`.

## Known gaps (not implemented)

| Gap | Risk | Design for the fix |
|---|---|---|
| Keyword-based risk classification (labels and paths) | An app whose payment endpoint is `/erp/bills/2/settle` passes both gates | A per-application **action manifest**: `{method, path pattern, risk, required approver}` maintained with the app owner. Unknown non-GET endpoints default to "approve". |
| Approval lives in the agent process | A compromised agent process could skip its own gate | A server-side **approval token**: the target system accepts a high-risk request only with a short-lived token signed by the approval service, bound to (user, endpoint, record id, amount). |
| Redaction only knows vault values | A secret the agent discovers on a page (e.g. an API key in an email) is not redacted | Add pattern-based detectors (high-entropy strings, known key formats) to `Redactor`. |
| Prompt-injection resistance is prompt-based | The model may still be steered into *allowed* but unwanted actions (e.g. entering a fake invoice) | Data-flow tagging: values extracted from untrusted sources need a second source or human confirmation before being written. |
| Screenshots | A page may display sensitive data, and screenshots are stored unredacted in `runs/` | Mask regions of known sensitive fields before saving; set a retention policy on `runs/`. |
| Single-user localhost tool | There are no user accounts; anyone with local shell access is trusted | A real deployment needs authentication, per-user vault scopes and an audit log of approvals. |
| The simulated ERP itself has an open redirect (`next=`) | Only matters in the fake app; it is kept as a realistic hazard and the agent is protected against it | n/a |

## Reporting

This is a prototype for a hiring assignment. Please open an issue on the repository.
