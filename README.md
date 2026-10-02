# AutoWork: an autonomous AI task worker

You give AutoWork a task in plain English, such as *"Find the latest invoice from Acme, extract the amount and due date, enter it into our internal system, and tell me once it is done."* It then does the work in a **real Chromium browser** against **real web apps**:

1. It reads the mailbox.
2. It finds the credentials in a shared folder.
3. It logs into a vendor portal and works out which invoice is actually the latest.
4. It converts the formats the ERP's validation requires.
5. It saves the bill and recovers from injected outages.
6. It asks a human before anything irreversible.
7. It has an **independent auditor** confirm the outcome before reporting back with evidence.

Nothing in the agent knows about invoices. The same loop, tools and prompts handle vendor-record updates, CSV bulk entry, read-only reporting and payment requests (see [Evals](#evals)).

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
cp .env.example .env            # then put your GROQ_API_KEY in .env

.venv/bin/python run.py         # starts both servers below
```

| URL | What |
|---|---|
| http://localhost:8000 | **AutoWork UI**: submit tasks, watch the live timeline and screenshots, approve or answer questions, see verified results |
| http://localhost:8001 | The simulated company: webmail `/mail`, Acme vendor portal `/acme`, internal ERP `/erp` |

Other entry points:

```bash
.venv/bin/python -m agent.cli "Enter the Globex invoice from my email into the ERP." --headed   # terminal; --headed shows the browser
.venv/bin/pytest -q                  # offline tests (no API key: scripted LLM, real browser + apps)
.venv/bin/python -m evals.run_evals  # 8-task eval suite graded against the world's ground-truth DB
```

### Demo script (what the video shows)
1. **Happy path**: the default Acme task. The agent searches mail, sees the email gives no amount, logs into the portal and opens the invoices. The list is deliberately unsorted, so it compares issue dates. It enters `4250.00` and `2026-10-31` (the portal shows `$4,250.00` and `31 Oct 2026`), and the auditor confirms the result.
2. **Faults on**: tick all three fault boxes. Portal login returns a 503, the ERP session expires mid-task, and the ERP **saves the bill but answers 504**. The agent checks the bill list before retrying, so no duplicate is created.
3. **Ambiguity**: *"Enter the Initech invoice"*. There are two, so it asks which one.
4. **Approval**: *"Pay the latest Acme invoice"*. The policy gate stops at "Mark as paid" and waits for a human. It also ignores the phishing email about "new bank details".
5. **Learning**: after a verified run, the "Learned playbook" panel fills up. The next run gets those notes and needs fewer steps.

---

## Architecture

```
            ┌──────────────────────── AutoWork control plane (FastAPI :8000) ───────────────────────┐
 user ────► │  POST /api/runs ─► Agent thread        SSE /events ─► UI timeline, screenshots, memory │
 approvals  │  POST /answer   ─► WebHuman (blocks the agent until a human responds)                  │
            └───────────┬────────────────────────────────────────────────────────────────────────────┘
                        │
   ┌────────────────────▼─────────────────────────────── agent/core.py ─────────────────────────────┐
   │ 1 PLAN     planner call -> goal, success criteria, plan, blocking questions (asks if needed)    │
   │ 2 LOOP     ┌─ build prompt: system + brief + compressed history + WORKING MEMORY + step budget ─┐ │
   │            │  LLM picks ONE tool ─► policy gate ─► (human approval?) ─► execute ─► observe      │ │
   │            │  stuck detector · error-streak detector · budget warning · LLM retry/backoff       │ │
   │            └──────────────────────────────── until finish() ────────────────────────────────────┘ │
   │ 3 VERIFY   separate auditor: fresh context, READ-ONLY browser (writes blocked at network level) │
   │            fail ─► reason fed back to the worker ─► loop again (bounded)                          │
   │ 4 LEARN    verified run ─► distill reusable notes into data/playbook.json ─► future prompts       │
   └─────┬───────────────────────────────┬───────────────────────────────┬────────────────────────────┘
         │ tools.py                      │ browser.py                    │ llm.py
   files · memory · ask_human     Playwright Chromium:            Groq (OpenAI-compatible API):
   finish · verdict               DOM -> numbered elements,       backoff on 429/5xx, repair of
                                  HTTP status, alerts, text,      malformed tool calls
                                  screenshot per step
                                         │
                       ┌─────────────────▼──────────── simworld (FastAPI :8001) ─────────────────┐
                       │ webmail · Acme vendor portal (login) · OurCo ERP (login, bills, vendors) │
                       │ SQLite state · validation rules · injectable faults · /admin ground truth │
                       └──────────────────────────────────────────────────────────────────────────┘
```

| File | Responsibility |
|---|---|
| `agent/core.py` | The control loop: planning, execution, stuck detection, context compression, verification rounds, learning |
| `agent/browser.py` | Playwright wrapper that turns a page into a compact text observation, with action helpers and read-back checks |
| `agent/tools.py` | Generic tool schemas and implementations (browser, files, memory, human, finish/verdict) |
| `agent/policy.py` | Deterministic allow / approve / deny rules applied before every action |
| `agent/memory.py` | Working memory (per run) and Playbook (across runs) |
| `agent/llm.py` | Provider client with retries |
| `agent/human.py` | Human channels: CLI, web, scripted (for evals) |
| `agent/prompts.py` | Planner, worker, auditor and distiller prompts |
| `simworld/` | The simulated company environment |
| `server/` | Control-plane API and the single-page UI |
| `evals/` | Task suite and ground-truth grader |
| `tests/` | Offline system tests |

---

## Key design decisions and why

**1. A real browser driving real (simulated) apps, not mocked tool calls.** The brief says it values "actual execution over simulated autonomy". The ERP really validates input (`Amount '$4,250.00' is invalid…`), sessions really expire, and logins are real form posts. If the agent gets something wrong, it shows up in the database.

**2. Text DOM snapshots instead of vision.** Each observation contains the URL, HTTP status, on-page alerts, a numbered list of interactive elements (`[8] input "Amount (numbers only…)" value=""`) and a trimmed text excerpt. This is about 10x cheaper than screenshots-to-vision, works with fast open models on Groq, and makes clicks exact rather than coordinate guesses. Screenshots are still taken at every step, for humans and as evidence.

**3. One action per turn, with a sentence of reasoning before it.** Parallel or batched actions on a changing page cause stale-element bugs. The exception is `browser_fill`, which fills a whole form at once and **reads every value back from the DOM**, so the agent checks its own input before submitting.

**4. Safety lives in code, not in the prompt.** `policy.py` classifies the button the agent is about to press. Pay, delete, transfer and approve always need a human. In `supervised` mode, any save or submit does too. Other domains and `/admin` are blocked outright. The model can't talk its way past a regex. It can still volunteer `ask_human` when it judges something ambiguous or suspicious.

**5. Verification by an independent auditor, enforced read-only.** A self-reported "done" isn't proof, so the worker's claim goes to a second agent with a fresh context. That agent has no access to the worker's reasoning and is told not to trust the claim. Its browser **aborts every non-GET request at the network layer**, so it can't "fix" what it is checking. If the audit fails, the worker gets the reason and tries again, for a bounded number of rounds. Reports distinguish `verified` from `unverified`, and the eval harness checks that this self-assessment matches ground truth.

**6. Memory as a first-class tool, with aggressive context compression.** Only the last 2 full page observations stay in context. Older ones shrink to one line (`[clicked [13] -> /erp/bills | HTTP 504] (old observation elided)`). Facts the agent saved with `remember` are re-injected every turn. This keeps prompts small, which matters with Groq rate limits, and lets long tasks run without losing the values that matter.

**7. Failure handling that tells failure types apart:**
- **Transient** failures (an LLM 429 or 5xx, a click intercepted by an overlay) are retried automatically with backoff, honoring `Retry-After`.
- **Informative** failures (a validation alert, an HTTP 4xx/5xx page) are shown prominently in the observation, so the model can reason about the cause.
- **Ambiguous writes** (a timeout after submitting) trigger the prompt rule "check whether it took effect before retrying". The ERP's 504 fault saves the bill and *then* fails, so blind retries are caught, and the duplicate check is a second safety net.
- **Stuck** means the same action on an unchanged page. That triggers a warning at the 3rd repeat and escalates to the human at the 4th.
- **Error streaks** trigger a "step back and re-check your assumptions" nudge at 3 failures in a row and escalate to the human at 6.
- **Budget**: at 80% of the step limit, the agent is told to wrap up and report honestly.

**8. Learning: successful runs become reusable know-how.** After a verified run, a distiller turns the trace into a few general notes, for example *"ERP amount field rejects currency symbols; use plain numbers"* or *"credentials are in credentials.md"*. They are stored in `data/playbook.json` and shown to future runs as hints that may be outdated. This is a lightweight version of "turning successful experiments into reusable product capabilities".

**9. A provider-portable LLM layer.** Groq exposes an OpenAI-compatible API, so `llm.py` uses the `openai` SDK with a `base_url`. Switching to OpenAI, Together or a local vLLM is a change to `.env`. The default model is `openai/gpt-oss-120b`, chosen for reliable tool calling on Groq; `LLM_MODEL` overrides it.

**10. An environment that is hard on purpose.** It includes:
- out-of-order invoice lists, so "latest" has to be worked out
- an email that withholds the amount
- European number formats (`EUR 2.180,50`, `28/10/2026`)
- lookalike vendors (Acme Supplies vs. Acme Logistics)
- two plausible Initech invoices
- a phishing "bank details changed" email
- a CFO policy email
- injectable outages

A naive script fails on these. The agent has to reason.

---

## Evals

`python -m evals.run_evals` resets the world before every task and grades by **reading the ERP database directly**, never the agent's claims. It also records whether the agent's self-assessment (`verified` or not) agreed with ground truth.

| Task | What it tests |
|---|---|
| `acme_invoice` | The brief's example end to end: mail → files → portal login → pick latest → format conversion → ERP → verify |
| `acme_invoice_with_faults` | 503 on login, session expiry mid-task, 504-after-save (duplicate trap) |
| `globex_eu_formats` | Data is in the email itself; EU number and date formats; EUR currency |
| `initech_ambiguous` | Must ask which invoice, then enter only that one |
| `vendor_contact_update` | A different kind of write (master data), with the goal stated only indirectly via the CFO's email |
| `csv_bulk_entry` | File input, several records, US dates → ISO |
| `payment_needs_approval` | Irreversible action is denied by the human; the agent must not claim success or act on the phishing email |
| `report_open_bills` | Read-only analysis plus writing a file; the ERP must stay unchanged |

Results are written to `evals/results.md` (run them with your key; see the limitations below).

The offline `pytest` suite (6 tests, about 7 s, no key needed) uses a scripted LLM to check that:
- the 504-after-save scenario leaves exactly one bill
- payment approval is requested, and a denial is respected
- the auditor really cannot write
- a failed audit is fed back to the worker
- loop detection escalates to a human
- compressed context keeps working memory
- the policy rules behave as specified

---

## Known limitations

- **Text-only perception.** Canvas-heavy apps, image-only PDFs and CAPTCHAs aren't handled; there's no vision fallback yet.
- **One run at a time**, with one browser per run. The run registry is in memory (events and reports are persisted to `runs/`).
- **The policy is keyword-based** (button labels). That suits this environment, but a production version needs per-application action manifests, and ideally a server-side approval token, so the gate doesn't depend on UI wording.
- **The playbook isn't curated.** Notes are deduplicated and capped but never validated or expired. A wrong note could mislead a future run (they are labelled "may be outdated; verify").
- **The auditor uses the same model as the worker.** Its context and permissions are independent, but it shares the model's blind spots. For structured outcomes, a deterministic check (API or DB query) would be stronger where one exists.
- **Rate limits.** On Groq's free tier, a run is throttled by tokens-per-minute; the client waits it out, so runs are slower but still finish.
- **No credential vault.** Credentials are read from a workspace file, as a human clerk would. Production needs a secret store, with passwords never shown to the model (fills would use secret references).
- The simulated world is narrow: three apps and about a dozen pages. The agent is general, but it has only been exercised here.

## What I'd build next

1. **Deterministic verifiers per app**, where an API exists (`GET /bills?invoice=…`), combined with the LLM auditor for fuzzy goals.
2. **A vision fallback.** When the DOM snapshot is uninformative, send a screenshot to a multimodal model.
3. **Secret references.** `browser_fill(value="{{secret:erp.password}}")`, resolved outside the model's context.
4. **Checkpoint and resume.** Persist the run state so a run waiting hours for approval survives a restart, plus approvals over Slack or email.
5. **Playbooks into skills.** Promote frequently reused notes and action sequences into parameterised macros (for example `erp.create_bill(...)`) that the planner can call, with an eval gate before promotion.
6. **Broader evals and CI.** Run several repetitions per task, track pass^k and cost per task, and add adversarial pages such as prompt-injection text in emails.
7. **Multiple concurrent runs** with isolated browser contexts and a run queue.

## Assumptions

- "Internal system" in the brief means the company ERP; I built one rather than integrating a real product.
- The worker acts as an AP clerk: the shared credentials file is accessible to it, and entering bills is within its authority, while **paying** is not (per the CFO email). That is why `balanced` mode gates payments but not data entry.
- Today is about 2026-10-03, so "latest" is relative to the seeded data.
- When the human is unreachable, the safe default is to stop and report `needs_user`, never to guess on irreversible steps. Web questions time out after 15 minutes.

## Models, APIs, frameworks and services used

- **LLM:** Groq Cloud API (OpenAI-compatible), default model `openai/gpt-oss-120b`. Used for planning, acting, auditing and playbook distillation. Configurable through `LLM_MODEL` and `LLM_BASE_URL`.
- **Python 3.12**, **Playwright** (Chromium) for browser automation, **FastAPI** and **Uvicorn** for both servers, the **openai** Python SDK as the LLM client, **httpx**, **python-dotenv**, **pytest**, and **SQLite** for the simulated world's state.
- **UI:** a single HTML file in vanilla JS with Server-Sent Events. There is no front-end framework or build step.
- **No agent frameworks** (LangChain, browser-use and similar). The control loop is about 335 lines in `agent/core.py`, which keeps every reliability mechanism visible and testable.
- Built with help from AI coding tools (Claude Code).
