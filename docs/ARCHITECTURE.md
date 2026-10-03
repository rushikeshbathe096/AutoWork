# Architecture

## Components

```mermaid
flowchart LR
    subgraph User["User's browser"]
        UI["Control UI<br/>server/static/index.html"]
    end

    subgraph CP["Control plane :8000 (server/app.py)"]
        API["/api/runs, /answer<br/>token + Origin + Host checks"]
        WH["WebHuman<br/>single-use qids"]
    end

    subgraph AG["Agent process (agent/)"]
        CORE["core.py<br/>plan → loop → verify → learn"]
        CTX["context.py<br/>prompt + compression"]
        STUCK["stuck.py<br/>repetition / error streak"]
        POL["policy.py<br/>label gate"]
        TOOLS["tools.py<br/>browser · files · memory · login"]
        BR["browser.py<br/>Playwright + route guard"]
        NET["netpolicy.py<br/>allowlist · high-risk requests"]
        VER["verifier.py<br/>read-only auditor"]
        VAULT["vault.py<br/>credentials + Redactor"]
        MEM["memory.py<br/>working memory · playbook"]
    end

    LLM[("LLM API<br/>OpenAI-compatible<br/>(Groq default)")]

    subgraph SW["Simulated company :8001 (simworld/)"]
        MAIL["Webmail"]
        ACME["Acme vendor portal"]
        ERP["OurCo ERP"]
        ADMIN["/admin/* (admin token)"]
        DB[("SQLite")]
    end

    UI -- "SSE events" --> API
    UI -- "start run / approve<br/>X-AutoWork-Token" --> API
    API --> CORE
    CORE <--> WH
    WH <--> API
    CORE --> CTX --> LLM
    CORE --> STUCK
    CORE --> POL
    CORE --> TOOLS --> BR
    BR --> NET
    TOOLS --> VAULT
    CORE --> MEM
    CORE --> VER --> BR
    BR -- "every request checked" --> MAIL & ACME & ERP
    MAIL & ACME & ERP --> DB
    ADMIN --> DB
    EVAL["evals/run_evals.py"] -- "X-Admin-Token" --> ADMIN
```

## One step of the worker loop

```mermaid
sequenceDiagram
    participant C as core.Agent
    participant L as LLM
    participant P as policy.py
    participant H as Human
    participant T as ToolBox
    participant B as Browser (route guard)
    participant A as App (ERP)

    C->>C: budget check (steps, tokens, active time)
    C->>L: system + brief + compressed history + memory
    L-->>C: reasoning + ONE tool call
    C->>C: repetition check (tool, args, page fingerprint)
    C->>P: evaluate(tool, args, current page)
    alt high-risk button / supervised write
        C->>H: approval (action + page fingerprint)
        H-->>C: approve / deny
        C->>C: page changed? → ask again
    end
    C->>T: run tool
    T->>B: click / fill / goto / login
    B->>B: normalise URL → allowlist → high-risk? → redirect check
    alt high-risk request not pre-approved
        B-->>C: ApprovalRequired(POST /erp/bills/2/pay)
        C->>H: approval for that exact request
        C->>T: replay once with that request allowed
    end
    B->>A: request
    A-->>B: page
    B-->>T: snapshot (numbered elements, alerts, status) + screenshot
    T-->>C: redacted, delimited observation
    C->>C: error-streak check, emit redacted events to events.jsonl and UI
```

## Run lifecycle

1. **Plan.** One JSON call returns the goal, success criteria, a plan and any blocking questions. The blocking questions are asked before any action.
2. **Execute.** The loop above runs until `finish` is called or a budget runs out.
3. **Verify.** If the status is `done`, the auditor (fresh context, same cookies, read-only page) checks each success criterion and returns a verdict. A failed verdict is fed back to the worker, for at most 2 rounds; an inconclusive one yields `unverified`.
4. **Learn.** For verified runs only, a distiller writes up to 5 general notes to `data/playbook.json`. They are injected into future prompts as hints that may be outdated.
5. **Report.** `runs/<id>/report.json` and `events.jsonl` (schema v1, redacted), plus a screenshot per step.

## Data flow and trust boundaries

| Boundary | What crosses it | Trust | Control |
|---|---|---|---|
| Web pages, emails and files → model | Page text, labels, values | **Untrusted** | Wrapped in `<<<UNTRUSTED_…>>>`; secrets redacted; size-limited |
| Model → actions | Tool calls | **Untrusted** | Schema validation; `policy.py`; network guard; approvals; budgets; stuck detection; value-provenance warning after fills (warns, does not block) |
| Agent → apps | HTTP requests | Must stay in scope | Origin allowlist, `/admin` block and high-risk gate on the normalized URL of every request, including redirects |
| Vault → browser | Usernames and passwords | Secret | Filled into the DOM directly; not in prompts; redacted from events, reports and observations (`test_no_secret_in_events_reports_or_prompts`); screenshots are not redacted |
| Agent → LLM provider | Prompts | Leaves the machine | Page and task content; vault secrets are redacted (tested). A secret that appears on a page and is not in the vault would be sent |
| Other local websites → control plane | HTTP | **Untrusted** | Host check, token, Origin check, no CORS |
| Eval harness → `/admin` | Reset, faults, ground truth | Trusted test code | Admin token |
| Worker → verifier | The claim, success criteria and recorded facts | The verifier treats them as claims to check | Fresh context; read-only network layer |

## Why the verifier shares cookies

The auditor opens a new page in the worker's browser *context*, so it is already logged in. That saves login steps and LLM tokens, and avoids a second set of sessions with side effects such as the session-expiry fault. It doesn't weaken the read-only rule, because that rule is enforced per request on the auditor's page, not per session.

## LLM providers: fallback chain and usage log

`agent/llm.py` talks to any OpenAI-compatible API. With `LLM_FALLBACK_MODELS` set, a model that is out of quota (a 429 asking for a wait that would exceed 5 minutes in total) or keeps failing (6 attempts, the last a 5xx, timeout or connection error) is replaced by the next model in the list for the rest of the run; bad keys and unknown models (401/403/404) stop the run instead. Each switch is an `llm_retry` event, and `report.json` records how many calls each model answered (`llm.calls_by_model`). The eval harness disables the chain so results stay per model ([ADR 0007](decisions/0007-provider-fallback-chain.md)).

Each LLM request (successful, or failed with a 429, 400, 5xx, timeout or connection error; not 401/403/404) appends one row to `runs/<id>/llm_usage.jsonl` (`agent/usage.py`): role, step, model, latency, provider-reported token counts, rate-limit headers, a redacted error body, and estimated sizes of each prompt section. No prompt content is stored. `python -m evals.usage_report <run dir>` turns it into per-role totals, peak requests and tokens per minute, the 429s, the largest prompt sections and how the prompt grows by step; it computes daily run capacity only for a complete, single-model run with a known daily limit (reported by the provider, or Groq's free tier of 200k tokens per model).
