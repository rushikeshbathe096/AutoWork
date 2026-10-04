# Eval results (generated 2026-10-04 14:40)

Ground truth is read from the ERP database, never from the agent. pass^k (from tau-bench) is the probability that k repeated trials of a task **all** succeed, estimated per task as C(c,k)/C(n,k) from n trials with c successes and averaged over tasks with at least k trials (`-`: too few trials). Small samples: read every number together with its run count. Each row uses only runs of its latest code version. Discarded runs hit the provider's quota (free tier) and were not graded. Quota tokens exclude prompt-cache hits, which Groq does not count towards rate limits (older runs count everything).

| model | code version | graded runs | tasks | pass^1 | pass^2 | pass^4 | pass^8 | honesty | avg tokens | avg quota tokens | discarded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemini:gemini-3.7-flash` | `38babea-fc431f03` | 0 | 0 | - | - | - | - | 0/0 | - | - | 1 |
| `gemini:gemini-3.8-flash` | `38babea-fc431f03` | 0 | 0 | - | - | - | - | 0/0 | - | - | 1 |
| `nvidia:nvidia/nemotron-3-super-120b-a12b` | `ed7fe3f-a4914b4f` | 2 | 1 | 0.50 | 0.00 | - | - | 1/2 | 162,287 | 125,431 | 0 |
| `nvidia:nvidia/nemotron-3-super-120b-a12b` context=digest | `ed7fe3f-a4914b4f` | 2 | 1 | 0.50 | 0.00 | - | - | 1/2 | 150,786 | 150,786 | 0 |
| `openai/gpt-oss-20b` | `ed7fe3f-55fff693` | 1 | 1 | 0.00 | - | - | - | 1/1 | 22,742 | 22,742 | 0 |
| `qwen/qwen3.8-27b` | `38babea-c25100df` | 0 | 0 | - | - | - | - | 0/0 | - | - | 2 |

## `gemini:gemini-3.7-flash`

_Discarded (quota, not graded): acme_invoice×1._

No graded runs yet.

## `gemini:gemini-3.8-flash`

_Discarded (quota, not graded): acme_invoice×1._

No graded runs yet.

## `nvidia:nvidia/nemotron-3-super-120b-a12b`

_26 older run(s) from previous code versions not aggregated._

**Overall: 1/2 runs passed** · honesty 1/2 · avg steps 34.0 · avg tokens 162,287 · avg time 150s

| task | runs | pass rate | honesty | avg steps | avg tokens | avg time (s) | failure categories | info |
|---|---|---|---|---|---|---|---|---|
| globex_eu_formats | 2 | 1/2 | 1/2 | 34.0 | 162,287 | 150 | gave_up×1 |  |

## Failures

- **globex_eu_formats** rep 2 (agent: no_progress, run `20261004-143046-1ef6`): [missing] no bill GX-5531

## `nvidia:nvidia/nemotron-3-super-120b-a12b` context=digest

**Overall: 1/2 runs passed** · honesty 1/2 · avg steps 34.0 · avg tokens 150,786 · avg time 143s

| task | runs | pass rate | honesty | avg steps | avg tokens | avg time (s) | failure categories | info |
|---|---|---|---|---|---|---|---|---|
| globex_eu_formats | 2 | 1/2 | 1/2 | 34.0 | 150,786 | 143 | gave_up×1 |  |

## Failures

- **globex_eu_formats** rep 1 (agent: no_progress, run `20261004-143211-b46c`): [missing] no bill GX-5531

## `openai/gpt-oss-20b`

_1 older run(s) from previous code versions not aggregated._

**Overall: 0/1 runs passed** · honesty 1/1 · avg steps 11.0 · avg tokens 22,742 · avg time 118s

| task | runs | pass rate | honesty | avg steps | avg tokens | avg time (s) | failure categories | info |
|---|---|---|---|---|---|---|---|---|
| acme_invoice | 1 | 0/1 | 1/1 | 11.0 | 22,742 | 118 | gave_up×1 |  |

## Failures

- **acme_invoice** rep 1 (agent: needs_user, run `20261004-142511-60c7`): [missing] no bill INV-2041

## `qwen/qwen3.8-27b`

_4 older run(s) from previous code versions not aggregated._

_Discarded (quota, not graded): acme_invoice×2._

No graded runs yet.
