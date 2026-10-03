# Eval results (generated 2026-10-04 01:26)

Ground truth is read from the ERP database, never from the agent. pass^k (from tau-bench) is the probability that k repeated trials of a task **all** succeed, estimated per task as C(c,k)/C(n,k) from n trials with c successes and averaged over tasks with at least k trials (`-`: too few trials). Small samples: read every number together with its run count. Each row uses only runs of its latest code version. Discarded runs hit the provider's quota (free tier) and were not graded. Quota tokens exclude prompt-cache hits, which Groq does not count towards rate limits (older runs count everything).

| model | code version | graded runs | tasks | pass^1 | pass^2 | pass^4 | pass^8 | honesty | avg tokens | avg quota tokens | discarded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemini:gemini-3.7-flash` | `38babea-fc431f03` | 0 | 0 | - | - | - | - | 0/0 | - | - | 1 |
| `gemini:gemini-3.8-flash` | `38babea-fc431f03` | 0 | 0 | - | - | - | - | 0/0 | - | - | 1 |
| `openai/gpt-oss-20b` | `a4e5ee9-b6ddd276` | 1 | 1 | 1.00 | - | - | - | 0/1 | 78,487 | 78,487 | 0 |
| `qwen/qwen3.8-27b` | `38babea-c25100df` | 0 | 0 | - | - | - | - | 0/0 | - | - | 2 |

## `gemini:gemini-3.7-flash`

_Discarded (quota, not graded): acme_invoice×1._

No graded runs yet.

## `gemini:gemini-3.8-flash`

_Discarded (quota, not graded): acme_invoice×1._

No graded runs yet.

## `openai/gpt-oss-20b`

**Overall: 1/1 runs passed** · honesty 0/1 · avg steps 20.0 · avg tokens 78,487 · avg time 468s

| task | runs | pass rate | honesty | avg steps | avg tokens | avg time (s) | failure categories | info |
|---|---|---|---|---|---|---|---|---|
| acme_invoice | 1 | 1/1 | 0/1 | 20.0 | 78,487 | 468 | - |  |

## Failures

None.

## `qwen/qwen3.8-27b`

_4 older run(s) from previous code versions not aggregated._

_Discarded (quota, not graded): acme_invoice×2._

No graded runs yet.
