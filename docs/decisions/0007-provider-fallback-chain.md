# ADR 0007: Provider fallback chain, disabled during evals

**Status:** accepted (2026-10-04)

## Context
Every provider available for free has a small daily quota: Groq allows about 200k tokens per model per day, and one
task costs roughly 60-190k; Gemini's free tier allows 20 requests per model per day, and one task needs 20-40 calls.
A single provider therefore stops a demo or a reviewer's run partway through a task, which surfaces as a failure even
though the agent did nothing wrong. Several providers expose OpenAI-compatible APIs, each with its own quota.

## Decision
`LLM_FALLBACK_MODELS` is an ordered list of models, each with an optional `provider:` prefix (`gemini:`, `nvidia:`,
`openrouter:`) that selects its base URL and API key. `LLMClient` starts on `LLM_MODEL` and switches, **for the rest
of the run**, to the next model when the current one:

- **is out of quota**: a 429 whose requested wait (`Retry-After`, Groq's "try again in 7h8m", Gemini's `retryDelay`)
  would push the call's total waiting past 5 minutes. This is a heuristic on the wait, not detection of the error
  type, because the providers word their quota errors differently; or
- **keeps failing**: a call used all 6 attempts and the last failure was a server error, timeout or connection error.

Configuration errors (bad key, unknown model, rejected request) are not in either category and raise immediately:
switching would hide a broken setup. The switch is logged as an `llm_retry` event (old model, reason, new model),
shown in the UI timeline, and `report.json` records the calls each model answered (`llm.calls_by_model`).

**Evals never fall back.** `Settings.for_model` drops the chain, because results are reported per model: a run that
switched model midway would credit one model with another's work. When an eval model's quota runs out, the run is
recorded as discarded (not graded) and the suite stops for that model.

## Alternatives
- **One provider, wait out the quota**: a daily limit means hours; unusable for a demo.
- **LiteLLM router**: does fallbacks, but adds a dependency and its own retry semantics on top of ours.
- **Switch back to the primary when its quota resets**: more calls to a provider known to be exhausted, for little
  gain within one run.

## Consequences
+ A run continues through quota exhaustion. Verified live on 2026-10-04 (run `20261004-010939-bc58`): Gemini's daily
  quota 429 (asking for 15,618 s) switched to Groq in the same second; the run then switched twice more as quotas
  ran out and finished `verified` on NVIDIA.
+ Unit-tested with fake SDK clients: switching order, staying on the new model, and "quota exhausted" only once the
  last model is exhausted.
− Models differ in how well they follow the prompts. A run can start on a strong model and finish on a weaker one,
  and the auditor runs on whichever model is current when it starts.
− A per-minute limit that asks for a long wait is indistinguishable from a daily limit and triggers a switch too.
− Fallback models have different quality: on the eval suite, NVIDIA Nemotron 3 Super passed 5/11 (see README).
