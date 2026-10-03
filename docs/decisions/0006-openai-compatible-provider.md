# ADR 0006: Groq via the OpenAI-compatible API, provider configurable

**Status:** accepted

## Context
The user chose Groq. Groq exposes an OpenAI-compatible chat-completions API with tool calling.

## Decision
`agent/llm.py` uses the `openai` SDK with `base_url` from settings (`LLM_BASE_URL`, default Groq), `LLM_MODEL` (default `qwen/qwen3.8-27b`) and `LLM_API_KEY` (`GROQ_API_KEY` accepted as fallback). The client handles provider quirks itself: 429 with Retry-After or "try again in Xms", 5xx backoff, Groq's `tool_use_failed` (retry with a correction), fallback from `tool_choice="required"` to `"auto"`, and JSON-mode failure.

## Alternatives
- **Groq SDK**: equivalent features, but ties the code to one vendor.
- **LiteLLM**: multi-provider routing; an extra dependency for a capability we get from one `base_url`.

## Consequences
+ Switching to OpenAI, Together, vLLM or Ollama is a `.env` change.
+ Retry/fallback paths are unit-tested with an injected fake SDK client.
− Whether the default model is available on a given account, and how well it follows the prompts, has **not been verified** in this repo yet (no API key was available during development).

## Update after the first live runs (2026-10-03)

The default model changed from `openai/gpt-oss-120b` to `qwen/qwen3.8-27b`, for measured reasons:

- **gpt-oss-120b writes its tool call inside its hidden reasoning** instead of emitting it. Groq then rejects the
  response with `tool_use_failed` and an empty `failed_generation`. Replaying one captured failing request 3 times
  gave the same result each time; with `tool_choice="auto"` the response ended with empty content and reasoning
  ending in the call's arguments (`...Click link 7.{"element_id":7}`). A correction message cannot fix this, and a
  32k output budget did not either. Which setting (if any) fixes it is still open.
- The same request on **gpt-oss-20b** produced a proper tool call 15/15 times across five settings, so it is
  specific to the 120b model, not to the prompts.
- **qwen3.8-27b** completed the example task 2/2, verified and correct against the database.

Two client fixes came out of this: rate-limit waits have their own time budget instead of consuming the failure
retries (a free tier's 429s had aborted healthy runs), and an empty `failed_generation` is treated as "no call was
written" (retry with a bigger output budget and lower reasoning effort) rather than "the call was malformed".

