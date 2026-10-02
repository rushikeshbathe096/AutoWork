# ADR 0006: Groq via the OpenAI-compatible API, provider configurable

**Status:** accepted

## Context
The user chose Groq. Groq exposes an OpenAI-compatible chat-completions API with tool calling.

## Decision
`agent/llm.py` uses the `openai` SDK with `base_url` from settings (`LLM_BASE_URL`, default Groq), `LLM_MODEL` (default `openai/gpt-oss-120b`) and `LLM_API_KEY` (`GROQ_API_KEY` accepted as fallback). The client handles provider quirks itself: 429 with Retry-After or "try again in Xms", 5xx backoff, Groq's `tool_use_failed` (retry with a correction), fallback from `tool_choice="required"` to `"auto"`, and JSON-mode failure.

## Alternatives
- **Groq SDK**: equivalent features, but ties the code to one vendor.
- **LiteLLM**: multi-provider routing; an extra dependency for a capability we get from one `base_url`.

## Consequences
+ Switching to OpenAI, Together, vLLM or Ollama is a `.env` change.
+ Retry/fallback paths are unit-tested with an injected fake SDK client.
− Whether the default model is available on a given account, and how well it follows the prompts, has **not been verified** in this repo yet (no API key was available during development).
