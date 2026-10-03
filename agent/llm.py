"""LLM client for any OpenAI-compatible chat-completions API (Groq by default; OpenAI, Together,
vLLM, Ollama... by setting LLM_BASE_URL / LLM_MODEL / LLM_API_KEY). Configuration comes from
agent.config.Settings.

Reliability concerns handled here (so the agent loop doesn't have to):
  * 429 rate limits / 5xx  -> exponential backoff honoring Retry-After
  * Groq `tool_use_failed` -> the model emitted a malformed tool call; retry with a nudge
  * ...with an EMPTY failed_generation -> a reasoning model spent its whole output budget thinking
    and never wrote the call; retry with a bigger budget and lower reasoning effort (a nudge can't help)
  * tool_choice="required" unsupported -> fall back to "auto"
  * a model out of quota (or down) -> switch to the next of LLM_FALLBACK_MODELS, e.g. Groq -> Gemini ->
    NVIDIA -> OpenRouter; each provider has its own free-tier quota
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import openai

from .config import Fallback, Settings, api_model_of
from .usage import UsageLogger

log = logging.getLogger("autowork.llm")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    usage: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """The provider will not serve us for longer than we are willing to wait (e.g. a free tier's daily
    token limit). An infrastructure condition, not agent behaviour: eval harnesses must not grade it."""


class ProviderUnavailable(LLMError):
    """The provider kept failing (5xx, timeouts, connection errors) through every retry."""


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        client: openai.OpenAI | None = None,
        sleep: Callable[[float], None] = time.sleep,
        fallback_clients: list[openai.OpenAI] | None = None,
    ):
        """`client`, `fallback_clients` and `sleep` are injectable so retry and fallback behaviour can be
        unit-tested without network or waiting."""
        s = settings or Settings.from_env()
        self.max_output_tokens, self.reasoning_effort = s.llm_max_output_tokens, s.llm_reasoning_effort
        if client is None:
            if not s.llm_api_key:
                raise LLMError("No LLM API key: set LLM_API_KEY (or GROQ_API_KEY) in .env, see .env.example")
            client = self._make_client(s.llm_base_url, s.llm_api_key, s.llm_timeout_s)
        fallbacks: list[tuple[Fallback, openai.OpenAI]] = []
        for i, fb in enumerate(s.llm_fallbacks):
            if fallback_clients is not None:
                fallbacks.append((fb, fallback_clients[i]))
            elif fb.api_key:
                fallbacks.append((fb, self._make_client(fb.base_url, fb.api_key, s.llm_timeout_s)))
        self._fallbacks = fallbacks
        self.models_used: list[str] = []
        self._use(s.llm_model, client)
        self._sleep = sleep
        self.on_retry: Callable[[str], None] | None = None
        # cached_tokens: the part of prompt_tokens served from the provider's prompt cache. On Groq it doesn't
        # count towards rate limits, so free-tier throughput depends on it (measured ~51% on gpt-oss).
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "retries": 0}
        self.quota_exhausted = False  # set when a call gave up on rate limits; the agent turns errors into reports
        self.usage_logger: UsageLogger | None = None

    def set_usage_logger(self, logger: UsageLogger | None) -> None:
        self.usage_logger = logger

    @staticmethod
    def _make_client(base_url: str, api_key: str, timeout: float) -> openai.OpenAI:
        # max_retries=0: retries are handled below, with provider-specific logic the SDK doesn't know
        return openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout)

    def _use(self, model: str, client: openai.OpenAI) -> None:
        self.model = model  # as configured, e.g. "gemini:gemini-3.8-flash": used in reports
        self.api_model = api_model_of(model)  # as the provider expects it
        self.client = client
        self.models_used.append(model)
        self._tool_choice_required_ok = True
        self._parallel_param_ok = True  # not every OpenAI-compatible endpoint accepts parallel_tool_calls

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        require_tool: bool = False,
        json_mode: bool = False,
        role: str = "worker",
        step: int = 0,
        **kw: Any,
    ) -> LLMResponse:
        """One chat completion (arguments as in _chat_once). If the current model is out of quota or keeps
        failing, switch for good to the next fallback model and repeat the call there: the conversation is
        plain OpenAI-format messages, so any provider can continue it."""
        while True:
            try:
                return self._chat_once(
                    messages,
                    tools=tools,
                    require_tool=require_tool,
                    json_mode=json_mode,
                    role=role,
                    step=step,
                    **kw,
                )
            except (QuotaExhausted, ProviderUnavailable) as e:
                if not self._fallbacks:
                    self.quota_exhausted = isinstance(e, QuotaExhausted)
                    raise
                fb, client = self._fallbacks.pop(0)
                self._retry(f"{self.model} unavailable ({str(e)[:150]}); switching to fallback model {fb.model}")
                self._use(fb.model, client)

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        require_tool: bool = False,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_attempts: int = 6,
        max_rate_limit_wait_s: float = 300,
        role: str = "worker",
        step: int = 0,
    ) -> LLMResponse:
        """`max_attempts` bounds real failures (malformed output, 5xx, connection errors). Rate limits have
        their own budget in seconds of waiting: on a free tier a 429 is throttling, not a failure, and
        letting it consume `max_attempts` aborted a healthy run mid-task (first live eval)."""
        msgs = list(messages)
        attempt, rate_hits, rate_waited = 0, 0, 0.0
        transient: Exception | None = None  # the last failure, if it was the provider's rather than the model's
        out_tokens, effort = self.max_output_tokens, self.reasoning_effort
        while attempt < max_attempts:
            kwargs: dict = dict(model=self.api_model, messages=msgs, temperature=temperature, max_tokens=out_tokens)
            if effort:
                kwargs["reasoning_effort"] = effort
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "required" if (require_tool and self._tool_choice_required_ok) else "auto"
                if self._parallel_param_ok:
                    kwargs["parallel_tool_calls"] = False
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            t0_wall = time.time()
            t0_mono = time.monotonic()
            raw_headers: dict[str, Any] = {}
            try:
                if hasattr(self.client.chat.completions, "with_raw_response"):
                    raw = self.client.chat.completions.with_raw_response.create(**kwargs)
                    resp = raw.parse()
                    raw_headers = dict(raw.headers)
                else:
                    resp = self.client.chat.completions.create(**kwargs)
                    raw_headers = {}
            except openai.RateLimitError as e:
                latency_ms = round((time.monotonic() - t0_mono) * 1000, 1)
                err_headers = dict(e.response.headers) if hasattr(e, "response") and e.response is not None else {}
                err_body = ""
                if hasattr(e, "response") and e.response is not None:
                    err_body = getattr(e.response, "text", "") or ""
                if not err_body and hasattr(e, "body"):
                    err_body = json.dumps(e.body) if isinstance(e.body, dict) else str(e.body or "")
                if not err_body:
                    err_body = str(e)

                if self.usage_logger:
                    self.usage_logger.log_call(
                        timestamp_wall=t0_wall,
                        timestamp_mono=t0_mono,
                        latency_ms=latency_ms,
                        role=role,
                        step=step,
                        model=self.model,
                        base_url=str(getattr(self.client, "base_url", "")),
                        messages=msgs,
                        tools=tools,
                        success=False,
                        error_type="RateLimitError",
                        status_code=429,
                        error_body=err_body,
                        headers=err_headers,
                        max_tokens=out_tokens,
                        reasoning_effort=effort,
                    )

                rate_hits += 1
                wait = _retry_after(e) or min(60, 2**rate_hits)
                if rate_waited + wait > max_rate_limit_wait_s:
                    raise QuotaExhausted(
                        f"still rate limited after waiting {rate_waited:.0f}s (next wait {wait:.0f}s): {str(e)[:200]}"
                    ) from e
                rate_waited += wait
                self._retry(f"rate limited by provider, waiting {wait:.0f}s ({rate_waited:.0f}s so far)")
                self._sleep(wait)
                continue
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                latency_ms = round((time.monotonic() - t0_mono) * 1000, 1)
                err_headers = dict(e.response.headers) if hasattr(e, "response") and e.response is not None else {}
                if self.usage_logger:
                    self.usage_logger.log_call(
                        timestamp_wall=t0_wall,
                        timestamp_mono=t0_mono,
                        latency_ms=latency_ms,
                        role=role,
                        step=step,
                        model=self.model,
                        base_url=str(getattr(self.client, "base_url", "")),
                        messages=msgs,
                        tools=tools,
                        success=False,
                        error_type=type(e).__name__,
                        status_code=getattr(e, "status_code", None),
                        error_body=str(e),
                        headers=err_headers,
                        max_tokens=out_tokens,
                        reasoning_effort=effort,
                    )
                attempt += 1
                transient = e
                wait = min(30, 2**attempt)
                self._retry(f"provider error {type(e).__name__}, retrying in {wait}s")
                self._sleep(wait)
                continue
            except openai.BadRequestError as e:
                latency_ms = round((time.monotonic() - t0_mono) * 1000, 1)
                err_headers = dict(e.response.headers) if hasattr(e, "response") and e.response is not None else {}
                if self.usage_logger:
                    self.usage_logger.log_call(
                        timestamp_wall=t0_wall,
                        timestamp_mono=t0_mono,
                        latency_ms=latency_ms,
                        role=role,
                        step=step,
                        model=self.model,
                        base_url=str(getattr(self.client, "base_url", "")),
                        messages=msgs,
                        tools=tools,
                        success=False,
                        error_type="BadRequestError",
                        status_code=getattr(e, "status_code", 400),
                        error_body=str(e),
                        headers=err_headers,
                        max_tokens=out_tokens,
                        reasoning_effort=effort,
                    )
                transient = None
                body = str(e)
                if "tool_use_failed" in body and _empty_generation(e):
                    attempt += 1
                    out_tokens = min(out_tokens * 2, 32_768)
                    effort = "low" if effort else effort
                    self._retry(
                        f"model used its output budget on reasoning without calling a tool; retrying "
                        f"with max_tokens={out_tokens}" + (", reasoning_effort=low" if effort else "")
                    )
                    continue
                if "tool_use_failed" in body or "Failed to call a function" in body:
                    attempt += 1
                    self._retry(f"model produced a malformed tool call; retrying with a correction: {body[:300]}")
                    # Replace, don't stack: repeated corrections only grow the prompt.
                    msgs = list(messages) + [
                        {
                            "role": "user",
                            "content": "Your previous tool call was malformed. Call exactly ONE tool with valid JSON "
                            "arguments that match its schema.",
                        }
                    ]
                    continue
                if "parallel_tool_calls" in body and self._parallel_param_ok:
                    self._parallel_param_ok = False  # the agent loop already executes only the first call
                    self._retry("provider rejected parallel_tool_calls; retrying without it")
                    continue
                if "tool_choice" in body and self._tool_choice_required_ok:
                    self._tool_choice_required_ok = False
                    self._retry("provider rejected tool_choice=required; falling back to auto")
                    continue
                if json_mode and "json" in body.lower():
                    json_mode = False
                    self._retry("JSON mode failed validation; retrying without it")
                    continue
                raise LLMError(f"LLM request rejected: {body[:500]}") from e

            latency_ms = round((time.monotonic() - t0_mono) * 1000, 1)
            self.stats["calls"] += 1
            if resp.usage:
                self.stats["prompt_tokens"] += resp.usage.prompt_tokens or 0
                self.stats["completion_tokens"] += resp.usage.completion_tokens or 0
                details = getattr(resp.usage, "prompt_tokens_details", None)
                self.stats["cached_tokens"] += getattr(details, "cached_tokens", None) or 0
            if self.usage_logger:
                self.usage_logger.log_call(
                    timestamp_wall=t0_wall,
                    timestamp_mono=t0_mono,
                    latency_ms=latency_ms,
                    role=role,
                    step=step,
                    model=self.model,
                    base_url=str(getattr(self.client, "base_url", "")),
                    messages=msgs,
                    tools=tools,
                    success=True,
                    headers=raw_headers,
                    resp_usage=resp.usage,
                    max_tokens=out_tokens,
                    reasoning_effort=effort,
                )
            msg = resp.choices[0].message
            calls = []
            for tc in msg.tool_calls or []:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be an object")
                except ValueError:
                    args = {"__invalid_json__": tc.function.arguments}
                calls.append(ToolCall(tc.id, tc.function.name, args, tc.function.arguments or ""))
            return LLMResponse(msg.content or "", calls, resp.usage.model_dump() if resp.usage else {})
        if transient is not None:
            raise ProviderUnavailable(f"{self.model} failed {max_attempts} times: {str(transient)[:200]}")
        raise LLMError(f"LLM call failed after {max_attempts} attempts")

    def _retry(self, msg: str) -> None:
        self.stats["retries"] += 1
        log.warning(msg)
        if self.on_retry:
            self.on_retry(msg)


def _empty_generation(e: openai.APIStatusError) -> bool:
    """Groq reports a tool call that was never written as tool_use_failed with failed_generation=''."""
    body = e.body if isinstance(e.body, dict) else {}
    err = body.get("error", body) if isinstance(body.get("error", body), dict) else {}
    if "failed_generation" in err:
        return not err["failed_generation"]
    return bool(re.search(r"failed_generation['\"]?\s*:\s*(''|\"\")", str(e)))


_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)(ms|s))?")


def _duration_s(text: str) -> float | None:
    """'7h8m49.92s' -> 25729.92, '6m29.6s', '350ms', '46179s'. None if nothing parses."""
    m = _DURATION.fullmatch(text.strip())
    if not m or not any(m.groups()):
        return None
    h, mins, num, unit = m.groups()
    secs = float(num or 0) / (1000 if unit == "ms" else 1)
    return int(h or 0) * 3600 + int(mins or 0) * 60 + secs


def _retry_after(e: openai.APIStatusError) -> float | None:
    """How long the provider asks us to wait. A long wait (a daily quota) must be read in full: reading only
    the seconds part of Groq's '7h8m49s' made the client retry a model that was blocked for hours."""
    try:
        h = e.response.headers.get("retry-after")
        if h:
            return float(h) + 0.5
    except (AttributeError, ValueError):  # no/odd header: fall back to parsing the message
        pass
    text = str(e)
    for pattern in (r"try again in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+m?s)?)", r"retryDelay'?\"?:\s*'?\"?([\d.]+s)"):
        m = re.search(pattern, text)  # Groq: "try again in 7h8m49.92s"; Gemini: 'retryDelay': '46179s'
        if m and (v := _duration_s(m.group(1))) is not None:
            return v + 0.5
    return None


def parse_json(text: str) -> dict:
    """Lenient JSON extraction from model text (handles code fences / leading prose)."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON object in model output")
