"""LLM client for any OpenAI-compatible chat-completions API (Groq by default; OpenAI, Together,
vLLM, Ollama... by setting LLM_BASE_URL / LLM_MODEL / LLM_API_KEY). Configuration comes from
agent.config.Settings.

Reliability concerns handled here (so the agent loop doesn't have to):
  * 429 rate limits / 5xx  -> exponential backoff honoring Retry-After
  * Groq `tool_use_failed` -> the model emitted a malformed tool call; retry with a nudge
  * tool_choice="required" unsupported -> fall back to "auto"
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

from .config import Settings

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


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        client: openai.OpenAI | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """`client` and `sleep` are injectable so retry behaviour can be unit-tested without network or waiting."""
        s = settings or Settings.from_env()
        self.model = s.llm_model
        if client is None:
            if not s.llm_api_key:
                raise LLMError("No LLM API key: set LLM_API_KEY (or GROQ_API_KEY) in .env, see .env.example")
            # max_retries=0: retries are handled below, with provider-specific logic the SDK doesn't know
            client = openai.OpenAI(
                api_key=s.llm_api_key, base_url=s.llm_base_url, max_retries=0, timeout=s.llm_timeout_s
            )
        self.client = client
        self._sleep = sleep
        self.on_retry: Callable[[str], None] | None = None
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0}
        self._tool_choice_required_ok = True

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        require_tool: bool = False,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_attempts: int = 6,
    ) -> LLMResponse:
        msgs = list(messages)
        for attempt in range(1, max_attempts + 1):
            kwargs: dict = dict(model=self.model, messages=msgs, temperature=temperature, max_tokens=2048)
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "required" if (require_tool and self._tool_choice_required_ok) else "auto"
                kwargs["parallel_tool_calls"] = False
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                wait = _retry_after(e) or min(60, 2**attempt)
                self._retry(f"rate limited by provider, waiting {wait:.0f}s (attempt {attempt})")
                self._sleep(wait)
                continue
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                wait = min(30, 2**attempt)
                self._retry(f"provider error {type(e).__name__}, retrying in {wait}s")
                self._sleep(wait)
                continue
            except openai.BadRequestError as e:
                body = str(e)
                if "tool_use_failed" in body or "Failed to call a function" in body:
                    self._retry("model produced a malformed tool call; retrying with a correction")
                    msgs = msgs + [
                        {
                            "role": "user",
                            "content": "Your previous tool call was malformed. Call exactly ONE tool with valid JSON "
                            "arguments that match its schema.",
                        }
                    ]
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

            self.stats["calls"] += 1
            if resp.usage:
                self.stats["prompt_tokens"] += resp.usage.prompt_tokens or 0
                self.stats["completion_tokens"] += resp.usage.completion_tokens or 0
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
        raise LLMError(f"LLM call failed after {max_attempts} attempts")

    def _retry(self, msg: str) -> None:
        self.stats["retries"] += 1
        log.warning(msg)
        if self.on_retry:
            self.on_retry(msg)


def _retry_after(e: openai.APIStatusError) -> float | None:
    try:
        h = e.response.headers.get("retry-after")
        if h:
            return float(h) + 0.5
    except (AttributeError, ValueError):  # no/odd header: fall back to parsing the message
        pass
    m = re.search(r"try again in ([\d.]+)(m?s)", str(e))
    if m:
        v = float(m.group(1))
        return (v / 1000 if m.group(2) == "ms" else v) + 0.5
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
