"""LLM usage instrumentation and section-by-section prompt size breakdown.

WHY: Diagnose exactly which provider rate limit (RPM, TPM, RPD, TPD) is binding
and which parts of our prompts consume the most tokens, without modifying agent
behavior or leaking sensitive data.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import tiktoken

from .vault import Redactor

# Approximation using tiktoken cl100k_base as required
_ENC: tiktoken.Encoding | None
try:
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    _ENC = None


def count_tokens(text: str) -> int:
    """Approximate token count using tiktoken cl100k_base."""
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    # Fallback approximation: ~4 characters per token
    return max(1, len(text) // 4)


def _split_latest_observation(text: str) -> tuple[str, str, str]:
    """Split observation text into (dom_elements, page_text, other_meta)."""
    dom = ""
    page = ""
    other = ""

    dom_marker = "INTERACTIVE ELEMENTS:"
    page_marker = "PAGE TEXT:"

    if dom_marker in text and page_marker in text:
        parts = text.split(dom_marker, 1)
        other += parts[0]
        rest = parts[1]
        dom_parts = rest.split(page_marker, 1)
        dom = dom_marker + dom_parts[0]
        page = page_marker + dom_parts[1]
    elif dom_marker in text:
        parts = text.split(dom_marker, 1)
        other += parts[0]
        dom = dom_marker + parts[1]
    elif page_marker in text:
        parts = text.split(page_marker, 1)
        other += parts[0]
        page = page_marker + parts[1]
    elif "wrap_untrusted" in text or "UNTRUSTED_FILE" in text:
        page = text
    else:
        other = text

    return dom, page, other


def analyze_sections(
    role: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, int]]:
    """Decompose prompt messages and tools into the 10 canonical sections, measuring sizes only.

    Canonical sections:
      - system_prompt
      - tool_schemas
      - task_text
      - plan
      - working_memory
      - playbook_notes
      - conversation_history
      - latest_observation_dom
      - latest_observation_page_text
      - other
    """
    sec_text: dict[str, list[str]] = {
        "system_prompt": [],
        "tool_schemas": [],
        "task_text": [],
        "plan": [],
        "working_memory": [],
        "playbook_notes": [],
        "conversation_history": [],
        "latest_observation_dom": [],
        "latest_observation_page_text": [],
        "other": [],
    }

    # 1. Tool schemas
    if tools:
        sec_text["tool_schemas"].append(json.dumps(tools))

    # Helper to separate playbook notes from system prompt
    pb_marker = "Notes learned from previous successful runs"

    def split_system_playbook(sys_content: str) -> None:
        if pb_marker in sys_content:
            parts = sys_content.split(pb_marker, 1)
            sec_text["system_prompt"].append(parts[0])
            sec_text["playbook_notes"].append(pb_marker + parts[1])
        else:
            sec_text["system_prompt"].append(sys_content)

    if role == "planner":
        if messages:
            split_system_playbook(str(messages[0].get("content") or ""))
        if len(messages) > 1:
            sec_text["task_text"].append(str(messages[1].get("content") or ""))
        for m in messages[2:]:
            sec_text["other"].append(str(m.get("content") or ""))

    elif role == "distiller":
        if messages:
            sec_text["system_prompt"].append(str(messages[0].get("content") or ""))
        if len(messages) > 1:
            content = str(messages[1].get("content") or "")
            if "Task: " in content and "\n\nTrace:\n" in content:
                parts = content.split("\n\nTrace:\n", 1)
                task_part = parts[0].replace("Task: ", "", 1)
                sec_text["task_text"].append(task_part)
                sec_text["conversation_history"].append(parts[1])
                sec_text["other"].append("Task: \n\nTrace:\n")
            else:
                sec_text["conversation_history"].append(content)
        for m in messages[2:]:
            sec_text["conversation_history"].append(str(m.get("content") or ""))

    elif role == "verifier":
        if messages:
            sec_text["system_prompt"].append(str(messages[0].get("content") or ""))
        if len(messages) > 1:
            opening = str(messages[1].get("content") or "")
            # Opening has: USER TASK, SUCCESS CRITERIA, WORKER'S CLAIM, Facts, Sources
            # Parse sections out of opening
            task_m = re.search(r"USER TASK:\s*(.*?)(?=\n\nSUCCESS CRITERIA:|\Z)", opening, re.S)
            crit_m = re.search(r"SUCCESS CRITERIA:\s*(.*?)(?=\n\nWORKER'S CLAIM:|\Z)", opening, re.S)
            claim_m = re.search(r"WORKER'S CLAIM:\s*(.*?)(?=\n\nFacts the worker recorded:|\Z)", opening, re.S)
            facts_m = re.search(
                r"Facts the worker recorded:\s*(.*?)(?=\n\nSources the worker looked at:|\Z)", opening, re.S
            )

            if task_m:
                sec_text["task_text"].append(task_m.group(1))
            if crit_m:
                sec_text["plan"].append("SUCCESS CRITERIA:\n" + crit_m.group(1))
            if claim_m:
                sec_text["plan"].append("WORKER'S CLAIM:\n" + claim_m.group(1))
            if facts_m:
                sec_text["working_memory"].append(facts_m.group(1))

            # Remainder of opening goes to other
            sec_text["other"].append("USER TASK:\n\nStart page: http://localhost:8001/")

        # Audit steps history
        audit_turns = messages[2:]
        if audit_turns:
            # Older messages go to conversation history, latest tool observation gets split
            # Find the last tool message
            last_tool_idx = -1
            for idx in range(len(audit_turns) - 1, -1, -1):
                if audit_turns[idx].get("role") == "tool":
                    last_tool_idx = idx
                    break

            for idx, m in enumerate(audit_turns):
                if idx == last_tool_idx:
                    dom, page, other = _split_latest_observation(str(m.get("content") or ""))
                    if dom:
                        sec_text["latest_observation_dom"].append(dom)
                    if page:
                        sec_text["latest_observation_page_text"].append(page)
                    if other:
                        sec_text["other"].append(other)
                else:
                    role_tag = m.get("role", "")
                    content_str = str(m.get("content") or "")
                    tc = m.get("tool_calls")
                    if tc:
                        content_str += " " + json.dumps(tc)
                    if role_tag == "user" and content_str.startswith("[audit step"):
                        sec_text["other"].append(content_str)
                    else:
                        sec_text["conversation_history"].append(content_str)

    else:
        # Worker (and default fallback)
        if messages:
            split_system_playbook(str(messages[0].get("content") or ""))

        # Brief is messages[1]
        if len(messages) > 1:
            brief = str(messages[1].get("content") or "")
            task_m = re.search(r"TASK FROM USER:\s*(.*?)(?=\n\nGOAL:|\Z)", brief, re.S)
            goal_m = re.search(r"GOAL:\s*(.*?)(?=\n\nSUCCESS CRITERIA:|\Z)", brief, re.S)
            crit_m = re.search(r"SUCCESS CRITERIA:\s*(.*?)(?=\n\nINITIAL PLAN|\Z)", brief, re.S)
            plan_m = re.search(r"INITIAL PLAN [^:]*:\s*(.*?)(?=\n\nCLARIFICATIONS|\n\nStart page:|\Z)", brief, re.S)
            clar_m = re.search(r"CLARIFICATIONS FROM USER:\s*(.*?)(?=\n\nStart page:|\Z)", brief, re.S)

            if task_m:
                sec_text["task_text"].append(task_m.group(1))
            else:
                sec_text["task_text"].append(brief)

            plan_parts = []
            if goal_m:
                plan_parts.append("GOAL: " + goal_m.group(1))
            if crit_m:
                plan_parts.append("SUCCESS CRITERIA:\n" + crit_m.group(1))
            if plan_m:
                plan_parts.append("INITIAL PLAN:\n" + plan_m.group(1))
            if clar_m:
                plan_parts.append("CLARIFICATIONS:\n" + clar_m.group(1))
            if plan_parts:
                sec_text["plan"].append("\n\n".join(plan_parts))

            sec_text["other"].append("Start page: http://localhost:8001/ . Begin.")

        # Status message is messages[-1] if len(messages) > 2
        if len(messages) > 2:
            status_msg = str(messages[-1].get("content") or "")
            if "WORKING MEMORY:" in status_msg:
                parts = status_msg.split("WORKING MEMORY:", 1)
                sec_text["other"].append(parts[0])
                rest = parts[1]
                if "\nContinue:" in rest:
                    mem_part, cont_part = rest.split("\nContinue:", 1)
                    sec_text["working_memory"].append(mem_part)
                    sec_text["other"].append("\nContinue:" + cont_part)
                else:
                    sec_text["working_memory"].append(rest)
            else:
                sec_text["other"].append(status_msg)

        # Turns are messages[2:-1]
        turns_msgs = messages[2:-1] if len(messages) > 3 else []
        if turns_msgs:
            # The latest observation is the last tool message in turns_msgs
            last_tool_idx = -1
            for idx in range(len(turns_msgs) - 1, -1, -1):
                if turns_msgs[idx].get("role") == "tool":
                    last_tool_idx = idx
                    break

            for idx, m in enumerate(turns_msgs):
                if idx == last_tool_idx:
                    dom, page, other = _split_latest_observation(str(m.get("content") or ""))
                    if dom:
                        sec_text["latest_observation_dom"].append(dom)
                    if page:
                        sec_text["latest_observation_page_text"].append(page)
                    if other:
                        sec_text["other"].append(other)
                else:
                    content_str = str(m.get("content") or "")
                    tc = m.get("tool_calls")
                    if tc:
                        content_str += " " + json.dumps(tc)
                    sec_text["conversation_history"].append(content_str)

    # Compute characters and tokens for each section (sizes only, no content)
    result: dict[str, dict[str, int]] = {}
    for sec_name, chunks in sec_text.items():
        combined = "\n".join(c for c in chunks if c)
        result[sec_name] = {
            "chars": len(combined),
            "tokens": count_tokens(combined),
        }

    return result


def detect_provider(model: str, base_url: str) -> str:
    """Infer the LLM provider from the model prefix or base URL."""
    prefix, sep, _ = model.partition(":")
    if sep and prefix in ("gemini", "nvidia", "openrouter"):
        return prefix
    b = base_url.lower()
    if "groq.com" in b:
        return "groq"
    if "openai.com" in b:
        return "openai"
    if "together" in b:
        return "together"
    if "deepseek" in b:
        return "deepseek"
    if "anthropic" in b:
        return "anthropic"
    if "integrate.api.nvidia.com" in b:
        return "nvidia"
    if "openrouter.ai" in b:
        return "openrouter"
    if "generativelanguage.googleapis.com" in b:
        return "gemini"
    return "custom"


def extract_rate_limit_headers(headers: dict[str, Any] | None) -> dict[str, str]:
    """Generically extract all x-ratelimit-* and retry-after headers."""
    if not headers:
        return {}
    res: dict[str, str] = {}
    for k, v in headers.items():
        k_lower = str(k).lower()
        if k_lower.startswith("x-ratelimit-") or k_lower == "retry-after":
            res[k_lower] = str(v)
    return res


class UsageLogger:
    """Appends size-only, privacy-safe instrumentation to runs/<run_id>/llm_usage.jsonl."""

    def __init__(self, run_dir: Path, run_id: str, redactor: Redactor | None = None):
        self.run_dir = run_dir
        self.run_id = run_id
        self.redactor = redactor or Redactor([])
        self.log_path = self.run_dir / "llm_usage.jsonl"

    def log_call(
        self,
        *,
        timestamp_wall: float,
        timestamp_mono: float,
        latency_ms: float,
        role: str,
        step: int,
        model: str,
        base_url: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        success: bool,
        error_type: str | None = None,
        status_code: int | None = None,
        error_body: str = "",
        headers: dict[str, Any] | None = None,
        resp_usage: Any = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Write one usage entry to llm_usage.jsonl."""
        provider = detect_provider(model, base_url)
        rate_headers = extract_rate_limit_headers(headers)

        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        reasoning_tokens: int | None = None
        cached_tokens: int | None = None
        total_tokens: int | None = None

        if resp_usage is not None:
            if isinstance(resp_usage, dict):
                prompt_tokens = resp_usage.get("prompt_tokens")
                completion_tokens = resp_usage.get("completion_tokens")
                total_tokens = resp_usage.get("total_tokens")
                cached_tokens = (resp_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                reasoning_tokens = (resp_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            else:
                prompt_tokens = getattr(resp_usage, "prompt_tokens", None)
                completion_tokens = getattr(resp_usage, "completion_tokens", None)
                total_tokens = getattr(resp_usage, "total_tokens", None)
                cached_details = getattr(resp_usage, "prompt_tokens_details", None)
                if cached_details:
                    cached_tokens = getattr(cached_details, "cached_tokens", None)
                comp_details = getattr(resp_usage, "completion_tokens_details", None)
                if comp_details:
                    reasoning_tokens = getattr(comp_details, "reasoning_tokens", None)

        sections = analyze_sections(role, messages, tools)

        # Redact any strings that might contain sensitive info (never prompt or page content)
        safe_error_body = self.redactor.text(error_body) if error_body else ""
        safe_headers = self.redactor.obj(rate_headers)

        row: dict[str, Any] = {
            "timestamp_wall": round(timestamp_wall, 3),
            "timestamp_mono": round(timestamp_mono, 3),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(timestamp_wall))
            + f".{int((timestamp_wall % 1) * 1000):03d}Z",
            "run_id": self.run_id,
            "step": step,
            "role": role,
            "model": model,
            "provider": provider,
            "latency_ms": latency_ms,
            "success": success,
            "error_type": error_type,
            "status_code": status_code,
            "error_body": safe_error_body if safe_error_body else None,
            "rate_limit_headers": safe_headers,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens,
            "sections": sections,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort if reasoning_effort else None,
        }

        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass
