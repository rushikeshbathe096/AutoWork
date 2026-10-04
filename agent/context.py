"""Prompt construction with context compression.

WHY: full page observations are 1-3k tokens each. Keeping all of them would blow the context
(and Groq's tokens-per-minute limit) after ~15 steps. Only the most recent observations stay in
full; older ones are replaced by their one-line `short` form. Facts the agent needs later are
kept in WorkingMemory, which is re-rendered into every prompt.

Two schemes (AUTOWORK_CONTEXT):
  classic  every past turn stays a native assistant tool-call + tool-result pair.
  digest   only the recent turns stay native; older ones become one line each in a single EARLIER STEPS
           message. Measured on a replayed run, ~44% of the history was per-turn message scaffolding (role,
           call id, function wrapper), which a one-line entry doesn't pay. Errors, approvals/denials, human
           answers and injected notes are kept in full text in the digest; only page observations shrink to
           their short form (URL, title, HTTP status, alerts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .interfaces import Message
from .llm import ToolCall
from .tools import ToolResult

INJECT = "__inject__"  # a Turn that is a system note to the model, not a tool call
FULL_IF_SHORTER_THAN = 400  # small results are cheap: always keep them verbatim
RECENT_REASONING_TURNS = 6  # older assistant reasoning is truncated to 200 chars
RECENT_NATIVE_TURNS = 3  # digest: turns kept as native tool-call/result pairs (extended to cover keep_full)
DIGEST_ARGS_CHARS = 160


@dataclass
class Turn:
    """One assistant tool call + its result, kept so the prompt can be rebuilt each step."""

    content: str
    call_id: str
    name: str
    raw_args: str
    result: ToolResult | None = None
    extra_calls: list[ToolCall] = field(default_factory=list)  # parallel calls we refused to execute


def note(text: str) -> Turn:
    return Turn(text, "", INJECT, "")


def tool_call(call_id: str, name: str, raw_args: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw_args or "{}"}}


def build_messages(
    system: str, brief: str, turns: list[Turn], status: str, keep_full: int = 2, scheme: str = "classic"
) -> list[Message]:
    """system + task brief + compressed history + a trailing status message (step budget, memory)."""
    msgs: list[Message] = [{"role": "system", "content": system}, {"role": "user", "content": brief}]
    big = [i for i, t in enumerate(turns) if t.result and len(t.result.text) > FULL_IF_SHORTER_THAN]
    keep = set(big[-keep_full:]) if keep_full > 0 else set()
    start = 0
    if scheme == "digest":
        # Native tail: the last RECENT_NATIVE_TURNS turns, extended back so the kept-full observations stay native.
        start = min([max(0, len(turns) - RECENT_NATIVE_TURNS), *keep])
        if start:
            lines, n = [], 0
            for t in turns[:start]:
                n += t.name != INJECT  # number tool calls only, so notes don't shift the step numbers
                lines.append(_digest_line(n, t))
            msgs.append(
                {
                    "role": "user",
                    "content": "EARLIER STEPS (one line each, oldest first; facts you need are in WORKING MEMORY):\n"
                    + "\n".join(lines),
                }
            )
    elif scheme != "classic":
        raise ValueError(f"unknown context scheme {scheme!r}")
    for i, t in enumerate(turns):
        if i < start:
            continue
        if t.name == INJECT:
            msgs.append({"role": "user", "content": t.content})
            continue
        recent = i >= len(turns) - RECENT_REASONING_TURNS
        calls = [tool_call(t.call_id, t.name, t.raw_args)]
        calls += [tool_call(c.id, c.name, c.raw_arguments) for c in t.extra_calls]
        msgs.append(
            {"role": "assistant", "content": (t.content if recent else t.content[:200]) or None, "tool_calls": calls}
        )
        if t.result is None:
            res = "(no result)"
        elif i in keep or len(t.result.text) <= FULL_IF_SHORTER_THAN:
            res = t.result.text
        else:
            res = t.result.short
        msgs.append({"role": "tool", "tool_call_id": t.call_id, "content": res})
        for c in t.extra_calls:
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": c.id,
                    "content": "Not executed: only one tool call per turn is allowed.",
                }
            )
    msgs.append({"role": "user", "content": status})
    return msgs


def _digest_line(n: int, t: Turn) -> str:
    """One step as one line: `7. remember(key="x", value="y") -> Stored 'x'`. Notes keep their full text."""
    if t.name == INJECT:
        return "   note: " + " ".join(t.content.split())
    return f"{n}. {t.name}({_compact_args(t.raw_args)}) -> {_outcome(t.result)}"


def _compact_args(raw: str) -> str:
    try:
        a = json.loads(raw or "{}")
    except ValueError:
        a = None
    if not isinstance(a, dict):
        s = raw or ""
    elif isinstance(a.get("fields"), list):  # form fills: [id]=value pairs instead of nested JSON
        s = "fields: " + ", ".join(
            f"[{f.get('element_id')}]={json.dumps(f.get('value'), ensure_ascii=False)}"
            for f in a["fields"]
            if isinstance(f, dict)
        )
    else:
        s = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in a.items())
    return s if len(s) <= DIGEST_ARGS_CHARS else s[:DIGEST_ARGS_CHARS] + "…"


def _outcome(r: ToolResult | None) -> str:
    """Page observations shrink to their short form; everything else (errors, denials, human answers, memory
    writes, fill read-backs) is small and decision-relevant, so it is kept in full."""
    if r is None:
        return "(no result)"
    if r.text.startswith("URL: ") or len(r.text) > FULL_IF_SHORTER_THAN:
        text = r.short.removesuffix(" (old observation elided)")
    else:
        text = r.text
    text = " ".join(text.split())
    return text if r.ok else f"FAILED: {text}"
