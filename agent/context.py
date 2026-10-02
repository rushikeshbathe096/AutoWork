"""Prompt construction with context compression.

WHY: full page observations are 1-3k tokens each. Keeping all of them would blow the context
(and Groq's tokens-per-minute limit) after ~15 steps. Only the most recent observations stay in
full; older ones are replaced by their one-line `short` form. Facts the agent needs later are
kept in WorkingMemory, which is re-rendered into every prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .interfaces import Message
from .llm import ToolCall
from .tools import ToolResult

INJECT = "__inject__"  # a Turn that is a system note to the model, not a tool call
FULL_IF_SHORTER_THAN = 400  # small results are cheap: always keep them verbatim
RECENT_REASONING_TURNS = 6  # older assistant reasoning is truncated to 200 chars


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


def build_messages(system: str, brief: str, turns: list[Turn], status: str, keep_full: int = 2) -> list[Message]:
    """system + task brief + compressed history + a trailing status message (step budget, memory)."""
    msgs: list[Message] = [{"role": "system", "content": system}, {"role": "user", "content": brief}]
    big = [i for i, t in enumerate(turns) if t.result and len(t.result.text) > FULL_IF_SHORTER_THAN]
    keep = set(big[-keep_full:]) if keep_full > 0 else set()
    for i, t in enumerate(turns):
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
