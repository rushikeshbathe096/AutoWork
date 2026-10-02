"""Independent verification of the worker's claim.

WHY a separate auditor: the worker saying "done" is not evidence. The auditor gets a fresh context
(none of the worker's reasoning, so it cannot inherit its mistakes), is told not to trust the claim,
and its browser is read-only AT THE NETWORK LAYER (see BrowserSession.read_only), so it can look but
cannot "fix" what it is checking.

It shares the worker's browser *context* (cookies) so it doesn't have to log in again. Sharing cookies
does not weaken read-only: that is enforced per request, not per session.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import prompts
from .browser import ApprovalRequired, BrowserSession
from .context import tool_call
from .interfaces import LLM, Browser, Emit, Message
from .tools import VERIFIER_TOOLS, ToolBox, ToolResult, tool_args_preview
from .vault import Redactor, Vault

MAX_AUDIT_STEPS = 10


@dataclass
class Claim:
    task: str
    success_criteria: list[str]
    summary: str
    evidence: list[str]
    facts: dict[str, str]


class Verifier:
    def __init__(
        self,
        llm: LLM,
        emit: Emit,
        workspace: Path,
        shots_dir: Path,
        vault: Vault | None,
        redactor: Redactor,
        over_budget: Callable[[], str | None],
        rel: Callable[[str | None], str | None],
    ):
        self.llm, self.emit, self.workspace, self.shots_dir = llm, emit, workspace, shots_dir
        self.vault, self.redactor, self.over_budget, self.rel = vault, redactor, over_budget, rel

    def verify(self, claim: Claim, worker_browser: Browser) -> dict:
        """Returns {'passed', 'reason', 'evidence'} plus 'inconclusive' when no verdict was reached."""
        self.emit("verify_start", {"summary": claim.summary})
        login_paths = frozenset(self.vault.login_paths()) if self.vault else frozenset()
        vb = BrowserSession(self.shots_dir, read_only=True, login_paths=login_paths).start(
            shared_context=worker_browser.context
        )
        tools = ToolBox(vb, self.workspace, vault=self.vault, redactor=self.redactor)
        msgs = self._opening(claim)
        try:
            for step in range(1, MAX_AUDIT_STEPS + 1):
                if why := self.over_budget():
                    self.emit("warning", {"message": f"Verification stopped: {why}"})
                    break
                r = self.llm.chat(msgs, tools=VERIFIER_TOOLS, require_tool=True)
                if not r.tool_calls:
                    msgs.append({"role": "user", "content": "Call a tool (verdict when done)."})
                    continue
                c = r.tool_calls[0]
                if c.name == "verdict":
                    v = {
                        "passed": bool(c.arguments.get("passed")),
                        "reason": c.arguments.get("reason", ""),
                        "evidence": c.arguments.get("evidence", []),
                    }
                    self.emit("verify_result", v)
                    return v
                try:
                    res = tools.run(c.name, c.arguments, step)
                except ApprovalRequired as e:  # cannot happen in read-only mode; handled for completeness
                    res = ToolResult(f"BLOCKED: {e}", "blocked", ok=False)
                self.emit(
                    "verify_step",
                    {
                        "tool": c.name,
                        "args": tool_args_preview(c.arguments),
                        "ok": res.ok,
                        "text": res.text[:1500],
                        "screenshot": self.rel(res.screenshot),
                    },
                )
                msgs.append(
                    {
                        "role": "assistant",
                        "content": r.content or None,
                        "tool_calls": [tool_call(c.id, c.name, c.raw_arguments)],
                    }
                )
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": res.text[:5000]})
        finally:
            vb.close()
        v = {"passed": False, "inconclusive": True, "reason": "Auditor did not reach a verdict", "evidence": []}
        self.emit("verify_result", v)
        return v

    @staticmethod
    def _opening(claim: Claim) -> list[Message]:
        criteria = "\n".join(f"- {c}" for c in claim.success_criteria) or "- (derive from the task)"
        return [
            {"role": "system", "content": prompts.VERIFIER},
            {
                "role": "user",
                "content": f"USER TASK:\n{claim.task}\n\nSUCCESS CRITERIA:\n{criteria}\n\n"
                f"WORKER'S CLAIM:\n{claim.summary}\nEvidence claimed: {claim.evidence}\n\n"
                f"Facts the worker recorded: {json.dumps(claim.facts)}\n\n"
                "Start page: http://localhost:8001/",
            },
        ]
