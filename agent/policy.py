"""Action policy: decides, BEFORE execution, whether an action may run, needs a human, or is blocked.

This is deliberately code, not prompt. The LLM can be talked into anything; a regex on the
button it is about to press cannot. The model can still *volunteer* to ask (ask_human tool),
but irreversible actions are gated regardless of what the model thinks.

Modes:
  autonomous  - no approvals (blocked actions still blocked)
  balanced    - approval for high-risk actions (payments, deletion, money movement)   [default]
  supervised  - approval also for any data-changing submit (save/create/update/send)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HIGH_RISK = re.compile(r"\b(pay|paid|payment|delete|remove|wire|transfer|refund|approve|cancel|terminate)\b", re.I)
WRITE = re.compile(r"\b(save|submit|create|update|send|confirm|post|apply|add)\b", re.I)


@dataclass
class Decision:
    verdict: str  # allow | approve | deny
    reason: str = ""
    risk: str = "low"


def evaluate(tool: str, args: dict, snapshot, mode: str = "balanced") -> Decision:
    if tool == "browser_click" and snapshot is not None:
        el = snapshot.element(int(args.get("element_id", -1))) if str(args.get("element_id", "")).isdigit() else None
        if el is None:
            return Decision("allow")  # browser layer will report the bad id
        label = " ".join(str(el.get(k, "")) for k in ("text", "label", "href"))
        is_control = el["tag"] == "button" or el.get("type") in ("submit", "button")
        if is_control and HIGH_RISK.search(label):
            risk = "high"
            if mode != "autonomous":
                return Decision("approve", f'Irreversible action: pressing "{label.strip()}" on {snapshot.url}', risk)
        if is_control and WRITE.search(label) and not snapshot.has_password:
            if mode == "supervised":
                return Decision(
                    "approve", f'Data-changing action: pressing "{label.strip()}" on {snapshot.url}', "medium"
                )
            return Decision("allow", risk="medium")
    if tool == "write_file":
        p = str(args.get("path", ""))
        if p.startswith("/") or ".." in p:
            return Decision("deny", "Files may only be written inside the workspace directory")
    return Decision("allow")
