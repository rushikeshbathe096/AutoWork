"""The agent: plan -> (think -> gate -> act -> observe -> record)* -> finish -> independent verify -> learn.

Key reliability mechanisms, all in code rather than in the prompt:
  * policy gate before every action (agent/policy.py), with human approval
  * stuck detection: repeated identical action on an unchanged page, and consecutive-error streaks,
    escalate first to a warning, then to the human
  * context compression: old observations shrink to one line; working memory carries facts forward
  * step budget with a wrap-up warning
  * verification by a separate auditor with a fresh context and a network-enforced read-only browser;
    a failed audit sends the worker back to fix things (bounded rounds)
"""
from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from . import policy, prompts
from .browser import ApprovalRequired, BrowserSession
from .llm import LLMClient, LLMError, parse_json
from .memory import Playbook, WorkingMemory
from .tools import VERIFIER_TOOLS, WORKER_TOOLS, ToolBox, ToolResult, tool_args_preview
from .vault import Redactor, Vault

log = logging.getLogger("autowork.agent")

SCHEMA_VERSION = 1  # bump when the shape of events.jsonl / report.json changes


class Human(Protocol):
    def ask(self, kind: str, question: str, options: list[str] | None = None) -> dict:
        """kind: 'approval' -> {'approved': bool, 'comment': str}; 'clarification' -> {'answer': str}"""


@dataclass
class Report:
    run_id: str
    task: str
    status: str                 # verified | unverified | failed | needs_user | budget_exhausted | error
    summary: str
    evidence: list[str] = field(default_factory=list)
    verification: dict | None = None
    memory: dict = field(default_factory=dict)
    steps: int = 0
    duration_s: float = 0.0
    llm: dict = field(default_factory=dict)
    screenshots: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION


@dataclass
class Turn:
    """One assistant tool call + its result, kept so the prompt can be rebuilt with compression."""
    content: str
    call_id: str
    name: str
    raw_args: str
    result: ToolResult | None = None
    extra_calls: list = field(default_factory=list)  # parallel calls we refused to execute


class Agent:
    def __init__(self, llm: LLMClient, human: Human, workspace: Path, runs_dir: Path, playbook: Playbook | None,
                 emit: Callable[[str, dict], None] = lambda t, d: None, mode: str = "balanced", max_steps: int = 40,
                 headless: bool = True, keep_full_observations: int = 2, max_verify_rounds: int = 2,
                 vault: Vault | None = None, max_tokens_total: int = 400_000, max_active_seconds: float = 1800):
        self.llm, self.human, self.workspace, self.playbook = llm, human, workspace, playbook
        self.mode, self.max_steps, self.headless = mode, max_steps, headless
        self._sink = emit
        self.vault = vault
        self.redactor = Redactor(vault.secret_values() if vault else [])
        self.max_tokens_total, self.max_active_seconds = max_tokens_total, max_active_seconds
        self._t0 = time.monotonic()
        self._human_wait_s = 0.0
        self._seq = 0
        self.keep_full = keep_full_observations
        self.max_verify_rounds = max_verify_rounds
        self.run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.run_dir = runs_dir / self.run_id
        self.memory = WorkingMemory()
        self.turns: list[Turn] = []
        self.step = 0
        llm.on_retry = lambda m: self.emit("llm_retry", {"message": m})

    def emit(self, kind: str, data: dict) -> None:
        """Single exit point for the structured trace: redact secrets, persist to events.jsonl, forward.
        WHY one place: every log line and UI message passes here, so redaction cannot be forgotten."""
        data = self.redactor.obj(data)
        self._seq += 1
        try:
            with open(self.run_dir / "events.jsonl", "a") as f:
                f.write(json.dumps({"schema": SCHEMA_VERSION, "seq": self._seq, "t": round(time.time(), 2),
                                    "type": kind, "data": data}, default=str) + "\n")
        except OSError as e:
            log.warning("could not persist event: %s", e)
        self._sink(kind, data)

    def _budget_exceeded(self) -> str | None:
        """Hard limits so a confused model cannot burn money or run forever. Human wait time is
        excluded from the clock: waiting 10 minutes for an approval is not the agent misbehaving."""
        used = self.llm.stats.get("prompt_tokens", 0) + self.llm.stats.get("completion_tokens", 0)
        if used > self.max_tokens_total:
            return f"token budget exhausted ({used} > {self.max_tokens_total})"
        active = time.monotonic() - self._t0 - self._human_wait_s
        if active > self.max_active_seconds:
            return f"time budget exhausted ({active:.0f}s > {self.max_active_seconds:.0f}s)"
        return None

    # ======================================================================= public
    def run(self, task: str) -> Report:
        t0 = time.time()
        self._t0 = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.emit("start", {"run_id": self.run_id, "task": task, "mode": self.mode, "model": self.llm.model})
        self.browser = BrowserSession(self.run_dir / "shots", headless=self.headless,
                                      gate_high_risk=self.mode != "autonomous").start()
        self.tools = ToolBox(self.browser, self.workspace, self.memory, self.vault, self.redactor)
        try:
            report = self._run(task)
        except Exception as e:  # noqa: BLE001 - last line of defence: always return a report
            self.emit("error", {"message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
            report = self._report(task, "error", f"The run crashed: {type(e).__name__}: {e}", [])
        finally:
            self.browser.close()
        report.duration_s = round(time.time() - t0, 1)
        report.llm = dict(self.llm.stats)
        safe = self.redactor.obj(asdict(report))
        (self.run_dir / "report.json").write_text(json.dumps(safe, indent=2))
        self.emit("final", safe)
        return report

    # ======================================================================= phases
    def _run(self, task: str) -> Report:
        plan = self._plan(task)
        brief = self._brief(task, plan)
        verify_rounds = 0
        while True:
            fin = self._execute(brief)
            if fin is None:
                why = self._budget_exceeded() or f"step limit ({self.max_steps}) reached"
                return self._report(task, "budget_exhausted",
                                    f"Stopped after {self.step} steps without finishing: {why}.", [])
            status, summary, evidence = fin.get("status"), fin.get("summary", ""), fin.get("evidence", []) or []
            if status != "done":
                return self._report(task, "needs_user" if status == "needs_user" else "failed", summary, evidence)
            verdict = self._verify(task, plan, summary, evidence)
            if verdict.get("passed"):
                self._learn(task)
                return self._report(task, "verified", summary, evidence, verdict)
            verify_rounds += 1
            if verify_rounds > self.max_verify_rounds or verdict.get("inconclusive"):
                return self._report(task, "unverified", summary, evidence, verdict)
            self.emit("warning", {"message": "Verification failed, sending the worker back to fix it",
                                  "reason": verdict.get("reason")})
            self._inject(f"INDEPENDENT AUDIT FAILED: {verdict.get('reason')}\nAuditor evidence: "
                         f"{verdict.get('evidence')}\nInvestigate, fix the problem, and call finish again.")

    def _plan(self, task: str) -> dict:
        sys = prompts.PLANNER.replace("{playbook}", self._playbook_block())
        try:
            r = self.llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": task}], json_mode=True)
            plan = parse_json(r.content)
        except (LLMError, ValueError) as e:
            self.emit("warning", {"message": f"Planning failed ({e}); continuing without a plan"})
            plan = {"goal": task, "success_criteria": [], "plan": [], "assumptions": [], "blocking_questions": []}
        self.emit("plan", plan)
        answers = []
        for q in (plan.get("blocking_questions") or [])[:2]:
            a = self._ask_human("clarification", q)
            answers.append(f"Q: {q}\nA: {a.get('answer', '')}")
        plan["clarifications"] = answers
        return plan

    def _brief(self, task: str, plan: dict) -> str:
        parts = [f"TASK FROM USER:\n{task}", f"GOAL: {plan.get('goal', task)}"]
        if plan.get("success_criteria"):
            parts.append("SUCCESS CRITERIA:\n" + "\n".join(f"- {c}" for c in plan["success_criteria"]))
        if plan.get("plan"):
            parts.append("INITIAL PLAN (adapt as needed):\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan["plan"])))
        if plan.get("clarifications"):
            parts.append("CLARIFICATIONS FROM USER:\n" + "\n".join(plan["clarifications"]))
        parts.append("Start page: http://localhost:8001/ . Begin.")
        return "\n\n".join(parts)

    # ======================================================================= main loop
    def _execute(self, brief: str) -> dict | None:
        self.brief = brief
        sys = prompts.WORKER.replace("{playbook}", self._playbook_block()).replace(
            "{sites}", ", ".join(self.vault.sites()) if self.vault else "none configured")
        seen_actions: dict[str, int] = {}
        error_streak = 0
        while self.step < self.max_steps:
            if why := self._budget_exceeded():
                self.emit("warning", {"message": f"Stopping: {why}"})
                return None
            self.step += 1
            messages = self._messages(sys)
            resp = self.llm.chat(messages, tools=WORKER_TOOLS, require_tool=True)
            if not resp.tool_calls:
                self.emit("thought", {"step": self.step, "text": resp.content})
                self._inject("You must respond with a tool call. If the task is complete, call finish.")
                continue
            call, extras = resp.tool_calls[0], resp.tool_calls[1:]
            turn = Turn(resp.content.strip(), call.id, call.name, call.raw_arguments or json.dumps(call.arguments),
                        extra_calls=extras)
            self.turns.append(turn)
            if turn.content:
                self.emit("thought", {"step": self.step, "text": turn.content})
            self.emit("action", {"step": self.step, "tool": call.name, "args": tool_args_preview(call.arguments)})

            if call.name == "finish":
                turn.result = ToolResult("Finish received; handing over to verification.", "finish called")
                return call.arguments

            # --- stuck detection: same action on the same page state
            fp = self.browser.last.fingerprint() if self.browser.last else ""
            sig = f"{call.name}|{json.dumps(call.arguments, sort_keys=True)}|{fp}"
            seen_actions[sig] = seen_actions.get(sig, 0) + 1
            if seen_actions[sig] >= 4 and call.name not in ("remember",):
                ans = self._ask_human("clarification",
                                      f"I seem to be stuck: I've tried `{call.name}` {seen_actions[sig] - 1} times on "
                                      f"the same page without progress. Last reasoning: {turn.content[:300]!r}. "
                                      "How should I proceed?")
                turn.result = ToolResult(f"Action NOT executed (loop detected). Human guidance: {ans.get('answer')}",
                                         "loop detected; human consulted")
                seen_actions[sig] = 0
                continue

            result = self._act(call.name, call.arguments, turn.content)
            if seen_actions[sig] == 3:
                result.text += ("\n\nWARNING: you have now performed this exact action 3 times on an unchanged page. "
                                "It is not working. Try a different approach.")
                self.emit("warning", {"message": f"Repeated action detected: {call.name}"})
            turn.result = result
            self.emit("observation", {"step": self.step, "tool": call.name, "ok": result.ok,
                                      "text": result.text[:4000], "screenshot": self._rel(result.screenshot)})
            if call.name == "remember":
                self.emit("memory", self.memory.as_dict())

            error_streak = 0 if result.ok else error_streak + 1
            if error_streak == 3:
                result.text += ("\n\nNOTE: 3 consecutive actions have failed. Step back: re-read the page, check your "
                                "assumptions (are you logged in? right page? right format?), or ask the human.")
            if error_streak >= 6:
                ans = self._ask_human("clarification", f"I've hit {error_streak} failures in a row. Latest: "
                                                       f"{result.short}. How should I proceed?")
                self._inject(f"Human guidance after repeated failures: {ans.get('answer')}")
                error_streak = 0
            if self.step == int(self.max_steps * 0.8):
                self._inject(f"You have used {self.step}/{self.max_steps} steps. Wrap up: complete the essential "
                             "part, verify, and call finish (status failed/needs_user if you cannot complete).")
        return None

    def _act(self, name: str, args: dict, reasoning: str) -> ToolResult:
        if name == "ask_human":
            a = self._ask_human("clarification", args.get("question", ""), args.get("options"))
            return ToolResult(f"Human answered: {a.get('answer', '')}", f"human answered: {a.get('answer', '')[:200]}")
        decision = policy.evaluate(name, args, self.browser.last, self.mode)
        if decision.verdict == "deny":
            self.emit("policy", {"verdict": "deny", "reason": decision.reason})
            return ToolResult(f"BLOCKED by policy: {decision.reason}", "blocked by policy", ok=False)
        if decision.verdict == "approve":
            self.emit("policy", {"verdict": "approve", "reason": decision.reason, "risk": decision.risk})
            denied = self._approve_bound_action(name, args, decision.reason, reasoning)
            if denied:
                return denied
            self.browser.preapprove("*")  # the human approved this button: allow the one request it sends
        try:
            return self.tools.run(name, args, self.step)
        except ApprovalRequired as e:
            return self._approve_network_request(name, args, e.key, reasoning)

    def _approve_bound_action(self, name: str, args: dict, reason: str, reasoning: str) -> ToolResult | None:
        """Ask for approval of one exact action on one exact page state. If the page changed while the human
        was deciding, the approval no longer describes what would happen, so we ask again (max 3 times).
        Returns a denial ToolResult, or None when approved for the current page."""
        for _ in range(3):
            snap = self.browser.last
            fp = snap.fingerprint() if snap else ""
            action = {"tool": name, "args": args, "url": snap.url if snap else "", "page_fingerprint": fp}
            a = self._ask_human("approval", f"{reason}\n\nAgent's reasoning: {reasoning or '(none given)'}",
                                screenshot=snap.screenshot if snap else None, action=action)
            if not a.get("approved"):
                return ToolResult(f"DENIED by human: {a.get('comment') or 'no reason given'}. Do not retry this "
                                  "action. Continue without it or finish with status needs_user.", "denied by human",
                                  ok=False)
            now = self.browser.snapshot("approval-recheck") if snap else None
            if now is None or now.fingerprint() == fp:
                return None
            self.emit("warning", {"message": "Page changed while waiting for approval; asking again",
                                  "before": fp, "after": now.fingerprint()})
            reason = f"(Re-approval: the page changed) {reason}"
        return ToolResult("DENIED: the page kept changing during approval; not executed.", "approval unstable", ok=False)

    def _approve_network_request(self, name: str, args: dict, key: str, reasoning: str) -> ToolResult:
        """The browser aborted a high-risk request (e.g. POST /erp/bills/2/pay) that no button-level approval
        covered, e.g. a form submitted with Enter or by page JavaScript. Ask, then replay the action once
        with exactly that request allowed."""
        self.emit("policy", {"verdict": "approve", "reason": f"network gate: {key}", "risk": "high"})
        snap = self.browser.last
        a = self._ask_human("approval", f"The action {name} {tool_args_preview(args)} tries to send the high-risk "
                                        f"request {key}.\n\nAgent's reasoning: {reasoning or '(none given)'}",
                            screenshot=snap.screenshot if snap else None,
                            action={"tool": name, "args": args, "request": key})
        if not a.get("approved"):
            return ToolResult(f"DENIED by human: the request {key} was not sent. Do not retry it.",
                              "denied by human", ok=False)
        self.browser.preapprove(key)
        try:
            return self.tools.run(name, args, self.step)
        except ApprovalRequired as e:
            return ToolResult(f"BLOCKED: the action sent a different high-risk request ({e.key}) than the one "
                              f"approved ({key}); nothing was sent.", "approval mismatch", ok=False)

    # ======================================================================= context management
    def _messages(self, system: str) -> list[dict]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": self.brief}]
        obs_idx = [i for i, t in enumerate(self.turns) if t.result and len(t.result.text) > 400]
        keep = set(obs_idx[-self.keep_full:])
        for i, t in enumerate(self.turns):
            if t.name == "__inject__":
                msgs.append({"role": "user", "content": t.content})
                continue
            content = t.content if i >= len(self.turns) - 6 else t.content[:200]
            calls = [{"id": t.call_id, "type": "function", "function": {"name": t.name, "arguments": t.raw_args}}]
            calls += [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments or "{}"}}
                      for c in t.extra_calls]
            msgs.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            res = t.result.text if (t.result and (i in keep or len(t.result.text) <= 400)) else (
                t.result.short if t.result else "(no result)")
            msgs.append({"role": "tool", "tool_call_id": t.call_id, "content": res})
            for c in t.extra_calls:
                msgs.append({"role": "tool", "tool_call_id": c.id,
                             "content": "Not executed: only one tool call per turn is allowed."})
        msgs.append({"role": "user", "content": f"[status] step {self.step}/{self.max_steps}. WORKING MEMORY:\n"
                                                f"{self.memory.render()}\nContinue: one sentence of reasoning, then one tool call."})
        return msgs

    def _inject(self, text: str):
        self.turns.append(Turn(text, "", "__inject__", ""))

    # ======================================================================= verification
    def _verify(self, task: str, plan: dict, summary: str, evidence: list) -> dict:
        self.emit("verify_start", {"summary": summary})
        login_paths = frozenset(self.vault.login_paths()) if self.vault else frozenset()
        vb = BrowserSession(self.run_dir / "verify", read_only=True, login_paths=login_paths).start(
            shared_context=self.browser.context)
        tb = ToolBox(vb, self.workspace, vault=self.vault, redactor=self.redactor)
        criteria = "\n".join(f"- {c}" for c in plan.get("success_criteria", [])) or "- (derive from the task)"
        msgs = [{"role": "system", "content": prompts.VERIFIER},
                {"role": "user", "content": f"USER TASK:\n{task}\n\nSUCCESS CRITERIA:\n{criteria}\n\nWORKER'S CLAIM:\n"
                                            f"{summary}\nEvidence claimed: {evidence}\n\nFacts the worker recorded: "
                                            f"{json.dumps(self.memory.as_dict())}\n\nStart page: http://localhost:8001/"}]
        try:
            for vstep in range(1, 11):
                if why := self._budget_exceeded():
                    self.emit("warning", {"message": f"Verification stopped: {why}"})
                    break
                r = self.llm.chat(msgs, tools=VERIFIER_TOOLS, require_tool=True)
                if not r.tool_calls:
                    msgs.append({"role": "user", "content": "Call a tool (verdict when done)."})
                    continue
                c = r.tool_calls[0]
                if c.name == "verdict":
                    v = {"passed": bool(c.arguments.get("passed")), "reason": c.arguments.get("reason", ""),
                         "evidence": c.arguments.get("evidence", [])}
                    self.emit("verify_result", v)
                    return v
                try:
                    res = tb.run(c.name, c.arguments, vstep)
                except ApprovalRequired as e:  # unreachable in read-only mode; kept explicit
                    res = ToolResult(f"BLOCKED: {e}", "blocked", ok=False)
                self.emit("verify_step", {"tool": c.name, "args": tool_args_preview(c.arguments), "ok": res.ok,
                                          "text": res.text[:1500], "screenshot": self._rel(res.screenshot)})
                msgs.append({"role": "assistant", "content": r.content or None, "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments or "{}"}}]})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": res.text[:5000]})
        finally:
            vb.close()
        v = {"passed": False, "inconclusive": True, "reason": "Auditor did not reach a verdict", "evidence": []}
        self.emit("verify_result", v)
        return v

    # ======================================================================= learning
    def _learn(self, task: str):
        if not self.playbook:
            return
        trace = []
        for t in self.turns:
            if t.name == "__inject__":
                trace.append(f"[system note] {t.content[:200]}")
            elif t.result:
                trace.append(f"{t.name}({t.raw_args[:200]}) -> {t.result.short[:200]}")
        try:
            r = self.llm.chat([{"role": "system", "content": prompts.DISTILL},
                               {"role": "user", "content": f"Task: {task}\n\nTrace:\n" + "\n".join(trace[-60:])}],
                              json_mode=True)
            notes = parse_json(r.content).get("notes", [])[:5]
            self.playbook.add([n for n in notes if isinstance(n, str)], self.run_id)
            self.emit("learned", {"notes": notes})
        except (LLMError, ValueError) as e:
            self.emit("warning", {"message": f"Could not distill playbook notes: {e}"})

    # ======================================================================= helpers
    def _ask_human(self, kind: str, question: str, options: list | None = None, screenshot: str | None = None,
                   action: dict | None = None) -> dict:
        self.emit("human_request", {"kind": kind, "question": question, "options": options or [],
                                    "screenshot": self._rel(screenshot), "action": action})
        started = time.monotonic()
        ans = self.human.ask(kind, self.redactor.text(question), options)
        self._human_wait_s += time.monotonic() - started
        self.emit("human_response", {"kind": kind, **ans})
        return ans

    def _playbook_block(self) -> str:
        notes = self.playbook.render() if self.playbook else ""
        return prompts.PLAYBOOK_BLOCK.replace("{notes}", notes) if notes else ""

    def _rel(self, p: str | None) -> str | None:
        if not p:
            return None
        try:
            return str(Path(p).relative_to(self.run_dir.parent))
        except ValueError:
            return p

    def _report(self, task, status, summary, evidence, verification=None) -> Report:
        shots = sorted(str(p.relative_to(self.run_dir.parent)) for p in (self.run_dir / "shots").glob("*.png"))
        return Report(self.run_id, task, status, summary, list(evidence), verification, self.memory.as_dict(),
                      self.step, screenshots=shots[-6:])
