"""The agent: plan -> (think -> gate -> act -> observe -> record)* -> finish -> independent verify -> learn.

Reliability mechanisms, all in code rather than in the prompt:
  * plan validation: no run proceeds without a goal and success criteria (validate_plan)
  * policy gate before every action (policy.py) + network-level payment gate (browser.py), with
    approvals bound to the exact action and page state
  * stuck detection (stuck.py): repeated action on an unchanged page, consecutive-error streaks
  * context compression (context.py): old observations shrink to one line; working memory persists
  * budgets: steps, tokens, active wall-clock time
  * independent read-only verification (verifier.py), on its own model when one is
    configured; a failed audit sends the worker back (bounded)
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import policy, prompts
from .browser import ApprovalRequired, BrowserSession
from .context import Turn, build_messages, note
from .interfaces import LLM, Emit, Human
from .llm import LLMError, ProviderConfigError, QuotaExhausted, parse_json
from .memory import Playbook, WorkingMemory
from .provenance import Provenance
from .stuck import ErrorStreak, ProgressTracker, RepetitionDetector, Signal
from .tools import WORKER_TOOLS, ToolBox, ToolResult, tool_args_preview
from .usage import UsageLogger
from .vault import Redactor, Vault
from .verifier import Claim, Verifier
from .world import World, default_world

log = logging.getLogger("autowork.agent")

SCHEMA_VERSION = 1  # bump when the shape of events.jsonl / report.json changes
WRAP_UP_AT = 0.8  # fraction of the step budget at which the agent is told to wrap up


class PlanFailed(Exception):
    """The planner did not produce a valid plan, even after a retry. Without success criteria nothing can be
    verified, so the run stops instead of acting."""


def validate_plan(plan: object) -> str | None:
    """Why `plan` is unusable, or None. Required: a non-empty goal and a non-empty list of non-empty criteria
    strings. The optional lists, when present, must be lists of strings."""
    if not isinstance(plan, dict):
        return f"expected a JSON object, got {type(plan).__name__}"
    goal = plan.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        return "'goal' must be a non-empty string"
    crit = plan.get("success_criteria")
    if not isinstance(crit, list) or not crit:
        return "'success_criteria' must be a non-empty list"
    if not all(isinstance(c, str) and c.strip() for c in crit):
        return "every item of 'success_criteria' must be a non-empty string"
    for key in ("plan", "assumptions", "blocking_questions"):
        v = plan.get(key, [])
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            return f"'{key}' must be a list of strings"
    return None


@dataclass
class Report:
    run_id: str
    task: str
    # verified | unverified | failed | needs_user | budget_exhausted | no_progress | plan_failed | stopped | error
    status: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    verification: dict | None = None
    memory: dict = field(default_factory=dict)
    steps: int = 0
    duration_s: float = 0.0
    llm: dict = field(default_factory=dict)
    screenshots: list[str] = field(default_factory=list)
    verifier_model: str = ""  # the auditor's model (differs from the worker's when AUTOWORK_VERIFIER_MODEL is set)
    schema_version: int = SCHEMA_VERSION


class Agent:
    def __init__(
        self,
        llm: LLM,
        human: Human,
        workspace: Path,
        runs_dir: Path,
        playbook: Playbook | None,
        emit: Emit = lambda t, d: None,
        mode: str = "balanced",
        max_steps: int = 40,
        headless: bool = True,
        keep_full_observations: int = 2,
        max_verify_rounds: int = 2,
        vault: Vault | None = None,
        max_tokens_total: int = 400_000,
        max_active_seconds: float = 1800,
        context_scheme: str = "classic",
        world: World | None = None,
        verifier_llm: LLM | None = None,
        no_progress_steps: int = 8,
        should_stop: Callable[[], bool] | None = None,
    ):
        self.should_stop = should_stop or (lambda: False)  # the user pressed Stop (checked before every step)
        self.no_progress_steps = no_progress_steps  # 0 = never stop for lack of progress (offline replays)
        # WHY a separate auditor model when one is configured: a model auditing itself shares its own blind spots
        self.verifier_llm = verifier_llm or llm
        self.llm, self.human, self.workspace, self.playbook = llm, human, workspace, playbook
        self.mode, self.max_steps, self.headless = mode, max_steps, headless
        self.keep_full, self.max_verify_rounds = keep_full_observations, max_verify_rounds
        self.context_scheme = context_scheme
        self.world = world or default_world()
        self.vault = vault
        self.redactor = Redactor(vault.secret_values() if vault else [])
        self.max_tokens_total, self.max_active_seconds = max_tokens_total, max_active_seconds
        self._sink = emit
        self._t0 = time.monotonic()
        self._human_wait_s = 0.0
        self._seq = 0
        self.run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.run_dir = runs_dir / self.run_id
        self.memory = WorkingMemory()
        self.turns: list[Turn] = []
        self.step = 0
        self.brief = ""
        llm.on_retry = lambda m: self.emit("llm_retry", {"message": m})

    # ======================================================================= trace + budgets
    def emit(self, kind: str, data: dict) -> None:
        """Single exit point for the structured trace: redact secrets, persist to events.jsonl, forward.
        WHY one place: every event (log line, UI message, events.jsonl) passes here, so it is redacted in one spot."""
        data = self.redactor.obj(data)
        self._seq += 1
        line = {"schema": SCHEMA_VERSION, "seq": self._seq, "t": round(time.time(), 2), "type": kind, "data": data}
        try:
            with open(self.run_dir / "events.jsonl", "a") as f:
                f.write(json.dumps(line, default=str) + "\n")
        except OSError as e:
            log.warning("could not persist event: %s", e)
        self._sink(kind, data)

    def _budget_exceeded(self) -> str | None:
        """Hard limits so a confused model cannot burn money or run forever. Human wait time is
        excluded from the clock: waiting 10 minutes for an approval is not the agent misbehaving.
        A Stop from the user ends the run the same way, between steps."""
        if self.should_stop():
            return "stopped by the user"
        used = sum(m.stats.get("prompt_tokens", 0) + m.stats.get("completion_tokens", 0) for m in self._llms())
        if used > self.max_tokens_total:
            return f"token budget exhausted ({used} > {self.max_tokens_total})"
        active = time.monotonic() - self._t0 - self._human_wait_s
        if active > self.max_active_seconds:
            return f"time budget exhausted ({active:.0f}s > {self.max_active_seconds:.0f}s)"
        return None

    def _no_progress(self, progress: ProgressTracker) -> dict:
        why = ProgressTracker.why(progress.stalled)
        self.emit("warning", {"message": f"Stopping: {why}"})
        return {
            "status": "no_progress",
            "summary": f"Stopped early: {why}. Facts gathered so far:\n{self.memory.render()}",
            "evidence": [],
        }

    def _url(self) -> str:
        return self.browser.last.url if self.browser.last else ""

    def _llms(self) -> list[LLM]:
        return [self.llm] if self.verifier_llm is self.llm else [self.llm, self.verifier_llm]

    # ======================================================================= public
    def run(self, task: str) -> Report:
        t0 = time.time()
        self._t0 = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        usage = UsageLogger(self.run_dir, self.run_id, self.redactor)
        for m in self._llms():
            if hasattr(m, "set_usage_logger"):
                m.set_usage_logger(usage)
        log.info("run %s started (mode=%s, model=%s)", self.run_id, self.mode, self.llm.model)
        self.emit(
            "start",
            {
                "run_id": self.run_id,
                "task": task,
                "mode": self.mode,
                "model": self.llm.model,
                "verifier_model": self.verifier_llm.model,
                "context": self.context_scheme,
            },
        )
        self.browser = BrowserSession(
            self.run_dir / "shots",
            allowlist=self.world.allowlist(),
            risk=self.world.risk,
            base_url=self.world.start_url,
            headless=self.headless,
            gate_high_risk=self.mode != "autonomous",
        ).start()
        self.provenance = Provenance()
        self.provenance.observe(task)  # the user's own words are a source
        self.tools = ToolBox(self.browser, self.workspace, self.memory, self.vault, self.redactor, self.provenance)
        try:
            report = self._run(task)
        except QuotaExhausted as e:  # the provider stopped serving us: an infrastructure stop, not a crash
            log.warning("run %s stopped: LLM provider quota exhausted: %s", self.run_id, e)
            self.emit("warning", {"message": f"LLM provider quota exhausted, stopping: {e}"})
            report = self._report(
                task, "budget_exhausted", f"Stopped: the LLM provider's quota is exhausted ({e}).", []
            )
        except Exception as e:  # noqa: BLE001 - last line of defence: always return a report
            log.exception("run %s crashed", self.run_id)
            self.emit("error", {"message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
            report = self._report(task, "error", f"The run crashed: {type(e).__name__}: {e}", [])
        finally:
            self.browser.close()
        report.duration_s = round(time.time() - t0, 1)
        report.llm = dict(self.llm.stats)
        # Which model(s) actually answered; differs from the starting model after a provider fallback.
        report.llm["calls_by_model"] = dict(
            getattr(self.llm, "calls_by_model", None) or {self.llm.model: report.llm.get("calls", 0)}
        )
        report.verifier_model = self.verifier_llm.model
        if self.verifier_llm is not self.llm:
            v = self.verifier_llm
            report.llm["verifier"] = {
                **v.stats,
                "calls_by_model": dict(getattr(v, "calls_by_model", None) or {v.model: v.stats.get("calls", 0)}),
            }
        safe = self.redactor.obj(asdict(report))
        (self.run_dir / "report.json").write_text(json.dumps(safe, indent=2))
        self.emit("final", safe)
        log.info("run %s finished: %s in %s steps", self.run_id, report.status, report.steps)
        return report

    # ======================================================================= phases
    def _run(self, task: str) -> Report:
        try:
            plan = self._plan(task)
        except PlanFailed as e:
            self.emit("warning", {"message": f"Stopping: no valid plan ({e})"})
            return self._report(
                task,
                "plan_failed",
                f"Stopped before acting: the planner did not produce valid success criteria ({e}).",
                [],
            )
        self.brief = self._brief(task, plan, self.world.start_url)
        verify_rounds = 0
        while True:
            fin = self._execute()
            if fin is None:
                why = self._budget_exceeded() or f"step limit ({self.max_steps}) reached"
                ended = "stopped" if self.should_stop() else "budget_exhausted"
                return self._report(task, ended, f"Stopped after {self.step} steps without finishing: {why}.", [])
            status, summary, evidence = fin.get("status"), fin.get("summary", ""), fin.get("evidence", []) or []
            if status == "no_progress":  # set by the loop, never by the model (finish only allows its own three)
                return self._report(task, "no_progress", summary, evidence)
            if status != "done":
                return self._report(task, "needs_user" if status == "needs_user" else "failed", summary, evidence)
            claim = Claim(task, plan["success_criteria"], summary, evidence, record_locations=list(self.tools.records))
            # Saved so the audit can be replayed offline against the saved world (evals/reverify.py)
            (self.run_dir / "claim.json").write_text(json.dumps(self.redactor.obj(asdict(claim)), indent=2))
            verdict = self._verifier().verify(claim, self.browser)
            if verdict.get("passed") and validate_plan(plan) is None:  # never verified without valid criteria
                self._learn(task)
                return self._report(task, "verified", summary, evidence, verdict)
            verify_rounds += 1
            if verify_rounds > self.max_verify_rounds or verdict.get("inconclusive"):
                return self._report(task, "unverified", summary, evidence, verdict)
            self.emit(
                "warning",
                {"message": "Verification failed, sending the worker back to fix it", "reason": verdict.get("reason")},
            )
            self.turns.append(
                note(
                    f"INDEPENDENT AUDIT FAILED: {verdict.get('reason')}\nAuditor evidence: "
                    f"{verdict.get('evidence')}\nInvestigate, fix the problem, and call finish again."
                )
            )

    def _plan(self, task: str) -> dict:
        """Plan, validated. Invalid output gets one retry with the validation error; if that is invalid too, the
        run stops (PlanFailed). WHY strict: the success criteria are what the auditor checks, and live runs
        returned `{}` and the plan text as a dict key; both were accepted and the run went on with no criteria."""
        sys = prompts.PLANNER.replace("{playbook}", self._playbook_block()).replace(
            "{environment}", self.world.description
        )
        msgs: list[dict] = [{"role": "system", "content": sys}, {"role": "user", "content": task}]
        problem = ""
        plan: dict | None = None
        for attempt in (1, 2):
            try:
                # WHY the retry drops json_mode: NVIDIA's JSON mode on nemotron returned {"": ""} (the plan was in the
                # hidden reasoning); the same request without it returned a valid plan. parse_json reads plain text.
                r = self.llm.chat(msgs, json_mode=attempt == 1, role="planner", step=0)
            except (QuotaExhausted, ProviderConfigError):
                raise  # no point planning on: every later call would fail the same way
            except LLMError as e:
                problem = f"the planner call failed: {e}"
                self.emit("warning", {"message": f"Planning attempt {attempt} failed ({problem})"})
                continue
            try:
                plan = parse_json(r.content)
                problem = validate_plan(plan) or ""
            except ValueError as e:
                plan, problem = None, f"not a JSON object: {e}"
            if not problem:
                break
            self.emit("warning", {"message": f"Planning attempt {attempt} returned an invalid plan: {problem}"})
            msgs += [
                {"role": "assistant", "content": r.content or ""},
                {
                    "role": "user",
                    "content": f"That plan is invalid: {problem}. Reply with the JSON object again, "
                    "with every key described in your instructions.",
                },
            ]
        if problem or plan is None:
            raise PlanFailed(problem or "no plan")
        plan.setdefault("plan", [])
        plan.setdefault("assumptions", [])
        plan.setdefault("blocking_questions", [])
        self.emit("plan", plan)
        answers = []
        for q in (plan.get("blocking_questions") or [])[:2]:
            a = self._ask_human("clarification", q)
            answers.append(f"Q: {q}\nA: {a.get('answer', '')}")
        plan["clarifications"] = answers
        return plan

    @staticmethod
    def _brief(task: str, plan: dict, start_url: str) -> str:
        parts = [f"TASK FROM USER:\n{task}", f"GOAL: {plan.get('goal', task)}"]
        if plan.get("success_criteria"):
            parts.append("SUCCESS CRITERIA:\n" + "\n".join(f"- {c}" for c in plan["success_criteria"]))
        if plan.get("plan"):
            steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan["plan"]))
            parts.append("INITIAL PLAN (adapt as needed):\n" + steps)
        if plan.get("clarifications"):
            parts.append("CLARIFICATIONS FROM USER:\n" + "\n".join(plan["clarifications"]))
        parts.append(f"Start page: {start_url} . Begin.")
        return "\n\n".join(parts)

    # ======================================================================= main loop
    def _execute(self) -> dict | None:
        """Run steps until the model calls finish (returns its arguments) or a budget runs out (None)."""
        system = prompts.WORKER.replace("{playbook}", self._playbook_block()).replace(
            "{sites}", ", ".join(self.vault.sites()) if self.vault else "none configured"
        )
        repeats, errors, progress = RepetitionDetector(), ErrorStreak(), ProgressTracker(self.no_progress_steps)
        while self.step < self.max_steps:
            if why := self._budget_exceeded():
                self.emit("warning", {"message": f"Stopping: {why}"})
                return None
            self.step += 1
            status = (
                f"[status] step {self.step}/{self.max_steps}. WORKING MEMORY:\n{self.memory.render()}\n"
                "Continue: one sentence of reasoning, then one tool call."
            )
            resp = self.llm.chat(
                build_messages(system, self.brief, self.turns, status, self.keep_full, self.context_scheme),
                tools=WORKER_TOOLS,
                require_tool=True,
                role="worker",
                step=self.step,
            )
            if not resp.tool_calls:
                self.emit("thought", {"step": self.step, "text": resp.content})
                self.turns.append(note("You must respond with a tool call. If the task is complete, call finish."))
                continue
            call = resp.tool_calls[0]
            turn = Turn(
                resp.content.strip(),
                call.id,
                call.name,
                call.raw_arguments or json.dumps(call.arguments),
                extra_calls=resp.tool_calls[1:],
            )
            self.turns.append(turn)
            if turn.content:
                self.emit("thought", {"step": self.step, "text": turn.content})
            self.emit(
                "action",
                {
                    "step": self.step,
                    "tool": call.name,
                    "args": tool_args_preview(call.arguments),
                    "raw_args": call.raw_arguments or json.dumps(call.arguments),  # complete, for offline replay
                },
            )

            if call.name == "finish":
                turn.result = ToolResult("Finish received; handing over to verification.", "finish called")
                return call.arguments

            fp = self.browser.last.fingerprint() if self.browser.last else ""
            signal, seen = repeats.observe(call.name, call.arguments, fp)
            if signal is Signal.ESCALATE:
                ans = self._ask_human(
                    "clarification",
                    f"I seem to be stuck: I've tried `{call.name}` {seen - 1} times on the same page without "
                    f"progress. Last reasoning: {turn.content[:300]!r}. How should I proceed?",
                )
                turn.result = ToolResult(
                    f"Action NOT executed (loop detected). Human guidance: {ans.get('answer')}",
                    "loop detected; human consulted",
                )
                if progress.observe(self._url(), None, self.memory.as_dict()) is Signal.STOP:
                    return self._no_progress(progress)
                continue

            result = self._act(call.name, call.arguments, turn.content)
            stall = progress.observe(  # before any note is appended: notes are not information
                self._url(), self.tools.last_observation, self.memory.as_dict(), self.tools.last_changed_state
            )
            if signal is Signal.WARN:
                result.text += (
                    "\n\nWARNING: you have now performed this exact action 3 times on an unchanged "
                    "page. It is not working. Try a different approach."
                )
                self.emit("warning", {"message": f"Repeated action detected: {call.name}"})
            turn.result = result
            self.emit(
                "observation",
                {
                    "step": self.step,
                    "tool": call.name,
                    "ok": result.ok,
                    "text": result.text[:4000],
                    "screenshot": self._rel(result.screenshot),
                },
            )
            if call.name == "remember":
                self.emit("memory", self.memory.as_dict())

            streak = errors.observe(result.ok)
            if streak is Signal.WARN:
                result.text += (
                    "\n\nNOTE: 3 consecutive actions have failed. Step back: re-read the page, check "
                    "your assumptions (are you logged in? right page? right format?), or ask the human."
                )
            elif streak is Signal.ESCALATE:
                ans = self._ask_human(
                    "clarification",
                    f"I've hit {errors.count} failures in a row. Latest: {result.short}. How should I proceed?",
                )
                self.turns.append(note(f"Human guidance after repeated failures: {ans.get('answer')}"))
                errors.reset()
            if stall is Signal.WARN:
                result.text += (
                    f"\n\nNOTE: your last {progress.stalled} actions produced no new page, no new information and "
                    "no new facts. Stop varying the same approach. Re-read your earlier observations for links or "
                    "results you have not opened yet, try a different route, or call finish with status needs_user."
                )
                self.emit("warning", {"message": f"No progress in {progress.stalled} steps"})
            elif stall is Signal.STOP:
                return self._no_progress(progress)
            if self.step == int(self.max_steps * WRAP_UP_AT):
                self.turns.append(
                    note(
                        f"You have used {self.step}/{self.max_steps} steps. Wrap up: complete the essential part, "
                        "verify, and call finish (status failed/needs_user if you cannot complete)."
                    )
                )
        return None

    # ======================================================================= gated execution
    def _act(self, name: str, args: dict, reasoning: str) -> ToolResult:
        if name == "ask_human":
            a = self._ask_human("clarification", args.get("question", ""), args.get("options"))
            return ToolResult(f"Human answered: {a.get('answer', '')}", f"human answered: {a.get('answer', '')[:200]}")
        decision = policy.evaluate(name, args, self.browser.last, self.mode, self.world.risk.button_label)
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
            el = snap.element(int(args["element_id"])) if snap and str(args.get("element_id", "")).isdigit() else None
            if el:  # the UI names the approve button after it ("Approve "Mark as paid"")
                action["label"] = (el.get("text") or el.get("label") or "").strip()
            a = self._ask_human(
                "approval",
                f"{reason}\n\nAgent's reasoning: {reasoning or '(none given)'}",
                screenshot=snap.screenshot if snap else None,
                action=action,
            )
            if not a.get("approved"):
                return ToolResult(
                    f"DENIED by human: {a.get('comment') or 'no reason given'}. Do not retry this "
                    "action. Continue without it or finish with status needs_user.",
                    "denied by human",
                    ok=False,
                )
            now = self.browser.snapshot("approval-recheck") if snap else None
            if now is None or now.fingerprint() == fp:
                return None
            self.emit(
                "warning",
                {
                    "message": "Page changed while waiting for approval; asking again",
                    "before": fp,
                    "after": now.fingerprint(),
                },
            )
            reason = f"(Re-approval: the page changed) {reason}"
        return ToolResult(
            "DENIED: the page kept changing during approval; not executed.", "approval unstable", ok=False
        )

    def _approve_network_request(self, name: str, args: dict, key: str, reasoning: str) -> ToolResult:
        """The browser aborted a high-risk request (e.g. POST /erp/bills/2/pay) that no button-level approval
        covered, e.g. a form submitted with Enter or by page JavaScript. Ask, then replay the action once
        with exactly that request allowed."""
        self.emit("policy", {"verdict": "approve", "reason": f"network gate: {key}", "risk": "high"})
        snap = self.browser.last
        a = self._ask_human(
            "approval",
            f"The action {name} {tool_args_preview(args)} tries to send the high-risk "
            f"request {key}.\n\nAgent's reasoning: {reasoning or '(none given)'}",
            screenshot=snap.screenshot if snap else None,
            action={"tool": name, "args": args, "request": key},
        )
        if not a.get("approved"):
            return ToolResult(
                f"DENIED by human: the request {key} was not sent. Do not retry it.", "denied by human", ok=False
            )
        self.browser.preapprove(key)
        try:
            return self.tools.run(name, args, self.step)
        except ApprovalRequired as e:
            return ToolResult(
                f"BLOCKED: the action sent a different high-risk request ({e.key}) than the one "
                f"approved ({key}); nothing was sent.",
                "approval mismatch",
                ok=False,
            )

    # ======================================================================= verification + learning
    def _verifier(self) -> Verifier:
        return Verifier(
            self.verifier_llm,
            self.emit,
            self.workspace,
            self.run_dir / "verify",
            self.vault,
            self.redactor,
            self._budget_exceeded,
            self._rel,
            self.world,
        )

    def _learn(self, task: str) -> None:
        """After a VERIFIED run only: distill general notes for future runs (the playbook)."""
        if not self.playbook:
            return
        trace = []
        for t in self.turns:
            if t.result is None:
                trace.append(f"[system note] {t.content[:200]}")
            else:
                trace.append(f"{t.name}({t.raw_args[:200]}) -> {t.result.short[:200]}")
        try:
            r = self.llm.chat(
                [
                    {"role": "system", "content": prompts.DISTILL},
                    {"role": "user", "content": f"Task: {task}\n\nTrace:\n" + "\n".join(trace[-60:])},
                ],
                json_mode=True,
                role="distiller",
                step=self.step,
            )
            notes = parse_json(r.content).get("notes", [])[:5]
            self.playbook.add([n for n in notes if isinstance(n, str)], self.run_id)
            self.emit("learned", {"notes": notes})
        except (LLMError, ValueError) as e:
            self.emit("warning", {"message": f"Could not distill playbook notes: {e}"})

    # ======================================================================= helpers
    def _ask_human(
        self,
        kind: str,
        question: str,
        options: list | None = None,
        screenshot: str | None = None,
        action: dict | None = None,
    ) -> dict:
        self.emit(
            "human_request",
            {
                "kind": kind,
                "question": question,
                "options": options or [],
                "screenshot": self._rel(screenshot),
                "action": action,
            },
        )
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

    def _report(self, task: str, status: str, summary: str, evidence: list, verification: dict | None = None) -> Report:
        shots = sorted(str(p.relative_to(self.run_dir.parent)) for p in (self.run_dir / "shots").glob("*.png"))
        return Report(
            self.run_id,
            task,
            status,
            summary,
            list(evidence),
            verification,
            self.memory.as_dict(),
            self.step,
            screenshots=shots[-6:],
        )
