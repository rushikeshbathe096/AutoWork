"""Offline replay of a recorded run: the worker's recorded tool calls go through the CURRENT agent code (real
browser, real simulated world, real tools and prompt building), but no LLM is called.

Two uses:
  * rebuild the world state a past run left behind, when the run predates saved snapshots (evals/reverify.py);
  * measure the prompts the current code would send for a real trajectory (`--usage`: writes llm_usage.jsonl,
    which `python -m evals.usage_report` reads). Token counts there are tiktoken estimates of each section.

The verifier is NOT re-run: its recorded verdicts are replayed, so the worker is sent back (or not) exactly as in
the original run. What can't be replayed is reported, not guessed: a tool argument cut off in events.jsonl (older
runs stored a 300-char preview) is rebuilt from the observation when possible, else the run is not reconstructable.

    python -m evals.replay <run_id> [--usage]
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from agent.config import RUNS_DIR, WORKSPACE, WORLD_URL, admin_headers, reset_workspace
from agent.core import Agent, validate_plan
from agent.human import ScriptedHuman
from agent.llm import LLMResponse, ToolCall
from agent.vault import Vault
from agent.verifier import Claim
from evals.tasks import TASKS, Task


class NotReplayable(Exception):
    pass


@dataclass
class Recording:
    run_id: str
    task: Task
    mode: str
    context: str
    plan: dict
    steps: list[tuple[str, ToolCall | None]]  # (thought text, tool call or None) per worker step
    verdicts: list[dict]
    claims: list[dict]  # the summary/evidence of each claim that was verified, in order
    notes: list[str] = field(default_factory=list)  # where the replay had to deviate from the recording


def _fill_from_readback(observation: str) -> list[dict]:
    return [{"element_id": int(i), "value": v} for i, v in re.findall(r"\[(\d+)\] now = '(.*)'", observation)]


def task_of(run_id: str, task_text: str) -> Task | None:
    """The eval task of a run: by the id in the eval history (tasks can share a text: acme_invoice and
    acme_invoice_with_faults differ only in faults), else by the task text when it is unique."""
    from evals.run_evals import load_history

    tid = next((r["task"] for r in load_history() if r.get("run_id") == run_id and "task" in r), None)
    if tid:
        return next(t for t in TASKS if t.id == tid)
    same = [t for t in TASKS if t.task == task_text]
    return same[0] if len(same) == 1 else None


def load(run_id: str, runs_dir: Path = RUNS_DIR) -> Recording:
    events = [json.loads(line) for line in (runs_dir / run_id / "events.jsonl").read_text().splitlines()]
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e["data"])
    start = by_type["start"][0]
    task = task_of(run_id, start["task"])
    if task is None:
        raise NotReplayable(f"{run_id}: not an eval run (or its task text is shared and it has no history row)")
    notes: list[str] = []
    plan = {k: v for k, v in (by_type.get("plan") or [{}])[-1].items() if k != "clarifications"}
    if validate_plan(plan):
        notes.append(f"recorded plan invalid ({validate_plan(plan)}); replayed with the task as its only criterion")
        plan = {"goal": start["task"], "success_criteria": [start["task"]]}

    thoughts: dict[int, str] = {}
    steps: list[tuple[str, ToolCall | None]] = []
    actions: dict[int, dict] = {}
    obs_after: dict[int, str] = {}
    for e in events:
        d = e["data"]
        if e["type"] == "thought":
            thoughts[d["step"]] = d.get("text") or ""
        elif e["type"] == "action":
            actions[d["step"]] = d
        elif e["type"] == "observation":
            obs_after.setdefault(d["step"], d.get("text", ""))
    last_step = max([*thoughts, *actions, 0])
    claims = [{"summary": d.get("summary", "")} for d in by_type.get("verify_start", [])]
    final = (by_type.get("final") or [{}])[-1]
    for step in range(1, last_step + 1):
        a = actions.get(step)
        if a is None:
            steps.append((thoughts.get(step, ""), None))
            continue
        name, raw = a["tool"], a.get("raw_args") or a.get("args") or "{}"
        try:
            args = json.loads(raw)
        except ValueError:
            if name == "browser_fill" and (fields := _fill_from_readback(obs_after.get(step, ""))):
                args = {"fields": fields}
                notes.append(f"step {step}: fill arguments rebuilt from the read-back")
            elif name == "finish":
                m = re.search(r'"status":\s*"(\w+)"', raw)
                args = {"status": m.group(1) if m else "failed", "summary": final.get("summary", ""), "evidence": []}
                notes.append(f"step {step}: finish arguments rebuilt from the status and the final report")
            else:
                raise NotReplayable(f"{run_id}: step {step} {name} arguments were cut off in events.jsonl") from None
        steps.append((thoughts.get(step, ""), ToolCall(f"r{step}", name, args, json.dumps(args))))
    if claims:
        claims[-1]["evidence"] = final.get("evidence", [])
    verdicts = by_type.get("verify_result", [])
    return Recording(
        run_id,
        task,
        start.get("mode", "balanced"),
        start.get("context", "classic"),
        plan,
        steps,
        verdicts,
        claims,
        notes,
    )


class ReplayLLM:
    """Answers the planner and worker from the recording. Logs each call's prompt sections (no token usage)."""

    def __init__(self, rec: Recording):
        self.rec, self.model = rec, f"replay:{rec.run_id}"
        self._steps = list(rec.steps)
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        self.on_retry = None
        self.usage_logger = None

    def set_usage_logger(self, logger) -> None:
        self.usage_logger = logger

    def chat(self, messages, tools=None, require_tool=False, json_mode=False, role="worker", step=0, **kw):
        self.stats["calls"] += 1
        if self.usage_logger:
            self.usage_logger.log_call(
                timestamp_wall=0.0,
                timestamp_mono=0.0,
                latency_ms=0.0,
                role=role,
                step=step,
                model=self.model,
                base_url="",
                messages=messages,
                tools=tools,
                success=True,
            )
        if role == "planner":
            return LLMResponse(json.dumps(self.rec.plan), [])
        if role == "distiller":
            return LLMResponse(json.dumps({"notes": []}), [])
        if not self._steps:
            raise NotReplayable("the recording ended but the agent asked for another step (the replay diverged)")
        thought, call = self._steps.pop(0)
        return LLMResponse(thought, [call] if call else [])


class RecordedVerifier:
    """Stands in for the auditor: returns the recorded verdicts in order and keeps the claims it was given."""

    def __init__(self, rec: Recording, claims: list[Claim]):
        self.rec, self.claims = rec, claims

    def verify(self, claim: Claim, worker_browser) -> dict:
        i = len(self.claims)
        self.claims.append(claim)
        if i < len(self.rec.claims):  # the recorded summary is complete; finish args may have been cut off
            claim.summary = self.rec.claims[i]["summary"] or claim.summary
            if "evidence" in self.rec.claims[i]:
                claim.evidence = self.rec.claims[i]["evidence"]
        return self.rec.verdicts[i] if i < len(self.rec.verdicts) else {"passed": False, "reason": "not recorded"}


@dataclass
class Replay:
    recording: Recording
    status: str
    steps: int
    claims: list[Claim]  # as the current code builds them: criteria, summary, evidence, record locations
    sources: list[str]  # pages and files the worker observed
    facts: dict
    run_dir: Path
    summary: str = ""
    human_log: list[dict] = field(default_factory=list)


def replay(
    run_id: str,
    out_dir: Path | None = None,
    runs_dir: Path = RUNS_DIR,
    no_progress_steps: int = 0,
    context: str | None = None,
) -> Replay:
    """Reset the world (with the task's faults) and the workspace, then replay. The world is left in the state
    the recorded run produced, for the caller to inspect or audit."""
    rec = load(run_id, runs_dir)
    httpx.post(f"{WORLD_URL}/admin/reset", json=rec.task.faults or None, headers=admin_headers(), timeout=10)
    reset_workspace()
    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="replay-"))
    human = ScriptedHuman(approve=rec.task.approve, answers=rec.task.answers)
    agent = Agent(
        ReplayLLM(rec),
        human,
        WORKSPACE,
        out_dir,
        None,
        mode=rec.mode,
        vault=Vault.load(),
        max_tokens_total=10**9,
        max_active_seconds=10**6,
        context_scheme=context or rec.context,  # another scheme: same actions, its prompts (token A/B)
        # 0 reproduces what happened; N asks where the current no-progress stop would have ended the run
        no_progress_steps=no_progress_steps,
    )
    claims: list[Claim] = []
    agent._verifier = lambda: RecordedVerifier(rec, claims)  # type: ignore[method-assign]
    report = agent.run(rec.task.task)
    if report.status == "error" and not (no_progress_steps and "the recording ended" in report.summary):
        raise NotReplayable(f"{run_id}: replay failed: {report.summary}")
    return Replay(
        rec,
        report.status,
        report.steps,
        claims,
        list(agent.tools.sources),
        agent.memory.as_dict(),
        agent.run_dir,
        report.summary,
        human.log,
    )


def grade(r: Replay) -> bool:
    """Would the replayed outcome pass the task's ground-truth check (world now, the replay's status and summary)?"""
    from types import SimpleNamespace as NS

    state = httpx.get(f"{WORLD_URL}/admin/state", headers=admin_headers(), timeout=10).json()
    return not r.recording.task.check(state, NS(status=r.status, summary=r.summary), NS(log=r.human_log), WORKSPACE)


def tokens_after(run_id: str, step: int, runs_dir: Path = RUNS_DIR) -> int | None:
    """Tokens the recorded run spent on calls after `step` (worker steps after it, plus the audit and anything
    later), from its llm_usage.jsonl; None when the run predates the usage log."""
    log = runs_dir / run_id / "llm_usage.jsonl"
    if not log.exists():
        return None
    total, after = 0, False
    for line in log.read_text().splitlines():
        r = json.loads(line)
        after = after or (r.get("role") == "worker" and r.get("step", 0) > step) or r.get("role") == "verifier"
        if after:
            total += (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
    return total


def main() -> None:
    from evals.run_evals import ensure_world

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--usage", action="store_true", help="print where llm_usage.jsonl was written")
    ap.add_argument(
        "--no-progress",
        type=int,
        default=0,
        metavar="N",
        help="replay with the no-progress stop at N steps: where would the current code have stopped the run?",
    )
    ap.add_argument("--context", choices=("classic", "digest"), help="build prompts with this context scheme")
    a = ap.parse_args()
    ensure_world()
    for rid in a.run_ids:
        try:
            r = replay(rid, no_progress_steps=a.no_progress, context=a.context)
        except NotReplayable as e:
            print(f"{rid}: NOT REPLAYABLE: {e}")
            continue
        if a.no_progress:
            total = len(r.recording.steps)
            if r.status == "no_progress":
                saved = tokens_after(rid, r.steps)
                print(
                    f"{rid} {r.recording.task.id}: STOPPED at step {r.steps} of {total}; would pass: {grade(r)}; "
                    "tokens saved: " + (f"{saved:,} (measured)" if saved is not None else "unknown (no usage log)")
                )
            else:
                print(f"{rid}: not stopped ({r.status} after {r.steps} of {total} steps)")
            continue
        print(f"{rid}: replayed {r.steps} steps -> {r.status}; claims: {len(r.claims)}; notes: {r.recording.notes}")
        if a.usage:
            rows = [json.loads(x) for x in (r.run_dir / "llm_usage.jsonl").read_text().splitlines()]
            worker = [sum(sec["tokens"] for sec in x["sections"].values()) for x in rows if x["role"] == "worker"]
            print(f"  worker prompts: {len(worker)} calls, ~{sum(worker):,} tokens (tiktoken estimate)")
            print(f"  prompt sections: python -m evals.usage_report {r.run_dir}")


if __name__ == "__main__":
    main()
