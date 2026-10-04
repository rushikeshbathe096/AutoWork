"""Re-run ONLY the auditor of a past run, against the world state that run left behind, and grade its verdict
against ground truth. A verifier fix can be measured this way for a fraction of a full run's tokens.

The world comes from runs/<id>/world.db with the claim in runs/<id>/claim.json (saved since 2026-10-04). Older runs
are rebuilt by replaying the worker's recorded actions offline (evals/replay.py; no LLM calls).

    python -m evals.reverify <run_id...> [--verifier fixed|legacy|both] [--model MODEL]

Verdict grading: the work was really complete (by the task's ground-truth check) or not, and the auditor passed it
or not: correct pass, correct reject, FALSE PASS (the failure that matters) or false reject.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as NS

import httpx

from agent.browser import BrowserSession
from agent.config import (
    ROOT,
    RUNS_DIR,
    WORKSPACE,
    WORLD_URL,
    Settings,
    admin_headers,
    restore_world_snapshot,
    save_world_snapshot,
)
from agent.llm import LLMClient
from agent.usage import UsageLogger
from agent.vault import Redactor, Vault
from agent.verifier import Claim, Verifier
from evals.legacy_verifier import LegacyClaim, LegacyVerifier
from evals.replay import NotReplayable, replay, task_of
from evals.run_evals import ensure_world, load_history
from evals.tasks import TASKS, Task

RESULTS = ROOT / "evals" / "reverify.jsonl"


def _sources_and_facts(run_dir: Path) -> tuple[list[str], dict]:
    """What the legacy auditor was shown: pages/files the worker observed, and its working memory."""
    sources: list[str] = []
    facts: dict = {}
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        d = e["data"]
        if e["type"] == "observation" and d.get("text", "").startswith("URL: "):
            url = d["text"].split("\n", 1)[0][5:].strip()
            if url not in sources:
                sources.append(url)
        elif e["type"] == "memory":
            facts = d
    return sources, facts


def prepare(
    run_id: str, work: Path, task_id: str | None = None
) -> tuple[Task, Claim, list[str], dict, Path, list[str]]:
    """World snapshot + claim for a run: saved ones when present, else rebuilt by offline replay."""
    run_dir = RUNS_DIR / run_id
    notes: list[str] = []
    if (run_dir / "world.db").exists() and (run_dir / "claim.json").exists():
        claim = Claim(**json.loads((run_dir / "claim.json").read_text()))
        task = next((t for t in TASKS if t.id == task_id), None) if task_id else task_of(run_id, claim.task)
        if task is None:
            raise NotReplayable(f"{run_id}: not an eval run")
        sources, facts = _sources_and_facts(run_dir)
        return task, claim, sources, facts, run_dir / "world.db", ["saved snapshot"]
    r = replay(run_id, out_dir=work / "replay")
    if not r.claims:
        raise NotReplayable(f"{run_id}: the worker never claimed the task done, so there is nothing to audit")
    snap = save_world_snapshot(work / "world")
    if snap is None:
        raise NotReplayable("could not snapshot the world")
    notes = ["rebuilt by offline replay", *r.recording.notes]
    return r.recording.task, r.claims[-1], r.sources, r.facts, snap, notes


def prepare_graded(run_id: str, work: Path, human_log: list[dict], task_id: str | None = None):
    """prepare(), then grade the work against the restored world: (task, claim, sources, facts, snapshot, notes,
    work_complete, ground_truth_failures)."""
    task, claim, sources, facts, snap, notes = prepare(run_id, work, task_id)
    restore_world_snapshot(snap)  # grade the state the run left, not whatever the world holds now
    complete, failures = ground_truth(task, claim, human_log)
    return task, claim, sources, facts, snap, notes, complete, failures


def ground_truth(task: Task, claim: Claim, human_log: list[dict]) -> tuple[bool, list[str]]:
    state = httpx.get(f"{WORLD_URL}/admin/state", headers=admin_headers(), timeout=10).json()
    failures = task.check(state, NS(status="verified", summary=claim.summary), NS(log=human_log), WORKSPACE)
    complete = task.completed(state) if task.completed else (not failures and task.expect_verified)
    return bool(complete), [str(f) for f in failures]


def audit(variant: str, claim: Claim, sources: list[str], facts: dict, settings: Settings, out: Path) -> dict:
    llm = LLMClient(settings)
    out.mkdir(parents=True, exist_ok=True)
    llm.set_usage_logger(UsageLogger(out, out.parent.name))  # counted by the quota ledger like any other run
    vault = Vault.load()
    events: list[tuple[str, dict]] = []
    cls = LegacyVerifier if variant == "legacy" else Verifier
    v = cls(
        llm,
        lambda k, d: events.append((k, d)),
        WORKSPACE,
        out / "shots",
        vault,
        Redactor(vault.secret_values()),
        lambda: None,
        lambda p: p,
    )
    worker = BrowserSession(out / "worker-shots").start()  # the auditor borrows a browser context; a fresh one
    try:
        if variant == "legacy":
            c = LegacyClaim(claim.task, claim.success_criteria, claim.summary, claim.evidence, facts, sources)
            verdict = v.verify(c, worker)  # type: ignore[arg-type]
        else:
            verdict = v.verify(claim, worker)
    finally:
        worker.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.jsonl").write_text("\n".join(json.dumps({"type": k, "data": d}, default=str) for k, d in events))
    tokens = llm.stats["prompt_tokens"] + llm.stats["completion_tokens"]
    return {"verdict": verdict, "tokens": tokens, "calls": llm.stats["calls"], "quota": llm.quota_exhausted}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--verifier", choices=("fixed", "legacy", "both"), default="fixed")
    ap.add_argument("--model", help="auditor model (default: the model of the original run)")
    ap.add_argument("--task", help="eval task id, for a UI run whose task text several eval tasks share")
    a = ap.parse_args()
    ensure_world()
    rows = {r.get("run_id"): r for r in load_history() if "passed" in r}
    base = Settings.from_env()
    variants = ["legacy", "fixed"] if a.verifier == "both" else [a.verifier]
    table = []
    for rid in a.run_ids:
        row = rows.get(rid, {})
        work = Path(tempfile.mkdtemp(prefix=f"reverify-{rid}-"))
        try:
            task, claim, sources, facts, snap, notes, complete, failures = prepare_graded(
                rid, work, row.get("human", []), a.task
            )
        except (NotReplayable, KeyError, FileNotFoundError) as e:
            print(f"{rid}: SKIPPED: {e}", flush=True)
            continue
        for variant in variants:
            restore_world_snapshot(snap)
            settings = base.for_model(a.model or row.get("model") or base.llm_model)
            res = audit(variant, claim, sources, facts, settings, RUNS_DIR / rid / f"reverify-{variant}")
            passed = bool(res["verdict"].get("passed"))
            outcome = (
                ("correct pass" if passed else "false reject")
                if complete
                else ("FALSE PASS" if passed else "correct reject")
            )
            entry = {
                "when": time.strftime("%Y-%m-%d %H:%M"),
                "run_id": rid,
                "task": task.id,
                "variant": variant,
                "model": settings.llm_model,
                "work_complete": complete,
                "ground_truth_failures": failures,
                "auditor_passed": passed,
                "inconclusive": bool(res["verdict"].get("inconclusive")),
                "reason": str(res["verdict"].get("reason", ""))[:500],
                "outcome": outcome,
                "tokens": res["tokens"],
                "calls": res["calls"],
                "original_run_tokens": row.get("tokens"),
                "world": notes,
            }
            with open(RESULTS, "a") as f:
                f.write(json.dumps(entry) + "\n")
            table.append(entry)
            print(
                f"{rid} {task.id} [{variant}] -> {outcome} (auditor passed={passed}); "
                f"{res['tokens']:,} tokens vs {row.get('tokens') or 0:,} for the original run | "
                f"{entry['reason'][:160]}",
                flush=True,
            )
            if res["quota"]:
                print("*** auditor model out of quota; stopping", flush=True)
                return
    if table:
        print("\n| run | task | verifier | work complete | auditor | outcome | tokens | full run tokens |")
        print("|---|---|---|---|---|---|---|---|")
        for e in table:
            print(
                f"| {e['run_id']} | {e['task']} | {e['variant']} | {e['work_complete']} | "
                f"{'pass' if e['auditor_passed'] else 'fail'} | {e['outcome']} | {e['tokens']:,} | "
                f"{e['original_run_tokens'] or 0:,} |"
            )


if __name__ == "__main__":
    main()
