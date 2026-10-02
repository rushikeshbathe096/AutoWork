"""Run the eval suite against a live LLM.

  .venv/bin/python -m evals.run_evals                 # all tasks, fresh playbook each task
  .venv/bin/python -m evals.run_evals --only acme_invoice --repeat 3
  .venv/bin/python -m evals.run_evals --playbook      # let lessons accumulate across tasks

Starts the simulated world itself if it isn't running. Writes evals/results.md and results.json.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from agent.config import PLAYBOOK_PATH, RUNS_DIR, WORKSPACE, WORLD_URL, admin_headers, reset_workspace
from agent.core import Agent
from agent.human import ScriptedHuman
from agent.llm import LLMClient
from agent.memory import Playbook
from agent.vault import Vault

from .tasks import TASKS

HERE = Path(__file__).parent


def ensure_world():
    try:
        httpx.get(WORLD_URL, timeout=2)
        return
    except httpx.HTTPError:
        pass
    server = uvicorn.Server(uvicorn.Config("simworld.app:app", port=8001, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        try:
            httpx.get(WORLD_URL, timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--playbook", action="store_true", help="accumulate playbook notes across tasks")
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    ensure_world()
    pb_path = PLAYBOOK_PATH if a.playbook else HERE / ".eval_playbook.json"
    rows = []
    for t in [t for t in TASKS if not a.only or t["id"] in a.only]:
        for rep in range(a.repeat):
            if not a.playbook:
                pb_path.unlink(missing_ok=True)
            httpx.post(f"{WORLD_URL}/admin/reset", json=t["faults"] or None, headers=admin_headers(),
                       timeout=10).raise_for_status()
            reset_workspace()
            human = ScriptedHuman(approve=t.get("approve", False), answers=t.get("answers"))
            emit = (lambda k, d: None) if a.quiet else (
                lambda k, d, tid=t["id"]: print(f"  [{tid}] {k}: {json.dumps(d, default=str)[:160]}", flush=True))
            agent = Agent(LLMClient(), human, WORKSPACE, RUNS_DIR, Playbook(pb_path), emit=emit, mode=a.mode,
                          vault=Vault.load())
            print(f"\n=== {t['id']} (rep {rep + 1}) ===", flush=True)
            report = agent.run(t["task"])
            state = httpx.get(f"{WORLD_URL}/admin/state", headers=admin_headers(), timeout=10).json()
            try:
                errs = t["check"](state, report, human, WORKSPACE)
            except Exception as e:  # noqa: BLE001
                errs = [f"check crashed: {e}"]
            passed = not errs
            # "honest" = the agent's self-assessment agrees with ground truth
            honest = (report.status == "verified") == passed or t["id"] == "payment_needs_approval"
            rows.append(dict(task=t["id"], rep=rep + 1, passed=passed, agent_status=report.status, honest=honest,
                             steps=report.steps, seconds=report.duration_s, llm_calls=report.llm.get("calls"),
                             tokens=report.llm.get("prompt_tokens", 0) + report.llm.get("completion_tokens", 0),
                             errors=errs, run_id=report.run_id, summary=report.summary,
                             human=human.log))
            print(f"--> {'PASS' if passed else 'FAIL'} agent={report.status} steps={report.steps} {errs}", flush=True)

    (HERE / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    n = len(rows)
    lines = [f"# Eval results ({time.strftime('%Y-%m-%d %H:%M')}, model `{LLMClient().model}`)\n",
             f"**Ground-truth pass rate: {sum(r['passed'] for r in rows)}/{n}** · "
             f"self-assessment matches ground truth: {sum(r['honest'] for r in rows)}/{n}\n",
             "| task | result | agent status | steps | time (s) | LLM calls | tokens | notes |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {'✅' if r['passed'] else '❌'} | {r['agent_status']} | {r['steps']} | "
                     f"{r['seconds']} | {r['llm_calls']} | {r['tokens']} | {'; '.join(r['errors'])[:120]} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
