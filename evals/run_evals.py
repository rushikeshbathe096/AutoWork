"""Run the eval suite against a live LLM.

  .venv/bin/python -m evals.run_evals                 # all tasks, fresh playbook each task
  .venv/bin/python -m evals.run_evals --only acme_invoice --repeat 3
  .venv/bin/python -m evals.run_evals --playbook      # let lessons accumulate across tasks
  .venv/bin/python -m evals.run_evals --model openai/gpt-oss-20b --repeat 2   # override LLM_MODEL
  .venv/bin/python -m evals.run_evals --report        # only regenerate results.md from the history

Starts the simulated world itself if it isn't running. Every graded run is appended to evals/history.jsonl as
soon as it finishes (tagged with model and code version), and evals/results.md is regenerated from the whole
history. So runs can be spread over days and models, e.g. to stay within free-tier daily token limits.
If the provider's quota runs out mid-suite, that run is discarded (not the agent's fault) and the suite stops.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import httpx
import uvicorn

from agent.config import PLAYBOOK_PATH, RUNS_DIR, WORKSPACE, WORLD_URL, Settings, admin_headers, reset_workspace
from agent.core import Agent
from agent.human import ScriptedHuman
from agent.llm import LLMClient
from agent.memory import Playbook
from agent.vault import Vault

from .report import categorize, is_honest, render_history
from .tasks import TASKS, Failure, Task

HERE = Path(__file__).parent
HISTORY = HERE / "history.jsonl"
# Code whose changes can change eval outcomes. Results from different fingerprints are never aggregated.
BEHAVIOUR_CODE = ["agent", "simworld", "evals/tasks.py"]


class QuotaStop(Exception):
    pass


def code_version() -> str:
    """git commit + a fingerprint of the behaviour-relevant sources, so uncommitted edits count as a new version."""
    root = HERE.parent
    h = hashlib.sha256()
    for d in BEHAVIOUR_CODE:
        for f in sorted((root / d).rglob("*.py")) if (root / d).is_dir() else [root / d]:
            h.update(f.relative_to(root).as_posix().encode() + b"\0" + f.read_bytes())
    git = shutil.which("git")
    try:
        if not git:
            raise OSError("git not found")
        sha = subprocess.run(  # noqa: S603 - fixed argv, no user input
            [git, "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "nogit"
    return f"{sha}-{h.hexdigest()[:8]}"


def load_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    return [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]


def append_history(row: dict) -> None:
    with HISTORY.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def write_report() -> str:
    md = render_history(load_history(), time.strftime("%Y-%m-%d %H:%M"))
    (HERE / "results.md").write_text(md)
    return md


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
    ap.add_argument("--model", help="override LLM_MODEL for this run")
    ap.add_argument("--report", action="store_true", help="only regenerate results.md from the history")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip tasks that already have --repeat graded runs for this model, code version and condition "
        "(rerun the same command daily to finish a suite within free-tier quotas)",
    )
    a = ap.parse_args()
    if a.report:
        print(write_report())
        return
    settings = Settings.from_env()
    if a.model:
        settings = dataclasses.replace(settings, llm_model=a.model)
    unknown = set(a.only or []) - {t.id for t in TASKS}
    if unknown:
        raise SystemExit(f"unknown task id(s): {sorted(unknown)}")
    ensure_world()
    pb_path = PLAYBOOK_PATH if a.playbook else HERE / ".eval_playbook.json"
    tasks = [t for t in TASKS if not a.only or t.id in a.only]
    if a.resume:
        version = code_version()
        graded = Counter(
            r["task"]
            for r in load_history()
            if r.get("model") == settings.llm_model
            and r.get("code") == version
            and bool(r.get("playbook")) == a.playbook
            and not r.get("discarded")
        )
        done = [t.id for t in tasks if graded[t.id] >= a.repeat]
        tasks = [t for t in tasks if graded[t.id] < a.repeat]
        print(
            f"--resume: {len(done)} task(s) already done for this version, {len(tasks)} to go: {[t.id for t in tasks]}"
        )
        a.repeat_needed = {t.id: a.repeat - graded[t.id] for t in tasks}
    try:
        run_tasks(tasks, a, settings, pb_path)
    except QuotaStop as e:
        print(f"\n*** Stopped: {e}\n*** Runs finished before this are saved; rerun later or with another --model.")
    print("\n" + write_report())


def run_tasks(tasks: list[Task], a: argparse.Namespace, settings: Settings, pb_path: Path) -> list[dict]:
    rows: list[dict] = []
    version = code_version()
    for t in tasks:
        for rep in range(getattr(a, "repeat_needed", {}).get(t.id, a.repeat)):
            if not a.playbook:
                pb_path.unlink(missing_ok=True)
            httpx.post(
                f"{WORLD_URL}/admin/reset", json=t.faults or None, headers=admin_headers(), timeout=10
            ).raise_for_status()
            reset_workspace()
            human = ScriptedHuman(approve=t.approve, answers=t.answers)

            def emit(k: str, d: dict, tid: str = t.id) -> None:
                if not a.quiet:
                    print(f"  [{tid}] {k}: {json.dumps(d, default=str)[:160]}", flush=True)

            llm = LLMClient(settings)
            agent = Agent(
                llm,
                human,
                WORKSPACE,
                RUNS_DIR,
                Playbook(pb_path),
                emit=emit,
                mode=a.mode,
                vault=Vault.load(),
                max_tokens_total=settings.max_tokens_total,
                max_active_seconds=settings.max_active_seconds,
            )
            print(f"\n=== {t.id} (rep {rep + 1}) ===", flush=True)
            report = agent.run(t.task)
            if llm.quota_exhausted:
                # Recorded (not graded) so the report can show how many runs were dropped: silently excluding
                # them would be indistinguishable from cherry-picking.
                append_history(
                    dict(
                        model=settings.llm_model,
                        code=version,
                        when=time.strftime("%Y-%m-%d %H:%M"),
                        mode=a.mode,
                        playbook=a.playbook,
                        task=t.id,
                        rep=rep + 1,
                        discarded=True,
                        reason="provider quota exhausted",
                        steps=report.steps,
                        run_id=report.run_id,
                    )
                )
                raise QuotaStop(f"{settings.llm_model} is out of quota during {t.id}; that run was discarded")
            state = httpx.get(f"{WORLD_URL}/admin/state", headers=admin_headers(), timeout=10).json()
            try:
                failures = t.check(state, report, human, WORKSPACE)
            except Exception as e:  # noqa: BLE001 - a broken check must not abort the whole suite
                failures = [Failure("missing", f"check crashed: {e}")]
            passed = not failures
            completed = t.completed(state) if t.completed else (passed and t.expect_verified)
            row = dict(
                model=settings.llm_model,
                code=version,
                when=time.strftime("%Y-%m-%d %H:%M"),
                mode=a.mode,
                playbook=a.playbook,
                task=t.id,
                rep=rep + 1,
                passed=passed,
                agent_status=report.status,
                honest=is_honest(report.status, completed),
                category=categorize([f.category for f in failures], report.status),
                steps=report.steps,
                seconds=report.duration_s,
                llm_calls=report.llm.get("calls", 0),
                tokens=report.llm.get("prompt_tokens", 0) + report.llm.get("completion_tokens", 0),
                failures=[str(f) for f in failures],
                info=t.info(state) if t.info else "",
                run_id=report.run_id,
                summary=report.summary,
                human=human.log,
            )
            rows.append(row)
            append_history(row)  # immediately: a later crash or quota stop must not lose this run
            print(
                f"--> {'PASS' if passed else 'FAIL'} agent={report.status} steps={report.steps} "
                f"{[str(f) for f in failures]}",
                flush=True,
            )
    return rows


if __name__ == "__main__":
    main()
