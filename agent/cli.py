"""Terminal entry point:  python -m agent.cli "Find the latest Acme invoice and enter it into the ERP" """
from __future__ import annotations

import argparse
import json
import sys

from .config import PLAYBOOK_PATH, RUNS_DIR, WORKSPACE, reset_workspace
from .core import Agent
from .human import CLIHuman
from .llm import LLMClient
from .memory import Playbook
from .vault import Vault

COLORS = {"plan": "36", "thought": "37", "action": "33", "observation": "90", "warning": "31", "error": "31",
          "human_request": "35", "verify_result": "32", "policy": "35", "llm_retry": "31", "learned": "34"}


def printer(kind: str, data: dict):
    c = COLORS.get(kind, "0")
    if kind == "observation":
        first = data["text"].splitlines()[:3]
        msg = ("OK  " if data["ok"] else "FAIL ") + " / ".join(first)[:220]
    elif kind == "action":
        msg = f"#{data['step']} {data['tool']} {data['args']}"
    elif kind == "thought":
        msg = data["text"][:300]
    elif kind == "final":
        print(f"\n\033[1m=== {data['status'].upper()} ===\033[0m\n{data['summary']}")
        for e in data["evidence"]:
            print(f"  • {e}")
        if data.get("verification"):
            print(f"Audit: {data['verification'].get('reason')}")
        print(f"Steps: {data['steps']}  Time: {data['duration_s']}s  LLM: {data['llm']}")
        return
    elif kind in ("start", "human_response", "memory", "verify_step", "verify_start"):
        msg = json.dumps(data)[:240]
    else:
        msg = json.dumps(data, ensure_ascii=False)[:600]
    print(f"\033[{c}m[{kind}] {msg}\033[0m", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="AutoWork autonomous task worker")
    ap.add_argument("task")
    ap.add_argument("--mode", default="balanced", choices=["autonomous", "balanced", "supervised"])
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--no-playbook", action="store_true")
    ap.add_argument("--reset-workspace", action="store_true")
    a = ap.parse_args(argv)
    if a.reset_workspace or not WORKSPACE.exists():
        reset_workspace()
    agent = Agent(LLMClient(), CLIHuman(), WORKSPACE, RUNS_DIR, None if a.no_playbook else Playbook(PLAYBOOK_PATH),
                  emit=printer, mode=a.mode, max_steps=a.max_steps, headless=not a.headed, vault=Vault.load())
    r = agent.run(a.task)
    sys.exit(0 if r.status == "verified" else 1)


if __name__ == "__main__":
    main()
