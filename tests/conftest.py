"""Shared fixtures: the simulated world on :8001, a scripted LLM, and agent construction helpers."""

from __future__ import annotations

import json
import shutil
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from agent.config import admin_headers
from agent.core import Agent
from agent.human import ScriptedHuman
from agent.llm import LLMResponse, ToolCall
from agent.memory import Playbook
from agent.vault import Vault

ROOT = Path(__file__).resolve().parent.parent
W = "http://localhost:8001"


@pytest.fixture(scope="session", autouse=True)
def world():
    # Fail fast if something else (e.g. a running `python run.py`) already holds :8001; otherwise the
    # tests would silently talk to that server, which has a different admin token, and fail confusingly.
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8001)) == 0:
            pytest.exit("Port 8001 is already in use (is `python run.py` running?). Stop it and re-run the tests.", 2)
    server = uvicorn.Server(uvicorn.Config("simworld.app:app", port=8001, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        try:
            httpx.get(W + "/", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.1)
    yield
    server.should_exit = True


def admin_post(path, faults=None):
    r = httpx.post(W + path, json=faults, headers=admin_headers())
    r.raise_for_status()


def admin_state():
    return httpx.get(W + "/admin/state", headers=admin_headers())


@pytest.fixture
def ws(tmp_path):
    shutil.copytree(ROOT / "workspace_seed", tmp_path / "ws")
    return tmp_path / "ws"


class FakeLLM:
    """Replays a script. Items are dicts (JSON content) or (tool_name, args) tuples."""

    model = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self.stats = {"calls": 0}
        self.on_retry = None
        self.seen: list[list[dict]] = []
        self.tools_offered: list[list[str]] = []
        self.json_modes: list[bool] = []

    def chat(self, messages, tools=None, require_tool=False, json_mode=False, **kw):
        self.seen.append(messages)
        self.json_modes.append(json_mode)
        self.tools_offered.append([t["function"]["name"] for t in tools or []])
        self.stats["calls"] += 1
        item = self.script.pop(0)
        if isinstance(item, dict):
            return LLMResponse(json.dumps(item), [])
        name, args = item
        return LLMResponse(f"next: {name}", [ToolCall(f"c{self.stats['calls']}", name, args, json.dumps(args))])


class RoleLLM(FakeLLM):
    """A FakeLLM with one script per role (planner, worker, verifier, verifier_checklist, ...), so a test scripts
    each component separately and a code change that adds or removes a call in one role cannot shift the others.
    Items left unused at the end are fine: the old code path may not make every call the new one does."""

    def __init__(self, scripts: dict[str, list]):
        super().__init__([])
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.roles: list[str] = []

    def chat(self, messages, tools=None, require_tool=False, json_mode=False, role="worker", **kw):
        self.roles.append(role)
        script = self.scripts.get(role)
        if not script:
            raise AssertionError(f"scripted LLM has no more answers for role {role!r} (call {len(self.roles)})")
        self.script = [script.pop(0)]
        return super().chat(messages, tools=tools, require_tool=require_tool, json_mode=json_mode, **kw)


PLAN = {
    "goal": "enter Acme invoice",
    "success_criteria": ["bill INV-2041 exists once"],
    "plan": [],
    "assumptions": [],
    "blocking_questions": [],
}
LOGIN = [
    ("browser_goto", {"url": W + "/erp/bills/new"}),  # redirected to the login form (?next=...)
    ("login", {"site": "erp"}),
]  # vault fills it; back on /erp/bills/new


# The auditor's task-only checklist for the Acme invoice task, and a correct audit: it opens the SOURCE (Acme's
# portal) itself, then the record, and compares every field. C7 is PLAN's criterion.
ACME_CHECKLIST = {
    "fields": ["Vendor", "Invoice number", "Amount", "Currency", "Invoice date", "Due date"],
    "conditions": [],
    "source_apps": ["mail", "acme"],
    "values_in_task": False,
}
AUDIT_ACME_OK = [
    ACME_CHECKLIST,
    ("login", {"site": "acme"}),
    ("browser_goto", {"url": W + "/acme/invoices/INV-2041"}),
    ("browser_goto", {"url": W + "/erp/bills?q=INV-2041"}),
    (
        "verdict",
        {
            "passed": True,
            "reason": "one bill INV-2041 4250.00 USD, all fields match Acme's portal",
            "evidence": ["bills list", "Acme portal invoice INV-2041"],
            "source": W + "/acme/invoices/INV-2041",
            "checks": [
                {"id": "C1", "ok": True, "record_value": "Acme Supplies Inc.", "source_value": "Acme Supplies Inc."},
                {"id": "C2", "ok": True, "record_value": "INV-2041", "source_value": "INV-2041"},
                {"id": "C3", "ok": True, "record_value": "4250.00", "source_value": "4,250.00"},
                {"id": "C4", "ok": True, "record_value": "USD", "source_value": "USD"},
                {"id": "C5", "ok": True, "record_value": "2026-10-01", "source_value": "01 Oct 2026"},
                {"id": "C6", "ok": True, "record_value": "2026-10-31", "source_value": "31 Oct 2026"},
                {"id": "C7", "ok": True},
            ],
        },
    ),
]


def make_agent(tmp_path, ws, script, human=None, **kw):
    events = []
    a = Agent(
        script if isinstance(script, FakeLLM) else FakeLLM(script),
        human or ScriptedHuman(),
        ws,
        tmp_path / "runs",
        Playbook(tmp_path / "pb.json"),
        emit=lambda k, d: events.append((k, d)),
        vault=Vault.load(),
        **kw,
    )
    return a, events
