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

    def chat(self, messages, tools=None, require_tool=False, json_mode=False, **kw):
        self.seen.append(messages)
        self.stats["calls"] += 1
        item = self.script.pop(0)
        if isinstance(item, dict):
            return LLMResponse(json.dumps(item), [])
        name, args = item
        return LLMResponse(f"next: {name}", [ToolCall(f"c{self.stats['calls']}", name, args, json.dumps(args))])


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


def make_agent(tmp_path, ws, script, human=None, **kw):
    events = []
    a = Agent(
        FakeLLM(script),
        human or ScriptedHuman(),
        ws,
        tmp_path / "runs",
        Playbook(tmp_path / "pb.json"),
        emit=lambda k, d: events.append((k, d)),
        vault=Vault.load(),
        **kw,
    )
    return a, events
