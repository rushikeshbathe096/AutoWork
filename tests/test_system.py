"""Offline tests: real simulated world + real browser + real agent loop, with a scripted LLM.

They check the plumbing (policy gates, fault handling, read-only verifier, context
compression, memory), not the model's intelligence. Run: .venv/bin/pytest -q
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from agent import policy
from agent.browser import Snapshot
from agent.core import Agent
from agent.human import ScriptedHuman
from agent.llm import LLMResponse, ToolCall
from agent.memory import Playbook

ROOT = Path(__file__).resolve().parent.parent
W = "http://localhost:8001"


@pytest.fixture(scope="session", autouse=True)
def world():
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


PLAN = {"goal": "enter Acme invoice", "success_criteria": ["bill INV-2041 exists once"], "plan": [],
        "assumptions": [], "blocking_questions": []}
LOGIN = [("read_file", {"path": "credentials.md"}),
         ("browser_goto", {"url": W + "/erp/bills/new"}),
         ("browser_fill", {"fields": [{"element_id": 1, "value": "ap.clerk"}, {"element_id": 2, "value": "ledger-42"}]}),
         ("browser_click", {"element_id": 3})]


def make_agent(tmp_path, ws, script, human=None, **kw):
    events = []
    a = Agent(FakeLLM(script), human or ScriptedHuman(), ws, tmp_path / "runs", Playbook(tmp_path / "pb.json"),
              emit=lambda k, d: events.append((k, d)), **kw)
    return a, events


def test_ambiguous_timeout_then_check_before_retry(tmp_path, ws):
    httpx.post(W + "/admin/reset", json={"erp_submit_timeout": True})
    script = [PLAN, *LOGIN,
              ("browser_fill", {"fields": [{"element_id": 6, "value": "Acme Supplies Inc."},
                                           {"element_id": 7, "value": "INV-2041"},
                                           {"element_id": 8, "value": "4250.00"},
                                           {"element_id": 10, "value": "2026-10-01"},
                                           {"element_id": 11, "value": "2026-10-31"}]}),
              ("browser_click", {"element_id": 13}),                   # -> 504, but saved
              ("browser_goto", {"url": W + "/erp/bills?q=INV-2041"}),  # check before retrying
              ("remember", {"key": "bill", "value": "INV-2041 saved once"}),
              ("finish", {"status": "done", "summary": "Entered INV-2041", "evidence": ["bill list shows it"]}),
              # verifier: tries a write (must be blocked by network layer), then reads, then verdict
              ("browser_goto", {"url": W + "/erp/bills?q=INV-2041"}),
              ("verdict", {"passed": True, "reason": "one bill INV-2041 4250.00 USD", "evidence": ["bills list"]}),
              {"notes": ["ERP login is at /erp/login; credentials in credentials.md"]}]
    agent, events = make_agent(tmp_path, ws, script)
    r = agent.run("Enter the latest Acme invoice")
    assert r.status == "verified", r
    obs = [d for k, d in events if k == "observation"]
    assert any("HTTP STATUS: 504" in o["text"] for o in obs)
    bills = [b for b in httpx.get(W + "/admin/state").json()["bills"] if b["invoice_number"] == "INV-2041"]
    assert len(bills) == 1 and bills[0]["amount_cents"] == 425000
    assert Playbook(tmp_path / "pb.json").load()[0]["note"].startswith("ERP login")
    assert (tmp_path / "runs" / r.run_id / "report.json").exists()


def test_payment_requires_approval_and_denial_is_respected(tmp_path, ws):
    httpx.post(W + "/admin/reset")
    script = [PLAN, *LOGIN,
              ("browser_goto", {"url": W + "/erp/bills/2"}),
              ("browser_click", {"element_id": 6}),  # "Mark as paid"
              ("finish", {"status": "needs_user", "summary": "Payment not approved", "evidence": []})]
    human = ScriptedHuman(approve=False)
    agent, events = make_agent(tmp_path, ws, script, human=human)
    r = agent.run("Pay the Globex bill")
    assert r.status == "needs_user"
    assert human.log and human.log[0]["kind"] == "approval" and "Mark as paid" in human.log[0]["question"]
    state = httpx.get(W + "/admin/state").json()
    assert next(b for b in state["bills"] if b["id"] == 2)["status"] == "open"


def test_verifier_cannot_write(tmp_path, ws):
    httpx.post(W + "/admin/reset")
    script = [PLAN, *LOGIN,
              ("finish", {"status": "done", "summary": "done", "evidence": []}),
              ("browser_goto", {"url": W + "/erp/bills/2"}),
              ("browser_click", {"element_id": 6}),  # auditor tries "Mark as paid" -> must be blocked
              ("verdict", {"passed": False, "reason": "nothing entered", "evidence": []}),
              ("finish", {"status": "failed", "summary": "gave up", "evidence": []})]
    agent, events = make_agent(tmp_path, ws, script, mode="autonomous")
    r = agent.run("x")
    assert r.status == "failed"
    vsteps = [d for k, d in events if k == "verify_step"]
    assert any(not v["ok"] for v in vsteps), vsteps
    assert next(b for b in httpx.get(W + "/admin/state").json()["bills"] if b["id"] == 2)["status"] == "open"
    # the failed audit was fed back to the worker
    assert any("INDEPENDENT AUDIT FAILED" in str(m) for m in agent.llm.seen[-1])


def test_loop_detection_escalates_to_human(tmp_path, ws):
    httpx.post(W + "/admin/reset")
    same = ("browser_goto", {"url": W + "/mail"})
    script = [PLAN, same, same, same, same, same, ("finish", {"status": "failed", "summary": "stuck", "evidence": []})]
    human = ScriptedHuman(answers={"stuck": "stop and report"})
    agent, events = make_agent(tmp_path, ws, script, human=human)
    agent.run("x")
    assert any("stuck" in h["question"] for h in human.log)
    assert any(k == "warning" for k, _ in events)


def test_context_compression_keeps_memory(tmp_path, ws):
    httpx.post(W + "/admin/reset")
    script = [PLAN, ("browser_goto", {"url": W + "/mail"}), ("remember", {"key": "amount", "value": "4250.00"}),
              ("browser_goto", {"url": W + "/mail/1"}), ("browser_goto", {"url": W + "/mail/2"}),
              ("browser_goto", {"url": W + "/mail/3"}),
              ("finish", {"status": "failed", "summary": "x", "evidence": []})]
    agent, _ = make_agent(tmp_path, ws, script)
    agent.run("x")
    last = agent.llm.seen[-1]
    tool_msgs = [m["content"] for m in last if m["role"] == "tool"]
    assert "old observation elided" in tool_msgs[0]           # first inbox view compressed
    assert "amount: 4250.00" in last[-1]["content"]           # memory still present


def test_policy_rules():
    snap = Snapshot("http://x/erp/bills/1", "t", 200, [
        {"id": 1, "tag": "button", "text": "Mark as paid"}, {"id": 2, "tag": "button", "text": "Save bill"},
        {"id": 3, "tag": "a", "text": "Delete help article", "href": "/help"}], [], "", False)
    assert policy.evaluate("browser_click", {"element_id": 1}, snap, "balanced").verdict == "approve"
    assert policy.evaluate("browser_click", {"element_id": 1}, snap, "autonomous").verdict == "allow"
    assert policy.evaluate("browser_click", {"element_id": 2}, snap, "balanced").verdict == "allow"
    assert policy.evaluate("browser_click", {"element_id": 2}, snap, "supervised").verdict == "approve"
    assert policy.evaluate("browser_click", {"element_id": 3}, snap, "balanced").verdict == "allow"  # links navigate
    assert policy.evaluate("write_file", {"path": "../x"}, None).verdict == "deny"
