"""Offline tests: real simulated world + real browser + real agent loop, with a scripted LLM.

They check the plumbing (policy gates, fault handling, read-only verifier, context
compression, memory), not the model's intelligence. Run: .venv/bin/pytest -q
"""

from __future__ import annotations

import json

from agent import policy
from agent.browser import Snapshot
from agent.human import ScriptedHuman
from agent.memory import Playbook
from conftest import ACME_CHECKLIST, AUDIT_ACME_OK, LOGIN, PLAN, W, admin_post, admin_state, make_agent


def test_ambiguous_timeout_then_check_before_retry(tmp_path, ws):
    admin_post("/admin/reset", {"erp_submit_timeout": True})
    script = [
        PLAN,
        *LOGIN,
        (
            "browser_fill",
            {
                "fields": [
                    {"element_id": 6, "value": "Acme Supplies Inc."},
                    {"element_id": 7, "value": "INV-2041"},
                    {"element_id": 8, "value": "4250.00"},
                    {"element_id": 10, "value": "2026-10-01"},
                    {"element_id": 11, "value": "2026-10-31"},
                ]
            },
        ),
        ("browser_click", {"element_id": 13}),  # -> 504, but saved
        ("browser_goto", {"url": W + "/erp/bills?q=INV-2041"}),  # check before retrying
        ("remember", {"key": "bill", "value": "INV-2041 saved once"}),
        ("finish", {"status": "done", "summary": "Entered INV-2041", "evidence": ["bill list shows it"]}),
        *AUDIT_ACME_OK,
        {"notes": ["ERP login is at /erp/login; credentials in credentials.md"]},
    ]
    agent, events = make_agent(tmp_path, ws, script)
    r = agent.run("Enter the latest Acme invoice")
    assert r.status == "verified", r
    obs = [d for k, d in events if k == "observation"]
    assert any("HTTP STATUS: 504" in o["text"] for o in obs)
    bills = [b for b in admin_state().json()["bills"] if b["invoice_number"] == "INV-2041"]
    assert len(bills) == 1 and bills[0]["amount_cents"] == 425000
    assert Playbook(tmp_path / "pb.json").load()[0]["note"].startswith("ERP login")
    saved = json.loads((tmp_path / "runs" / r.run_id / "report.json").read_text())
    assert saved["llm"]["calls_by_model"] == {agent.llm.model: saved["llm"]["calls"]}  # credits the model that answered


def test_payment_requires_approval_and_denial_is_respected(tmp_path, ws):
    admin_post("/admin/reset", None)
    script = [
        PLAN,
        *LOGIN,
        ("browser_goto", {"url": W + "/erp/bills/2"}),
        ("browser_click", {"element_id": 6}),  # "Mark as paid"
        ("finish", {"status": "needs_user", "summary": "Payment not approved", "evidence": []}),
    ]
    human = ScriptedHuman(approve=False)
    agent, events = make_agent(tmp_path, ws, script, human=human)
    r = agent.run("Pay the Globex bill")
    assert r.status == "needs_user"
    assert human.log and human.log[0]["kind"] == "approval" and "Mark as paid" in human.log[0]["question"]
    state = admin_state().json()
    assert next(b for b in state["bills"] if b["id"] == 2)["status"] == "open"


def test_verifier_cannot_write(tmp_path, ws):
    admin_post("/admin/reset", None)
    script = [
        PLAN,
        *LOGIN,
        ("finish", {"status": "done", "summary": "done", "evidence": []}),
        ACME_CHECKLIST,
        ("browser_goto", {"url": W + "/erp/bills/2"}),
        ("browser_click", {"element_id": 6}),  # auditor tries "Mark as paid" -> must be blocked
        ("verdict", {"passed": False, "reason": "nothing entered", "evidence": []}),
        ("finish", {"status": "failed", "summary": "gave up", "evidence": []}),
    ]
    agent, events = make_agent(tmp_path, ws, script, mode="autonomous")
    r = agent.run("x")
    assert r.status == "failed"
    vsteps = [d for k, d in events if k == "verify_step"]
    assert any(not v["ok"] for v in vsteps), vsteps
    assert next(b for b in admin_state().json()["bills"] if b["id"] == 2)["status"] == "open"
    # the failed audit was fed back to the worker
    assert any("INDEPENDENT AUDIT FAILED" in str(m) for m in agent.llm.seen[-1])


def test_auditor_is_forced_to_a_verdict_and_not_pointed_at_the_workers_sources(tmp_path, ws):
    # Regression from a live eval: a correct run ended "unverified" because the auditor wandered until its
    # step budget ran out. On its last step it may now only call verdict.
    # Regression from 2026-10-04: the auditor was pointed at the pages the worker READ, and so inherited its blind
    # spots. Now it only gets where the worker's form submissions landed.
    from agent.verifier import MAX_AUDIT_STEPS

    admin_post("/admin/reset", None)
    wander = [("browser_read", {})] * (MAX_AUDIT_STEPS - 1)
    script = [
        PLAN,
        *LOGIN,
        ("finish", {"status": "done", "summary": "done", "evidence": []}),
        ACME_CHECKLIST,
        *wander,
        ("verdict", {"passed": False, "reason": "could not confirm the bill", "evidence": []}),
        ("finish", {"status": "failed", "summary": "gave up", "evidence": []}),
    ]
    agent, events = make_agent(tmp_path, ws, script, mode="autonomous")
    agent.run("x")
    result = next(d for k, d in events if k == "verify_result")
    assert not result.get("inconclusive") and result["reason"] == "could not confirm the bill"
    audit_calls = agent.llm.tools_offered[5 : 5 + MAX_AUDIT_STEPS]
    assert audit_calls[-1] == ["verdict"] and "browser_read" in audit_calls[0]
    checklist_prompt = agent.llm.seen[4]
    assert checklist_prompt[1]["content"] == "TASK:\nx"  # the auditor's checklist sees the task text only
    opening = agent.llm.seen[5][1]["content"]
    assert "RECORD LOCATIONS" in opening and "/erp/bills/new" not in opening  # the worker read it; not a record


def test_forced_verdict_cannot_pass_without_evidence(tmp_path, ws):
    from agent.verifier import MAX_AUDIT_STEPS

    admin_post("/admin/reset", None)
    script = [
        PLAN,
        *LOGIN,
        ("finish", {"status": "done", "summary": "done", "evidence": []}),
        ACME_CHECKLIST,
        *[("browser_read", {})] * (MAX_AUDIT_STEPS - 1),
        ("verdict", {"passed": True, "reason": "looks fine", "evidence": []}),  # a guess under budget pressure
        ("finish", {"status": "failed", "summary": "gave up", "evidence": []}),
    ]
    agent, events = make_agent(tmp_path, ws, script, mode="autonomous")
    r = agent.run("x")
    result = next(d for k, d in events if k == "verify_result")
    assert result["passed"] is False and "without evidence" in result["reason"]
    assert r.status != "verified"


def test_loop_detection_escalates_to_human(tmp_path, ws):
    admin_post("/admin/reset", None)
    same = ("browser_goto", {"url": W + "/mail"})
    script = [PLAN, same, same, same, same, same, ("finish", {"status": "failed", "summary": "stuck", "evidence": []})]
    human = ScriptedHuman(answers={"stuck": "stop and report"})
    agent, events = make_agent(tmp_path, ws, script, human=human)
    agent.run("x")
    assert any("stuck" in h["question"] for h in human.log)
    assert any(k == "warning" for k, _ in events)


def test_context_compression_keeps_memory(tmp_path, ws):
    admin_post("/admin/reset", None)
    script = [
        PLAN,
        ("browser_goto", {"url": W + "/mail"}),
        ("remember", {"key": "amount", "value": "4250.00"}),
        ("browser_goto", {"url": W + "/mail/1"}),
        ("browser_goto", {"url": W + "/mail/2"}),
        ("browser_goto", {"url": W + "/mail/3"}),
        ("finish", {"status": "failed", "summary": "x", "evidence": []}),
    ]
    agent, _ = make_agent(tmp_path, ws, script)
    agent.run("x")
    last = agent.llm.seen[-1]
    tool_msgs = [m["content"] for m in last if m["role"] == "tool"]
    assert "old observation elided" in tool_msgs[0]  # first inbox view compressed
    assert "amount: 4250.00" in last[-1]["content"]  # memory still present


def test_policy_rules():
    snap = Snapshot(
        "http://x/erp/bills/1",
        "t",
        200,
        [
            {"id": 1, "tag": "button", "text": "Mark as paid"},
            {"id": 2, "tag": "button", "text": "Save bill"},
            {"id": 3, "tag": "a", "text": "Delete help article", "href": "/help"},
        ],
        [],
        "",
        False,
    )
    assert policy.evaluate("browser_click", {"element_id": 1}, snap, "balanced").verdict == "approve"
    assert policy.evaluate("browser_click", {"element_id": 1}, snap, "autonomous").verdict == "allow"
    assert policy.evaluate("browser_click", {"element_id": 2}, snap, "balanced").verdict == "allow"
    assert policy.evaluate("browser_click", {"element_id": 2}, snap, "supervised").verdict == "approve"
    assert policy.evaluate("browser_click", {"element_id": 3}, snap, "balanced").verdict == "allow"  # links navigate
    assert policy.evaluate("write_file", {"path": "../x"}, None).verdict == "deny"


def test_busy_work_loop_stops_with_no_progress(tmp_path, ws):
    # Regression from vendor_contact_update (2026-10-04): ~25 steps of retyping the mail search box and reopening
    # the inbox. Every action succeeds and differs, so the repetition and error detectors never fire.
    admin_post("/admin/reset", None)
    loop = [
        ("browser_goto", {"url": W + "/mail"}),
        *[("browser_fill", {"fields": [{"element_id": 2, "value": q}]}) for q in ("vendor", "contact", "CFO")],
    ] * 4
    script = [PLAN, *loop]
    agent, events = make_agent(tmp_path, ws, script, no_progress_steps=8)
    r = agent.run("Our CFO emailed about a vendor contact change. Make sure the ERP reflects it.")
    assert r.status == "no_progress" and "no progress in the last 8 steps" in r.summary
    assert r.steps == 9  # step 1 opened the inbox (progress), then 8 steps with nothing new
    saved = json.loads((tmp_path / "runs" / r.run_id / "report.json").read_text())
    assert saved["status"] == "no_progress"
    notes = [d["message"] for k, d in events if k == "warning"]
    assert "No progress in 5 steps" in notes  # warned first


def test_no_progress_stop_can_be_disabled(tmp_path, ws):
    admin_post("/admin/reset", None)
    loop = [("browser_goto", {"url": W + "/mail"}), ("browser_fill", {"fields": [{"element_id": 2, "value": "x"}]})]
    script = [PLAN, *(loop * 6), ("finish", {"status": "failed", "summary": "gave up", "evidence": []})]
    agent, _ = make_agent(tmp_path, ws, script, no_progress_steps=0)
    assert agent.run("x").status == "failed"


def test_forced_verdict_gets_one_retry_when_the_model_returns_no_tool_call(tmp_path, ws):
    # Seen live (reverify, nemotron): the forced last-step verdict came back empty and the audit ended inconclusive.
    from agent.verifier import MAX_AUDIT_STEPS

    admin_post("/admin/reset", None)
    script = [
        PLAN,
        *LOGIN,
        ("finish", {"status": "done", "summary": "done", "evidence": []}),
        ACME_CHECKLIST,
        *[("browser_read", {})] * (MAX_AUDIT_STEPS - 1),
        {"no": "tool call"},  # the last step answered without calling verdict
        ("verdict", {"passed": False, "reason": "bill not found", "evidence": []}),
        ("finish", {"status": "failed", "summary": "gave up", "evidence": []}),
    ]
    agent, events = make_agent(tmp_path, ws, script, mode="autonomous")
    agent.run("x")
    result = next(d for k, d in events if k == "verify_result")
    assert result["reason"] == "bill not found" and not result.get("inconclusive")
    nudges = [m["content"] for m in agent.llm.seen[5 + MAX_AUDIT_STEPS - 3] if m["role"] == "user"]
    assert "2 steps left" in nudges[-1]


def test_no_progress_stop_also_counts_steps_the_loop_detector_blocked(tmp_path, ws):
    # Live 2026-10-04 (20261004-140643-d4d0): list_files with an invented argument, 8 times. Steps 4 and 8 were
    # blocked by the loop detector; the stop signal on step 8 was ignored and the run stopped one step late.
    admin_post("/admin/reset", None)
    script = [PLAN, *[("list_files", {"path": ""})] * 12]
    agent, _ = make_agent(tmp_path, ws, script, no_progress_steps=8)
    r = agent.run("x")
    assert r.status == "no_progress" and r.steps == 8
