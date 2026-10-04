"""The control UI: what the page may contain, and the backend it relies on (Stop, the world's apps, approvals)."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.human import ScriptedHuman, WebHuman
from conftest import LOGIN, PLAN, FakeLLM, W, admin_post, make_agent

ROOT = Path(__file__).resolve().parent.parent
PAGE = (ROOT / "server" / "static" / "index.html").read_text()
SCRIPT = PAGE[PAGE.index("<script>") :]


def test_page_never_builds_html_from_strings():
    """Page text, emails, model output and tool arguments are untrusted: only createElement/textContent."""
    code = re.sub(r"//[^\n]*", "", SCRIPT)  # the security comment names innerHTML
    for sink in (".innerHTML", ".outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in code, sink


def test_page_makes_no_external_requests():
    urls = set(re.findall(r"https?://[^\s'\"<>)]+", PAGE)) - {"http://www.w3.org/2000/svg"}
    assert urls == set()
    assert "<link" not in PAGE and not re.search(r"<script[^>]+src=", PAGE)


@pytest.mark.parametrize(
    "word",
    [
        "Substrate",
        "Synchronized",
        "sealed",
        "SYS_RDY",
        "Stripe",
        "QuickBooks",
        "SOC2",
        "enclave",
        "cryptographic",
        "Re-plan",
        "Restart pipeline",
        "Operations",
        "Agents",
        "confidence",
    ],
)
def test_page_has_no_invented_chrome_or_overclaims(word):
    assert word.lower() not in PAGE.lower()


def test_design_reference_is_not_served():
    import server.app as sa

    assert not (ROOT / "server" / "static" / "design.html").exists()
    assert (ROOT / "docs" / "design" / "design.html").exists()
    c = TestClient(sa.app, base_url="http://localhost:8000")
    for path in ("/design.html", "/static/design.html", "/files/../server/static/index.html"):
        assert c.get(path).status_code == 404, path
    page = c.get("/").text
    assert sa.CONTROL_TOKEN in page and "__AUTOWORK_TOKEN__" not in page


def test_world_api_lists_the_configured_apps():
    import server.app as sa

    apps = TestClient(sa.app, base_url="http://localhost:8000").get("/api/world").json()["apps"]
    assert {a["name"] for a in apps} >= {"mail", "acme", "erp", "helpdesk"}
    assert all(a["path"].startswith("/") for a in apps)


def test_stop_endpoint_needs_the_token_and_a_running_run():
    import server.app as sa

    c = TestClient(sa.app, base_url="http://localhost:8000")
    assert c.post("/api/runs/x/stop").status_code == 403
    h = {"X-AutoWork-Token": sa.CONTROL_TOKEN, "Origin": "http://localhost:8000"}
    assert c.post("/api/runs/20261004-000000-abcd/stop", headers=h).status_code == 404


def test_stop_ends_the_run_between_steps(tmp_path, ws):
    admin_post("/admin/reset")
    stop = threading.Event()
    script = [PLAN, *[("browser_goto", {"url": W + f"/mail/{i}"}) for i in range(1, 6)]]
    llm = FakeLLM(script)
    orig = llm.chat

    def chat(*a, **k):
        if llm.stats["calls"] == 2:  # planner and step 1 done: Stop is pressed while step 2 is being decided
            stop.set()
        return orig(*a, **k)

    llm.chat = chat  # type: ignore[method-assign]
    agent, _ = make_agent(tmp_path, ws, llm, should_stop=stop.is_set)
    r = agent.run("x")
    assert r.status == "stopped" and "stopped by the user" in r.summary and r.steps == 2


def test_stop_unblocks_a_question_the_run_is_waiting_on():
    events = []
    h = WebHuman(lambda k, d: events.append(d), timeout_s=30)
    out = {}
    t = threading.Thread(target=lambda: out.update(h.ask("approval", "Pay?")))
    t.start()
    for _ in range(100):
        if h.pending():
            break
        threading.Event().wait(0.01)
    h.cancel_all("Stopped by the user.")
    t.join(5)
    assert out == {"approved": False, "comment": "Stopped by the user.", "answer": "Stopped by the user."}


def test_approval_event_names_the_button(tmp_path, ws):
    """The approval card says what it approves ("Approve "Mark as paid""), from the page, not parsed from text."""
    admin_post("/admin/reset", None)
    script = [
        PLAN,
        *LOGIN,
        ("browser_goto", {"url": W + "/erp/bills/2"}),
        ("browser_click", {"element_id": 6}),
        ("finish", {"status": "needs_user", "summary": "Payment not approved", "evidence": []}),
    ]
    agent, events = make_agent(tmp_path, ws, script, human=ScriptedHuman(approve=False))
    agent.run("Pay the Globex bill")
    req = next(d for k, d in events if k == "human_request" and d["kind"] == "approval")
    assert req["action"]["label"] == "Mark as paid" and len(req["action"]["page_fingerprint"]) == 12
