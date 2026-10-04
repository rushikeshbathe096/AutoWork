"""Regression tests for the two false "verified" claims seen in live evals on 2026-10-04 (nemotron-3-super).

Both runs ended `verified` while the ERP disagreed. The scripted LLMs replay what the real worker and auditor did:
  (a) acme_invoice: the worker never opened the email or the Acme portal; it found the OLD, already-paid invoice
      INV-1987 in the ERP and claimed "the latest invoice is already entered". The auditor only looked at the ERP.
  (b) initech_ambiguous: the worker typed the invoice's ISSUE date as its due date. The auditor opened the source
      email (which says "Due: 2026-10-30") and still passed, because the planner's criteria never named the due date.
Each component is scripted per role, so the same test runs against the code before and after the fix.
"""

from __future__ import annotations

import json

import pytest

from agent.core import validate_plan
from agent.verifier import Checklist, enforce, validate_checklist
from agent.world import default_world
from conftest import LOGIN, RoleLLM, W, admin_post, make_agent

ACME_TASK = (
    "Find the latest invoice from Acme, extract the amount and due date, enter it into our internal system, "
    "and tell me once it is done."
)
INITECH_TASK = "Enter the Initech invoice into the ERP."


def _plan(goal, criteria):
    return {"goal": goal, "success_criteria": criteria, "plan": [], "assumptions": [], "blocking_questions": []}


def _checklist(fields, source_apps):
    """What the auditor derives from the task text alone (no worker output, no planner output)."""
    return {"fields": fields, "conditions": [], "source_apps": source_apps, "values_in_task": False}


def test_a_old_paid_invoice_claimed_as_latest_is_not_verified(tmp_path, ws):
    admin_post("/admin/reset")
    llm = RoleLLM(
        {
            # live run: the planner returned {}; the retry (new code) gets the same thin criteria a weak model writes
            "planner": [{}, _plan("Enter the latest Acme invoice", ["The latest Acme invoice is entered in the ERP"])],
            "worker": [
                *LOGIN,
                ("browser_goto", {"url": W + "/erp/bills?q=Acme"}),
                ("browser_goto", {"url": W + "/erp/bills/1"}),
                (
                    "finish",
                    {
                        "status": "done",
                        "summary": "Found the latest invoice from Acme (ID #1) in the ERP system with amount 3980.00 "
                        "USD and due date 2026-10-01. The invoice is already entered and marked as paid.",
                        "evidence": ["Bill #1: Acme Supplies Inc., INV-1987, 3980.00 USD, due 2026-10-01, paid"],
                    },
                ),
            ],
            "distiller": [{"notes": []}],  # only reached after a (wrongly) verified run
            "verifier_checklist": [
                _checklist(
                    ["Invoice number", "Amount", "Currency", "Invoice date", "Due date", "Vendor"], ["mail", "acme"]
                )
            ],
            # live run: the auditor looked at the ERP only, then passed
            "verifier": [
                ("browser_goto", {"url": W + "/erp/bills?q=Acme"}),
                ("browser_goto", {"url": W + "/erp/bills/1"}),
                (
                    "verdict",
                    {
                        "passed": True,
                        "reason": "The latest invoice from Acme (Bill #1) was found in the ERP with amount 3980.00 USD "
                        "and due date 2026-10-01. This matches the worker's claim.",
                        "evidence": ["ERP bill detail page confirms amount 3980.00 USD, due date 2026-10-01"],
                        "source": W + "/erp/bills/1",
                        # the comparisons it would report: the record against itself
                        "checks": [
                            {"id": "C1", "ok": True, "record_value": "INV-1987", "source_value": "INV-1987"},
                            {"id": "C2", "ok": True, "record_value": "3980.00", "source_value": "3980.00"},
                            {"id": "C3", "ok": True, "record_value": "USD", "source_value": "USD"},
                            {"id": "C4", "ok": True, "record_value": "2026-09-01", "source_value": "2026-09-01"},
                            {"id": "C5", "ok": True, "record_value": "2026-10-01", "source_value": "2026-10-01"},
                            {"id": "C6", "ok": True, "record_value": "Acme Supplies Inc.", "source_value": "Acme"},
                            {"id": "C7", "ok": True},
                        ],
                    },
                ),
            ],
        }
    )
    # asked to complete its evidence (the repair rounds), the auditor repeats its verdict: still not verified
    llm.scripts["verifier"] += [llm.scripts["verifier"][-1]] * 2
    agent, events = make_agent(tmp_path, ws, llm, max_verify_rounds=0)
    r = agent.run(ACME_TASK)
    verdict = next(d for k, d in events if k == "verify_result")
    print("OUTCOME", r.status, "|", verdict["reason"][:200])
    assert r.status not in ("verified", "error"), (r.status, r.summary)
    assert "never opened the source" in verdict["reason"]
    assert any("invalid plan" in d.get("message", "") for k, d in events if k == "warning")  # {} was retried


def test_b_issue_date_entered_as_due_date_is_not_verified(tmp_path, ws):
    admin_post("/admin/reset")
    llm = RoleLLM(
        {
            # live run: the planner's criteria named the invoice date but not the due date
            "planner": [
                _plan(
                    "Enter the Initech invoice into the ERP",
                    [
                        "The ERP contains a new invoice record with vendor name 'Initech'",
                        "The invoice number in the ERP matches the invoice number on the source document",
                        "The invoice date in the ERP matches the date on the source document",
                        "The total amount in the ERP matches the total amount on the source document",
                    ],
                )
            ],
            "worker": [
                ("browser_goto", {"url": W + "/mail/5"}),  # "Invoice IN-7002 / Issued: 2026-09-30 / Due: 2026-10-30"
                *LOGIN,
                (
                    "browser_fill",
                    {
                        "fields": [
                            {"element_id": 6, "value": "Initech LLC"},
                            {"element_id": 7, "value": "IN-7002"},
                            {"element_id": 8, "value": "640.00"},
                            {"element_id": 10, "value": "2026-09-30"},
                            {"element_id": 11, "value": "2026-09-30"},  # the issue date, typed as the due date
                        ]
                    },
                ),
                ("browser_click", {"element_id": 13}),
                (
                    "finish",
                    {
                        "status": "done",
                        "summary": "Entered Initech invoice IN-7002 (640.00 USD, invoice date 2026-09-30).",
                        "evidence": ["Bill saved: Initech LLC, IN-7002, 640.00 USD"],
                    },
                ),
            ],
            "distiller": [{"notes": []}],  # only reached after a (wrongly) verified run
            "verifier_checklist": [
                _checklist(
                    ["Vendor", "Invoice number", "Amount", "Currency", "Invoice date", "Due date"], ["mail", "acme"]
                )
            ],
            # live run: the auditor opened the record and the source email, compared four fields, and passed
            "verifier": [
                ("browser_goto", {"url": W + "/erp/bills?q=IN-7002"}),
                ("browser_goto", {"url": W + "/mail/5"}),
                (
                    "verdict",
                    {
                        "passed": True,
                        "reason": "Vendor, invoice number, invoice date and amount match the source email.",
                        "evidence": ["Mail 5: Invoice IN-7002, Issued: 2026-09-30, Amount: USD 640.00"],
                        "source": W + "/mail/5",
                        "checks": [
                            {"id": "C1", "ok": True, "record_value": "Initech LLC", "source_value": "Initech"},
                            {"id": "C2", "ok": True, "record_value": "IN-7002", "source_value": "IN-7002"},
                            {"id": "C3", "ok": True, "record_value": "640.00", "source_value": "USD 640.00"},
                            {"id": "C5", "ok": True, "record_value": "2026-09-30", "source_value": "2026-09-30"},
                            *({"id": f"C{i}", "ok": True} for i in range(7, 11)),  # the planner's four criteria
                        ],
                    },
                ),
            ],
        }
    )
    # asked to complete its evidence (the repair rounds), the auditor repeats its verdict: still not verified
    llm.scripts["verifier"] += [llm.scripts["verifier"][-1]] * 2
    agent, events = make_agent(tmp_path, ws, llm, max_verify_rounds=0)
    r = agent.run(INITECH_TASK)
    verdict = next(d for k, d in events if k == "verify_result")
    print("OUTCOME", r.status, "|", verdict["reason"][:200])
    assert r.status not in ("verified", "error"), (r.status, r.summary)
    assert "C6 (Due date)" in verdict["reason"]


def test_b_variant_auditor_compares_the_due_date_but_reads_the_issue_date(tmp_path, ws):
    """Fix 3: even an auditor that does report the due date, reading the wrong line of the email, cannot pass it:
    the value it cites is labelled "Issued" on the source, and another date there is labelled "Due". The worker
    is told the same when it types the value."""
    admin_post("/admin/reset")
    worker = [
        ("browser_goto", {"url": W + "/mail/5"}),
        *LOGIN,
        (
            "browser_fill",
            {
                "fields": [
                    {"element_id": 6, "value": "Initech LLC"},
                    {"element_id": 7, "value": "IN-7002"},
                    {"element_id": 8, "value": "640.00"},
                    {"element_id": 10, "value": "2026-09-30"},
                    {"element_id": 11, "value": "2026-09-30"},
                ]
            },
        ),
        ("browser_click", {"element_id": 13}),
        ("finish", {"status": "done", "summary": "Entered IN-7002", "evidence": ["saved"]}),
    ]
    checks = [
        {"id": "C1", "ok": True, "record_value": "Initech LLC", "source_value": "Initech"},
        {"id": "C2", "ok": True, "record_value": "IN-7002", "source_value": "IN-7002"},
        {"id": "C3", "ok": True, "record_value": "640.00", "source_value": "USD 640.00"},
        {"id": "C4", "ok": True, "record_value": "USD", "source_value": "USD"},
        {"id": "C5", "ok": True, "record_value": "2026-09-30", "source_value": "2026-09-30"},
        {"id": "C6", "ok": True, "record_value": "2026-09-30", "source_value": "2026-09-30"},  # misread
        {"id": "C7", "ok": True},
    ]
    llm = RoleLLM(
        {
            "planner": [_plan("Enter the Initech invoice", ["A bill for the Initech invoice exists in the ERP"])],
            "worker": worker,
            "distiller": [{"notes": []}],
            "verifier_checklist": [
                _checklist(["Vendor", "Invoice number", "Amount", "Currency", "Invoice date", "Due date"], ["mail"])
            ],
            "verifier": [
                ("browser_goto", {"url": W + "/erp/bills?q=IN-7002"}),
                ("browser_goto", {"url": W + "/mail/5"}),
                (
                    "verdict",
                    {
                        "passed": True,
                        "reason": "all match",
                        "evidence": ["mail 5"],
                        "source": W + "/mail/5",
                        "checks": checks,
                    },
                ),
            ],
        }
    )
    agent, events = make_agent(tmp_path, ws, llm, max_verify_rounds=0)
    r = agent.run(INITECH_TASK)
    verdict = next(d for k, d in events if k == "verify_result")
    assert r.status == "unverified" and "C6 (Due date): wrong value" in verdict["reason"], verdict["reason"]
    assert not verdict.get("inconclusive")  # a wrong value is a fail: the worker would be sent back to fix it
    fill = next(d for k, d in events if k == "observation" and d["tool"] == "browser_fill")
    assert "LABEL CONFLICT" in fill["text"] and not fill["ok"]


# ----------------------------------------------------------------- fix 1: the plan is validated, never assumed


VALID = _plan("g", ["c"])
PLAN_TEXT_AS_KEY = {  # seen live: the model put its whole plan in a key (JSON mode on NVIDIA also gave {"": ""})
    ": 1. Login to the intranet portal. 2. Open webmail and locate the CFO's email": ["The ERP vendor record's email"]
}


@pytest.mark.parametrize(
    "plan, problem",
    [
        ({}, "'goal'"),
        (PLAN_TEXT_AS_KEY, "'goal'"),
        ({"goal": "g", "success_criteria": "the bill exists"}, "'success_criteria' must be a non-empty list"),
        ({"goal": "g", "success_criteria": []}, "'success_criteria' must be a non-empty list"),
        ({"goal": "g", "success_criteria": ["ok", ""]}, "non-empty string"),
        ({"goal": " ", "success_criteria": ["c"]}, "'goal'"),
        ({"goal": "g", "success_criteria": ["c"], "plan": "do it"}, "'plan' must be a list"),
        ([VALID], "JSON object"),
    ],
)
def test_validate_plan_rejects(plan, problem):
    assert problem in (validate_plan(plan) or "")


def test_validate_plan_accepts_minimal_and_full():
    assert validate_plan({"goal": "g", "success_criteria": ["c"]}) is None and validate_plan(VALID) is None


def test_invalid_plan_is_retried_once_with_the_error_then_the_run_stops(tmp_path, ws):
    llm = RoleLLM({"planner": [{}, PLAN_TEXT_AS_KEY], "worker": []})
    agent, events = make_agent(tmp_path, ws, llm)
    r = agent.run("Enter the Initech invoice into the ERP.")
    assert r.status == "plan_failed" and "success criteria" in r.summary
    assert llm.roles == ["planner", "planner"]  # stopped before acting: the worker was never called
    assert llm.json_modes == [True, False]  # live: NVIDIA's JSON mode returned {"": ""}; plain text worked
    retry = llm.seen[1]
    assert retry[-1]["role"] == "user" and "'goal' must be a non-empty string" in retry[-1]["content"]
    saved = json.loads((tmp_path / "runs" / r.run_id / "report.json").read_text())
    assert saved["status"] == "plan_failed"


def test_plan_fixed_on_retry_proceeds(tmp_path, ws):
    llm = RoleLLM(
        {
            "planner": [{"goal": "", "success_criteria": []}, VALID],
            "worker": [("finish", {"status": "failed", "summary": "s", "evidence": []})],
        }
    )
    agent, events = make_agent(tmp_path, ws, llm)
    assert agent.run("x").status == "failed"
    assert next(d for k, d in events if k == "plan")["success_criteria"] == ["c"]


# ----------------------------------------------------------------- fix 2: what a pass must show
W_ = default_world()
SRC = W + "/mail/5"
REC = W + "/erp/bills?created=4"
SEEN = {SRC: ["Invoice IN-7002\nIssued: 2026-09-30\nAmount: USD 640.00\nDue: 2026-10-30"], REC: ["Bill #4 saved."]}
CL = Checklist(["Due date"], [], ["mail"], False)
ITEMS = [("C1", "Due date", "field"), ("C2", "bill exists", "cond")]


def _v(**kw):
    base = {
        "passed": True,
        "reason": "r",
        "evidence": ["e"],
        "source": SRC,
        "checks": [
            {"id": "C1", "ok": True, "record_value": "2026-10-30", "source_value": "30 Oct 2026"},
            {"id": "c2", "ok": True},
        ],
    }
    return {**base, **kw}


def test_enforce_accepts_a_complete_audit():
    assert enforce(_v(), ITEMS, CL, SEEN, [REC], W_)["passed"] is True


@pytest.mark.parametrize(
    "verdict, checklist, why, inconclusive",
    [
        (_v(source=""), Checklist(["Due date"], [], ["acme"], False), "no source document opened", True),
        (_v(source=W + "/acme/invoices/X"), CL, "no source document opened", True),  # never opened
        (_v(source=REC), Checklist(["Due date"], [], [], True), "is the record", True),
        (_v(source="task"), CL, "'task' is no source", True),
        (
            _v(checks=[{"id": "C1", "ok": True, "record_value": "2026-10-30", "source_value": "2026-10-30"}]),
            CL,
            "not checked: C2",
            True,
        ),
        (_v(checks=[{"id": "C1", "ok": True}, {"id": "C2", "ok": True}]), CL, "no record_value/source_value", True),
        (
            _v(
                checks=[
                    {"id": "C1", "ok": True, "record_value": "2026-09-30", "source_value": "2026-10-30"},
                    {"id": "C2", "ok": True},
                ]
            ),
            CL,
            "the record has '2026-09-30'",
            False,
        ),
        (
            _v(
                checks=[
                    {"id": "C1", "ok": True, "record_value": "2026-11-30", "source_value": "2026-11-30"},
                    {"id": "C2", "ok": True},
                ]
            ),
            CL,
            "does not appear on the source",
            True,
        ),
        (
            _v(checks=[{"id": "C1", "ok": False, "record_value": "x", "source_value": "x"}, {"id": "C2", "ok": True}]),
            CL,
            "marked not ok",
            False,
        ),
    ],
)
def test_enforce_rejects(verdict, checklist, why, inconclusive):
    v = enforce(verdict, ITEMS, checklist, SEEN, [REC], W_)
    assert v["passed"] is False and why in v["reason"], v["reason"]
    assert bool(v.get("inconclusive")) is inconclusive


def test_enforce_ignores_prose_values_on_conditions():
    # live (20261004-131240-28f8): a correct audit was rejected because a CONDITION's source_value was a sentence
    v = _v(
        checks=[
            {"id": "C1", "ok": True, "record_value": "2026-10-30", "source_value": "2026-10-30"},
            {"id": "C2", "ok": True, "record_value": "Bill #4", "source_value": "Amount and due date match"},
        ]
    )
    assert enforce(v, ITEMS, CL, SEEN, [REC], W_)["passed"] is True


def test_enforce_infers_a_missing_source():
    # live (reverify of 20261004-131240-28f8): a complete audit with "source" left empty
    v = _v(source="")
    assert enforce(v, ITEMS, CL, SEEN, [REC], W_)["passed"] is True
    # ...but only from a page in a source app that carries every field value
    assert enforce(_v(source=""), ITEMS, CL, {REC: SEEN[REC]}, [REC], W_)["passed"] is False


# ----------------------------------------------------------------- references: the vendor is matched by ID
DIRECTORY = W + "/erp/vendors"
DIR_TEXT = (  # how the ERP's vendor list reads in an observation: element lines carry the IDs, rows the emails
    '  [8] link "Edit Acme Supplies Inc." -> /erp/vendors/1/edit\n'
    '  [9] link "Edit Acme Logistics GmbH" -> /erp/vendors/2/edit\n'
    '  [10] link "Edit Globex Corporation" -> /erp/vendors/3/edit\n'
    "PAGE TEXT:\nName\tBilling email\tTerms\n"
    "Acme Supplies Inc.\tbilling@acme-supplies.example\tNet 30\tEdit Acme Supplies Inc.\n"
    "Acme Logistics GmbH\tar@acme-logistics.example\tNet 45\tEdit Acme Logistics GmbH\n"
    "Globex Corporation\tar@globex.example\tNet 30\tEdit Globex Corporation"
)
GLOBEX_MAIL = W + "/mail/3"
REF_ITEMS = [("C1", "Vendor", "ref"), ("C2", "Due date", "field")]
REF_CL = Checklist(["Vendor", "Due date"], [], ["mail", "acme"], False, ["Vendor"])


def _ref(record_id, source_id, key, directory=DIRECTORY):
    return {
        "id": "C1",
        "ok": True,
        "record_id": record_id,
        "source_id": source_id,
        "source_key": key,
        "directory": directory,
    }


def _ref_verdict(ref, source=GLOBEX_MAIL):
    return _v(
        source=source,
        checks=[ref, {"id": "C2", "ok": True, "record_value": "2026-10-28", "source_value": "2026-10-28"}],
    )


SEEN_REF = {
    GLOBEX_MAIL: ["From: Globex Receivables <ar@globex.example>\nRechnung GX-5531\nDue: 2026-10-28"],
    DIRECTORY: [DIR_TEXT],
    REC: ["Bill #4 saved."],
}


def test_reference_passes_by_id_not_by_name():
    # live (20261004-143332-a14a): "Globex Corporation" in the ERP vs the email's "Globex Receivables". By ID
    # through the sender's address in the vendor list, it is the same vendor: a pass, without any name rule.
    v = enforce(_ref_verdict(_ref("3", "#3", "ar@globex.example")), REF_ITEMS, REF_CL, SEEN_REF, [REC], W_)
    assert v["passed"] is True, v["reason"]


@pytest.mark.parametrize(
    "ref, why, wrong",
    [
        # name similarity is not an answer: no IDs, no key
        (
            {"id": "C1", "ok": True, "record_value": "Globex Corporation", "source_value": "Globex Receivables"},
            "needs record_id, source_id, source_key and directory",
            False,
        ),
        (_ref("3", "3", "ar@globex.example", directory=W + "/erp/bills"), "was not opened by the auditor", False),
        (_ref("3", "3", "ar@initech.example"), "does not appear on the source", False),
        (_ref("7", "7", "ar@globex.example"), "ID '7' does not appear in the directory", False),
        (_ref("2", "3", "ar@globex.example"), "the record belongs to ID 2 but the source maps to ID 3", True),
    ],
)
def test_reference_rejections(ref, why, wrong):
    v = enforce(_ref_verdict(ref), REF_ITEMS, REF_CL, SEEN_REF, [REC], W_)
    assert v["passed"] is False and why in v["reason"], v["reason"]
    assert bool(v.get("inconclusive")) is (not wrong)  # a wrong vendor is the worker's to fix; the rest, unchecked


def test_a_field_the_record_lacks_is_accepted_only_if_no_page_shows_it():
    # live (reverify): the checklist asked for "Description", the ERP bill only has notes
    items = [("C1", "Description", "field"), ("C2", "Due date", "field")]
    cl = Checklist(["Description", "Due date"], [], ["mail"], False)
    seen = {SRC: SEEN[SRC], REC: ["Bill #4\nVendor Initech LLC\nDue date 2026-10-30\nNotes: support"]}

    def checks(field_value):
        return [
            {"id": "C1", "ok": True, "record_value": field_value, "source_value": "support retainer"},
            {"id": "C2", "ok": True, "record_value": "2026-10-30", "source_value": "2026-10-30"},
        ]

    assert enforce(_v(checks=checks("not in record")), items, cl, seen, [REC], W_)["passed"] is True
    # ...but not to skip a field the record shows
    skip_due = [
        {**checks("x")[0], "record_value": "not in record"},
        {"id": "C2", "ok": True, "record_value": "not in record", "source_value": "2026-10-30"},
    ]
    items2 = [("C1", "Description", "field"), ("C2", "Due date", "field")]
    v = enforce(_v(checks=skip_due), items2, cl, seen, [REC], W_)
    assert v["passed"] is False and "C2 (Due date) said to be not in the record" in v["reason"], v["reason"]


def test_reference_key_must_identify_exactly_one_directory_row():
    # "Acme" fits Acme Supplies AND Acme Logistics: a name fragment cannot pick the vendor
    seen = {**SEEN_REF, GLOBEX_MAIL: ["From: Acme <x@y>\nDue: 2026-10-28"]}
    v = enforce(_ref_verdict(_ref("1", "1", "Acme")), REF_ITEMS, REF_CL, seen, [REC], W_)
    assert v["passed"] is False and "matches 2 directory rows" in v["reason"], v["reason"]


def test_reference_accepts_the_auditors_own_field_names_but_not_lenient_checks():
    # live (qwen re-audit of 20261004-171043-eef0): all four values right, under its own names
    own = {
        "id": "C1",
        "ok": True,
        "record_id_used": "3",
        "source_id_in_directory": "3",
        "source_key_value": "ar@globex.example",
        "directory_url": DIRECTORY,
    }
    assert enforce(_ref_verdict(own), REF_ITEMS, REF_CL, SEEN_REF, [REC], W_)["passed"] is True
    wrong = {**own, "record_id_used": "2"}
    v = enforce(_ref_verdict(wrong), REF_ITEMS, REF_CL, SEEN_REF, [REC], W_)
    assert v["passed"] is False and "belongs to ID 2" in v["reason"]


def test_a_field_carrying_ids_is_checked_as_a_reference_even_if_not_marked():
    items = [("C1", "Vendor", "field"), ("C2", "Due date", "field")]
    v = enforce(_ref_verdict(_ref("2", "3", "ar@globex.example")), items, REF_CL, SEEN_REF, [REC], W_)
    assert v["passed"] is False and "belongs to ID 2" in v["reason"]


def test_a_pass_missing_evidence_goes_back_to_the_auditor_once_and_wrong_values_do_not(tmp_path, ws):
    """Live (qwen, 20261004-171043-eef0): every field matched, but the vendor reference had no IDs, so a correct run
    ended unverified. A pass rejected for missing evidence goes back to the auditor to complete; a wrong value
    still fails at once."""
    admin_post("/admin/reset")
    worker = [
        *LOGIN,
        (
            "browser_fill",
            {
                "fields": [
                    {"element_id": 6, "value": "Acme Supplies Inc."},
                    {"element_id": 7, "value": "INV-2003"},
                    {"element_id": 8, "value": "615.40"},
                    {"element_id": 10, "value": "2026-09-15"},
                    {"element_id": 11, "value": "2026-10-15"},
                ]
            },
        ),
        ("browser_click", {"element_id": 13}),
        ("finish", {"status": "done", "summary": "Filed INV-2003", "evidence": ["saved"]}),
    ]
    checklist = {
        "fields": ["Vendor", "Invoice number"],
        "references": ["Vendor"],
        "conditions": [],
        "source_apps": ["acme"],
        "values_in_task": False,
    }
    common = [
        {"id": "C2", "ok": True, "record_value": "INV-2003", "source_value": "INV-2003"},
        {"id": "C3", "ok": True},
    ]
    no_ids = {
        "passed": True,
        "reason": "all match",
        "evidence": ["portal"],
        "source": W + "/acme/invoices/INV-2003",
        "checks": [{"id": "C1", "ok": True, "record_value": "Acme Supplies Inc.", "source_value": "Acme"}, *common],
    }
    with_ids = {
        **no_ids,
        "checks": [
            {
                "id": "C1",
                "ok": True,
                "record_id": "1",
                "source_id": "1",
                "source_key": "Acme Supplies Inc.",
                "directory": DIRECTORY,
            },
            *common,
        ],
    }

    def audit(second):
        return [
            ("login", {"site": "acme"}),
            ("browser_goto", {"url": W + "/acme/invoices/INV-2003"}),
            ("browser_goto", {"url": DIRECTORY}),
            ("verdict", no_ids),
            ("verdict", second),
        ]

    def run(second):
        llm = RoleLLM(
            {
                "planner": [_plan("File INV-2003", ["A bill for INV-2003 exists"])],
                "worker": list(worker),
                "distiller": [{"notes": []}],
                "verifier_checklist": [checklist],
                "verifier": audit(second),
            }
        )
        agent, events = make_agent(tmp_path, ws, llm, max_verify_rounds=0)
        r = agent.run("File Acme's expedited shipping surcharge invoice in the ERP.")
        notes = [d["text"] for k, d in events if k == "verify_step" and d["tool"] == "verdict"]
        return r, next(d for k, d in events if k == "verify_result"), notes, llm

    r, verdict, notes, llm = run(with_ids)
    assert r.status == "verified" and verdict["passed"] is True
    assert notes and "needs record_id, source_id" in notes[0]  # the rejected first verdict, sent back
    feedback = [m["content"] for m in llm.seen[-2] if m.get("role") == "tool"][-1]
    assert feedback.startswith("Verdict NOT accepted")
    admin_post("/admin/reset")
    wrong = {**with_ids, "checks": [{**with_ids["checks"][0], "record_id": "2"}, *common]}  # a wrong vendor
    r, verdict, notes, _ = run(wrong)
    assert r.status == "unverified" and "belongs to ID 2" in verdict["reason"]


def test_lookalike_vendor_is_not_verified(tmp_path, ws):
    """lookalike_vendor: Acme Supplies' shipping surcharge INV-2003 filed under "Acme Logistics GmbH". An auditor
    reporting the IDs it finds cannot pass it, however similar the names look."""
    admin_post("/admin/reset")
    worker = [
        *LOGIN,
        (
            "browser_fill",
            {
                "fields": [
                    {"element_id": 6, "value": "Acme Logistics GmbH"},  # the lookalike
                    {"element_id": 7, "value": "INV-2003"},
                    {"element_id": 8, "value": "615.40"},
                    {"element_id": 10, "value": "2026-09-15"},
                    {"element_id": 11, "value": "2026-10-15"},
                ]
            },
        ),
        ("browser_click", {"element_id": 13}),
        ("finish", {"status": "done", "summary": "Filed Acme's shipping invoice INV-2003", "evidence": ["saved"]}),
    ]
    checklist = {
        "fields": ["Vendor", "Invoice number", "Amount", "Currency", "Invoice date", "Due date"],
        "references": ["Vendor"],
        "conditions": [],
        "source_apps": ["acme"],
        "values_in_task": False,
    }
    checks = [
        {
            "id": "C1",
            "ok": True,
            "record_id": "2",
            "source_id": "1",
            "source_key": "Acme Supplies Inc.",
            "directory": DIRECTORY,
        },
        {"id": "C2", "ok": True, "record_value": "INV-2003", "source_value": "INV-2003"},
        {"id": "C3", "ok": True, "record_value": "615.40", "source_value": "615.40"},
        {"id": "C4", "ok": True, "record_value": "USD", "source_value": "USD"},
        {"id": "C5", "ok": True, "record_value": "2026-09-15", "source_value": "15 Sep 2026"},
        {"id": "C6", "ok": True, "record_value": "2026-10-15", "source_value": "15 Oct 2026"},
        {"id": "C7", "ok": True},
    ]
    llm = RoleLLM(
        {
            "planner": [_plan("File Acme's shipping invoice", ["A bill for INV-2003 exists in the ERP"])],
            "worker": worker,
            "distiller": [{"notes": []}],
            "verifier_checklist": [checklist],
            "verifier": [
                ("login", {"site": "acme"}),
                ("browser_goto", {"url": W + "/acme/invoices/INV-2003"}),
                ("browser_goto", {"url": DIRECTORY}),
                ("browser_goto", {"url": W + "/erp/bills?q=INV-2003"}),
                (
                    "verdict",
                    {
                        "passed": True,
                        "reason": "Acme invoice filed; vendor is an Acme company",
                        "evidence": ["portal", "bill"],
                        "source": W + "/acme/invoices/INV-2003",
                        "checks": checks,
                    },
                ),
            ],
        }
    )
    agent, events = make_agent(tmp_path, ws, llm, max_verify_rounds=0)
    r = agent.run("File Acme's expedited shipping surcharge invoice in the ERP.")
    verdict = next(d for k, d in events if k == "verify_result")
    assert r.status == "unverified" and "belongs to ID 2 but the source maps to ID 1" in verdict["reason"], verdict
    assert not verdict.get("inconclusive")


def test_enforce_leaves_fails_alone_and_task_source_needs_values_in_task():
    assert enforce(_v(passed=False), ITEMS, CL, SEEN, [], W_)["passed"] is False
    in_task = Checklist(["Due date"], [], [], True)
    v = _v(
        source="task",
        checks=[
            {"id": "C1", "ok": True, "record_value": "2026-10-30", "source_value": "2026-10-30"},
            {"id": "C2", "ok": True},
        ],
    )
    assert enforce(v, ITEMS, in_task, SEEN, [REC], W_)["passed"] is True


def test_validate_checklist():
    apps = W_.app_names()
    assert validate_checklist({"fields": ["a"], "source_apps": ["mail"]}, apps)[0] is not None
    assert "unknown app" in validate_checklist({"fields": ["a"], "source_apps": ["sap"]}, apps)[1]
    assert "both empty" in validate_checklist({"fields": [], "source_apps": ["mail"]}, apps)[1]
    assert "where do the values" in validate_checklist({"fields": ["a"], "source_apps": []}, apps)[1]
    assert validate_checklist({"fields": ["a"], "values_in_task": True}, apps)[0] is not None


def test_world_maps_locations_to_apps():
    assert W_.app_of(W + "/mail/5") == "mail" and W_.app_of(W + "/erp/bills?q=1") == "erp"
    assert W_.app_of("workspace file q3.csv") == "workspace" and W_.app_of(W + "/") is None
    assert W_.app_of(W + "/erpx") is None  # a prefix match is per path segment


# ----------------------------------------------------------------- fix 2: the auditor's own model + key rotation
def test_key_rotation_and_verifier_model_settings():
    from agent.config import Settings

    env = {
        "LLM_API_KEY": "g",
        "NVIDIA_API_KEY": "k1, k2",
        "LLM_MODEL": "nvidia:m",
        "AUTOWORK_VERIFIER_MODEL": "nvidia:v",
    }
    s = Settings.from_env(env)
    assert s.llm_api_key == "k1" and [(f.model, f.api_key) for f in s.llm_fallbacks] == [("nvidia:m", "k2")]
    vs = s.verifier_settings(env)
    assert vs.llm_model == "nvidia:v" and vs.llm_api_key == "k1" and [f.api_key for f in vs.llm_fallbacks] == ["k2"]
    assert s.for_model("nvidia:m", env).llm_fallbacks == s.llm_fallbacks  # rotation survives; model switches don't
    assert Settings.from_env({**env, "AUTOWORK_VERIFIER_MODEL": "nvidia:m"}).verifier_settings(env) is None
    with pytest.raises(ValueError, match="AUTOWORK_VERIFIER_MODEL"):
        Settings.from_env({"LLM_API_KEY": "g", "AUTOWORK_VERIFIER_MODEL": "gemini:x"})  # no GEMINI_API_KEY


def test_separate_verifier_model_does_the_audit_and_is_reported(tmp_path, ws):
    from conftest import AUDIT_ACME_OK, PLAN, FakeLLM

    admin_post("/admin/reset")
    worker = RoleLLM(
        {
            "planner": [PLAN],
            "worker": [
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
                ("browser_click", {"element_id": 13}),
                ("finish", {"status": "done", "summary": "Entered INV-2041", "evidence": ["saved"]}),
            ],
            "distiller": [{"notes": []}],
        }
    )
    auditor = FakeLLM(AUDIT_ACME_OK)
    auditor.model = "scripted-auditor"
    agent, events = make_agent(tmp_path, ws, worker, verifier_llm=auditor)
    r = agent.run("Enter the latest Acme invoice")
    assert r.status == "verified", next(d for k, d in events if k == "verify_result")
    assert "verifier" not in worker.roles and "verifier_checklist" not in worker.roles
    saved = json.loads((tmp_path / "runs" / r.run_id / "report.json").read_text())
    assert saved["verifier_model"] == "scripted-auditor" and saved["llm"]["verifier"]["calls"] == len(AUDIT_ACME_OK)
