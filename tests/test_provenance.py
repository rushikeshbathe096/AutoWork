"""Value provenance: form values must trace back to the task or to observations (agent/provenance.py)."""

from agent.memory import WorkingMemory
from agent.provenance import Provenance, amounts_in, dates_in, label_words, labelled_values, values_match
from agent.tools import ToolBox

PORTAL = "Issued\t01 Oct 2026\nPayment due\t31 Oct 2026\nTotal amount due\t$4,250.00 USD\nInvoice INV-2041"


def seen(*texts: str) -> Provenance:
    p = Provenance()
    for t in texts:
        p.observe(t)
    return p


def test_dates_are_compared_in_canonical_form():
    assert dates_in("31 Oct 2026") == {"2026-10-31"}
    assert dates_in("Oct 31, 2026") == {"2026-10-31"}
    assert dates_in("01.10.2026") == {"2026-10-01", "2026-01-10"}  # EU vs US reading: both are plausible


def test_amounts_handle_us_and_eu_grouping():
    assert {"4250"} <= amounts_in("$4,250.00")
    assert {"1234.56"} <= amounts_in("1.234,56 EUR")


def test_regression_invented_invoice_date_from_first_live_eval():
    p = seen("Find the latest invoice from Acme ...", PORTAL)
    assert p.is_sourced("2026-10-31") and p.is_sourced("2026-10-01") and p.is_sourced("4250.00")
    assert not p.is_sourced("2026-10-03")  # today's date, typed because the portal page had left the context


def test_free_text_and_empty_values_are_not_checked():
    p = seen(PORTAL)
    assert p.is_sourced("") and p.is_sourced("Entered from the Acme portal, September services, per email")


class FakeBrowser:
    def fill(self, fields):
        return "\n".join(f"[{f['element_id']}] now = {f['value']!r}" for f in fields)

    def snapshot(self, _):  # pragma: no cover - only used on BrowserError
        raise AssertionError


def test_fill_flags_unsourced_values_and_own_outputs_are_not_sources(tmp_path):
    p = seen(PORTAL)
    tb = ToolBox(FakeBrowser(), tmp_path, WorkingMemory(), provenance=p)  # type: ignore[arg-type]
    # Laundering attempt: remembering an invented value must not make it a source.
    tb.run("remember", {"key": "invoice_date", "value": "2026-10-03"}, 1)
    r = tb.run("browser_fill", {"fields": [{"element_id": 10, "value": "2026-10-03"}]}, 2)
    assert not r.ok and "UNSOURCED" in r.text and "2026-10-03" in r.text
    r = tb.run("browser_fill", {"fields": [{"element_id": 10, "value": "2026-10-01"}]}, 3)
    assert r.ok and "UNSOURCED" not in r.text


# ----------------------------------------------------------------- labels (2026-10-04: issue date typed as due date)
MAIL = "Invoice IN-7002\nIssued: 2026-09-30\nAmount: USD 640.00\nDue: 2026-10-30\nItems: September support retainer"


def test_labels_are_read_next_to_values_on_the_line_or_above():
    found = {(k, min(v), lab) for k, v, _, lab in labelled_values(MAIL + "\nPayment due\n31 Oct 2026")}
    assert ("date", "2026-09-30", "Issued") in found and ("date", "2026-10-30", "Due") in found
    assert ("date", "2026-10-31", "Payment due") in found  # label on the line above
    assert label_words("Due date (YYYY-MM-DD)") == {"due"}  # format hints and generic words dropped


def test_label_conflict_issue_date_in_due_date_field():
    p = seen(MAIL)
    why = p.label_conflict("Due date (YYYY-MM-DD)", "2026-09-30")
    assert why and "'Issued'" in why and "'Due'" in why
    assert p.label_conflict("Due date (YYYY-MM-DD)", "2026-10-30") is None  # the right value
    assert p.label_conflict("Invoice date (YYYY-MM-DD)", "2026-09-30") is None  # no label matches "invoice": silent
    q = seen(PORTAL)
    assert q.label_conflict("Due date", "01 Oct 2026") and q.label_conflict("Due date", "2026-10-31") is None


def test_label_conflict_stays_quiet_without_evidence():
    assert seen("Due 2026-10-30").label_conflict("Due date", "2026-10-30") is None
    assert seen("2026-09-30 and 2026-10-30").label_conflict("Due date", "2026-09-30") is None  # unlabelled
    assert seen(MAIL).label_conflict("Notes", "2026-09-30") is None  # the field's name matches no label
    assert seen(MAIL).label_conflict("Vendor", "Initech LLC") is None  # not a date or amount


def test_values_match_across_formats():
    assert values_match("2026-10-30", "30 Oct 2026") and values_match("640.00", "USD 640.00")
    assert values_match("Initech LLC", "Initech") and not values_match("2026-09-30", "2026-10-30")
    assert not values_match("INV-2041", "INV-1987") and values_match("IN-7002", "Invoice IN-7002")
    assert not values_match("Globex Corporation", "Globex Receivables")  # names: by ID in the directory instead
    assert not values_match("Acme Supplies Inc.", "Acme Logistics GmbH")
    assert values_match("USD", "usd") and not values_match("USD", "EUR")


def test_fill_warns_when_the_value_is_labelled_for_another_field(tmp_path):
    class B:
        last = type("S", (), {"element": staticmethod(lambda i: {"id": i, "label": "Due date (YYYY-MM-DD)"})})()

        def fill(self, fields):
            return "Filled fields (read back from page):\n[11] now = '2026-09-30'"

    tb = ToolBox(B(), tmp_path, WorkingMemory(), provenance=seen(MAIL))
    r = tb._t_browser_fill([{"element_id": 11, "value": "2026-09-30"}], step=1)
    assert not r.ok and "LABEL CONFLICT" in r.text and "'Issued'" in r.text
    ok = tb._t_browser_fill([{"element_id": 11, "value": "2026-10-30"}], step=2)
    assert ok.ok and "LABEL CONFLICT" not in ok.text
