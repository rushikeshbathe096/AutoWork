"""Value provenance: form values must trace back to the task or to observations (agent/provenance.py)."""

from agent.memory import WorkingMemory
from agent.provenance import Provenance, amounts_in, dates_in
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
