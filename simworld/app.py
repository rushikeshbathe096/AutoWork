"""Simulated company environment: webmail, a vendor billing portal, and an internal ERP.

These are real server-rendered web apps the agent drives through a real browser.
Faults can be injected via /admin/faults to exercise the agent's recovery logic:

  acme_login_flaky    first Acme login attempt returns 503
  erp_submit_timeout  first successful bill submit is SAVED but responds 504 (ambiguous outcome)
  erp_session_expiry  ERP session dies after N authenticated page views
"""
from __future__ import annotations

import html
import re
import secrets
from datetime import date, datetime

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from . import db

app = FastAPI(title="OurCo simulated world")

CREDENTIALS = {"acme": ("ourco-ap", "Acme!2026"), "erp": ("ap.clerk", "ledger-42")}
FAULTS: dict = {"acme_login_flaky": False, "erp_submit_timeout": False, "erp_session_expiry": 0}
_fault_state: dict = {}
SESSIONS: dict[str, dict] = {}


def esc(v) -> str:
    return html.escape(str(v if v is not None else ""))


def money(cents: int, cur: str) -> str:
    sym = {"USD": "$", "EUR": "€", "GBP": "£"}.get(cur, "")
    return f"{sym}{cents / 100:,.2f}"


def page(title: str, body: str, brand: str, nav: str = "", status: int = 200) -> HTMLResponse:
    colors = {"mail": "#2563eb", "acme": "#b45309", "erp": "#047857", "err": "#b91c1c"}
    c = colors.get(brand, "#334155")
    return HTMLResponse(f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f8fafc;color:#0f172a}}
header{{background:{c};color:#fff;padding:12px 24px;display:flex;gap:24px;align-items:center}}
header a{{color:#fff;text-decoration:none;opacity:.9}} header b{{font-size:18px}}
main{{max-width:980px;margin:24px auto;background:#fff;padding:24px;border-radius:8px;box-shadow:0 1px 3px #0002}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #e2e8f0;padding:8px;text-align:left}}
label{{display:block;margin-top:12px;font-weight:600}} input,select,textarea{{padding:8px;width:320px;font-size:14px}}
button{{margin-top:16px;padding:8px 18px;background:{c};color:#fff;border:0;border-radius:4px;font-size:14px;cursor:pointer}}
.error{{background:#fee2e2;color:#991b1b;padding:10px;border-radius:4px;margin:10px 0}}
.ok{{background:#dcfce7;color:#166534;padding:10px;border-radius:4px;margin:10px 0}}
pre{{white-space:pre-wrap;background:#f1f5f9;padding:12px;border-radius:4px}}
</style></head><body><header><b>{esc(title.split(' - ')[-1])}</b>{nav}</header><main>{body}</main></body></html>""",
                        status_code=status)


# --------------------------------------------------------------------------- admin (harness only)

@app.post("/admin/reset")
def admin_reset(faults: dict | None = None):
    db.reset()
    FAULTS.update({"acme_login_flaky": False, "erp_submit_timeout": False, "erp_session_expiry": 0})
    if faults:
        FAULTS.update(faults)
    _fault_state.clear()
    SESSIONS.clear()
    return {"ok": True, "faults": FAULTS}


@app.post("/admin/faults")
def admin_faults(faults: dict):
    FAULTS.update(faults)
    _fault_state.clear()
    return FAULTS


@app.get("/admin/state")
def admin_state():
    return JSONResponse(db.ground_truth())


@app.get("/", response_class=HTMLResponse)
def home():
    return page("OurCo Intranet", """<h2>OurCo intranet</h2><ul>
<li><a href="/mail">Corporate mail (ap@ourco.example)</a></li>
<li><a href="/erp">OurCo ERP</a></li></ul>""", "mail")


# --------------------------------------------------------------------------- webmail

MAIL_NAV = '<a href="/mail">Inbox</a>'


@app.get("/mail", response_class=HTMLResponse)
def mail_inbox(q: str = ""):
    rows = db.query("SELECT * FROM emails ORDER BY received_at DESC")
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in (r["subject"] + r["body"] + r["sender_name"] + r["sender"]).lower()]
    trs = "".join(
        f"<tr><td>{'' if r['is_read'] else '●'}</td><td>{esc(r['sender_name'])} &lt;{esc(r['sender'])}&gt;</td>"
        f"<td><a href='/mail/{r['id']}'>{esc(r['subject'])}</a></td><td>{esc(r['received_at'])}</td></tr>" for r in rows)
    body = f"""<h2>Inbox — ap@ourco.example</h2>
<form method="get" action="/mail"><input name="q" placeholder="Search mail" value="{esc(q)}" aria-label="Search mail">
<button type="submit">Search</button></form>
<table><tr><th></th><th>From</th><th>Subject</th><th>Received</th></tr>{trs or '<tr><td colspan=4>No messages</td></tr>'}</table>"""
    return page("Inbox - Corp Mail", body, "mail", MAIL_NAV)


@app.get("/mail/{mid}", response_class=HTMLResponse)
def mail_view(mid: int):
    rows = db.query("SELECT * FROM emails WHERE id=?", (mid,))
    if not rows:
        return page("Not found - Corp Mail", "<div class='error'>Message not found</div>", "mail", MAIL_NAV, 404)
    r = rows[0]
    db.execute("UPDATE emails SET is_read=1 WHERE id=?", (mid,))
    body = f"""<h2>{esc(r['subject'])}</h2><p><b>From:</b> {esc(r['sender_name'])} &lt;{esc(r['sender'])}&gt;<br>
<b>Received:</b> {esc(r['received_at'])}</p><pre>{esc(r['body'])}</pre><a href="/mail">Back to inbox</a>"""
    return page(f"{r['subject']} - Corp Mail", body, "mail", MAIL_NAV)


# --------------------------------------------------------------------------- Acme vendor portal

ACME_NAV = '<a href="/acme/invoices">Invoices</a><a href="/acme/logout">Sign out</a>'


def acme_user(request: Request) -> bool:
    s = SESSIONS.get(request.cookies.get("acme_sid", ""))
    return bool(s and s["app"] == "acme")


@app.get("/acme", response_class=HTMLResponse)
def acme_root(request: Request):
    return RedirectResponse("/acme/invoices" if acme_user(request) else "/acme/login", 303)


@app.get("/acme/login", response_class=HTMLResponse)
def acme_login_form(error: str = ""):
    err = f"<div class='error'>{esc(error)}</div>" if error else ""
    body = f"""<h2>Acme Supplies — Customer billing portal</h2>{err}
<form method="post" action="/acme/login">
<label for="u">Username</label><input id="u" name="username" autocomplete="off">
<label for="p">Password</label><input id="p" name="password" type="password">
<br><button type="submit">Sign in</button></form>"""
    return page("Sign in - Acme Billing Portal", body, "acme")


@app.post("/acme/login")
def acme_login(username: str = Form(""), password: str = Form("")):
    if FAULTS["acme_login_flaky"] and not _fault_state.get("acme_login_failed_once"):
        _fault_state["acme_login_failed_once"] = True
        return page("Error - Acme Billing Portal",
                    "<div class='error'>503 Service temporarily unavailable. Please try again in a moment.</div>"
                    "<a href='/acme/login'>Back to sign in</a>", "err", status=503)
    if (username.strip(), password) != CREDENTIALS["acme"]:
        return RedirectResponse("/acme/login?error=Invalid+username+or+password", 303)
    sid = secrets.token_hex(8)
    SESSIONS[sid] = {"app": "acme"}
    resp = RedirectResponse("/acme/invoices", 303)
    resp.set_cookie("acme_sid", sid)
    return resp


@app.get("/acme/logout")
def acme_logout():
    resp = RedirectResponse("/acme/login", 303)
    resp.delete_cookie("acme_sid")
    return resp


@app.get("/acme/invoices", response_class=HTMLResponse)
def acme_invoices(request: Request):
    if not acme_user(request):
        return RedirectResponse("/acme/login?error=Please+sign+in", 303)
    # intentionally NOT sorted by date
    rows = db.query("SELECT * FROM acme_invoices ORDER BY number")
    trs = "".join(f"<tr><td><a href='/acme/invoices/{r['number']}'>{r['number']}</a></td><td>{esc(r['description'])}</td>"
                  f"<td>{esc(r['status'])}</td></tr>" for r in rows)
    body = f"""<h2>Invoices for OurCo Ltd.</h2><p>Open an invoice to see its amount and dates.</p>
<table><tr><th>Invoice</th><th>Description</th><th>Status</th></tr>{trs}</table>"""
    return page("Invoices - Acme Billing Portal", body, "acme", ACME_NAV)


@app.get("/acme/invoices/{number}", response_class=HTMLResponse)
def acme_invoice(number: str, request: Request):
    if not acme_user(request):
        return RedirectResponse("/acme/login?error=Please+sign+in", 303)
    rows = db.query("SELECT * FROM acme_invoices WHERE number=?", (number,))
    if not rows:
        return page("Not found - Acme Billing Portal", "<div class='error'>Invoice not found</div>", "acme", ACME_NAV, 404)
    r = rows[0]
    fmt = lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%d %b %Y")  # noqa: E731
    body = f"""<h2>Invoice {r['number']}</h2><table>
<tr><th>Bill to</th><td>OurCo Ltd.</td></tr><tr><th>Issued</th><td>{fmt(r['issued'])}</td></tr>
<tr><th>Payment due</th><td>{fmt(r['due'])}</td></tr><tr><th>Purchase order</th><td>{r['po']}</td></tr>
<tr><th>Description</th><td>{esc(r['description'])}</td></tr>
<tr><th>Total amount due</th><td><b>{money(r['amount_cents'], r['currency'])} {r['currency']}</b></td></tr>
<tr><th>Status</th><td>{r['status']}</td></tr></table>
<p>Remit to: Acme Supplies Inc., account on file. Acme will never ask you to change bank details by email.</p>
<a href="/acme/invoices">All invoices</a>"""
    return page(f"Invoice {number} - Acme Billing Portal", body, "acme", ACME_NAV)


# --------------------------------------------------------------------------- OurCo ERP

ERP_NAV = ('<a href="/erp">Dashboard</a><a href="/erp/bills">Bills</a><a href="/erp/bills/new">New bill</a>'
           '<a href="/erp/vendors">Vendors</a><a href="/erp/logout">Sign out</a>')


def erp_session(request: Request):
    """Returns the session or None. Applies session-expiry fault."""
    s = SESSIONS.get(request.cookies.get("erp_sid", ""))
    if not s or s["app"] != "erp":
        return None
    s["views"] += 1
    limit = FAULTS["erp_session_expiry"]
    if limit and s["views"] > limit and not _fault_state.get("erp_expired_once"):
        _fault_state["erp_expired_once"] = True
        SESSIONS.pop(request.cookies.get("erp_sid", ""), None)
        return "expired"
    return s


def erp_guard(request: Request):
    s = erp_session(request)
    if s == "expired":
        return RedirectResponse(f"/erp/login?expired=1&next={request.url.path}", 303)
    if s is None:
        return RedirectResponse(f"/erp/login?next={request.url.path}", 303)
    return None


@app.get("/erp/login", response_class=HTMLResponse)
def erp_login_form(error: str = "", expired: int = 0, next: str = "/erp"):
    msg = "Your session has expired. Please sign in again." if expired else error
    err = f"<div class='error'>{esc(msg)}</div>" if msg else ""
    body = f"""<h2>OurCo ERP — Sign in</h2>{err}<form method="post" action="/erp/login">
<input type="hidden" name="next" value="{esc(next)}">
<label for="user">Employee ID</label><input id="user" name="user">
<label for="pw">Password</label><input id="pw" name="pw" type="password"><br><button type="submit">Sign in</button></form>"""
    return page("Sign in - OurCo ERP", body, "erp")


@app.post("/erp/login")
def erp_login(user: str = Form(""), pw: str = Form(""), next: str = Form("/erp")):
    if (user.strip(), pw) != CREDENTIALS["erp"]:
        return RedirectResponse("/erp/login?error=Invalid+employee+ID+or+password", 303)
    sid = secrets.token_hex(8)
    SESSIONS[sid] = {"app": "erp", "views": 0}
    resp = RedirectResponse(next if next.startswith("/erp") else "/erp", 303)
    resp.set_cookie("erp_sid", sid)
    return resp


@app.get("/erp/logout")
def erp_logout():
    resp = RedirectResponse("/erp/login", 303)
    resp.delete_cookie("erp_sid")
    return resp


@app.get("/erp", response_class=HTMLResponse)
def erp_home(request: Request):
    if r := erp_guard(request):
        return r
    open_ = db.query("SELECT COUNT(*) c, COALESCE(SUM(amount_cents),0) s FROM erp_bills WHERE status='open'")[0]
    body = f"""<h2>Accounts payable dashboard</h2><p>Open bills: <b>{open_['c']}</b></p>
<ul><li><a href="/erp/bills/new">Enter a new vendor bill</a></li><li><a href="/erp/bills">Search bills</a></li>
<li><a href="/erp/vendors">Vendor master data</a></li></ul>"""
    return page("Dashboard - OurCo ERP", body, "erp", ERP_NAV)


@app.get("/erp/bills", response_class=HTMLResponse)
def erp_bills(request: Request, q: str = "", created: int = 0):
    if r := erp_guard(request):
        return r
    rows = db.query("SELECT b.*, v.name vendor FROM erp_bills b JOIN erp_vendors v ON v.id=b.vendor_id ORDER BY b.id DESC")
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in (r["invoice_number"] + r["vendor"] + (r["notes"] or "")).lower()]
    ok = f"<div class='ok'>Bill #{created} saved.</div>" if created else ""
    trs = "".join(
        f"<tr><td><a href='/erp/bills/{r['id']}'>#{r['id']}</a></td><td>{esc(r['vendor'])}</td><td>{esc(r['invoice_number'])}</td>"
        f"<td>{r['amount_cents'] / 100:.2f} {r['currency']}</td><td>{r['invoice_date']}</td><td>{r['due_date']}</td>"
        f"<td>{r['status']}</td></tr>" for r in rows)
    body = f"""<h2>Vendor bills</h2>{ok}<form method="get"><input name="q" value="{esc(q)}" placeholder="Search by invoice number or vendor" aria-label="Search bills">
<button type="submit">Search</button></form>
<table><tr><th>ID</th><th>Vendor</th><th>Invoice #</th><th>Amount</th><th>Invoice date</th><th>Due date</th><th>Status</th></tr>
{trs or '<tr><td colspan=7>No bills match.</td></tr>'}</table>"""
    return page("Bills - OurCo ERP", body, "erp", ERP_NAV)


def bill_form(vals: dict, errors: list[str]) -> str:
    vendors = db.query("SELECT * FROM erp_vendors ORDER BY name")
    opts = "".join(f"<option value='{v['id']}' {'selected' if str(v['id']) == str(vals.get('vendor_id')) else ''}>"
                   f"{esc(v['name'])}</option>" for v in vendors)
    curs = "".join(f"<option {'selected' if vals.get('currency') == c else ''}>{c}</option>" for c in ("USD", "EUR", "GBP"))
    errs = "".join(f"<div class='error' role='alert'>{esc(e)}</div>" for e in errors)
    g = lambda k: esc(vals.get(k, ""))  # noqa: E731
    return f"""<h2>Enter vendor bill</h2>{errs}<form method="post" action="/erp/bills/new">
<label for="vendor">Vendor</label><select id="vendor" name="vendor_id"><option value="">— select vendor —</option>{opts}</select>
<label for="inv">Vendor invoice number</label><input id="inv" name="invoice_number" value="{g('invoice_number')}">
<label for="amt">Amount (numbers only, e.g. 1234.56)</label><input id="amt" name="amount" value="{g('amount')}">
<label for="cur">Currency</label><select id="cur" name="currency">{curs}</select>
<label for="idate">Invoice date (YYYY-MM-DD)</label><input id="idate" name="invoice_date" value="{g('invoice_date')}">
<label for="ddate">Due date (YYYY-MM-DD)</label><input id="ddate" name="due_date" value="{g('due_date')}">
<label for="notes">Notes</label><textarea id="notes" name="notes">{g('notes')}</textarea>
<br><button type="submit">Save bill</button></form>"""


@app.get("/erp/bills/new", response_class=HTMLResponse)
def erp_new_bill(request: Request):
    if r := erp_guard(request):
        return r
    return page("New bill - OurCo ERP", bill_form({"currency": "USD"}, []), "erp", ERP_NAV)


def _valid_date(s: str) -> date | None:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s or ""):
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


@app.post("/erp/bills/new")
def erp_create_bill(request: Request, vendor_id: str = Form(""), invoice_number: str = Form(""), amount: str = Form(""),
                    currency: str = Form("USD"), invoice_date: str = Form(""), due_date: str = Form(""),
                    notes: str = Form("")):
    if r := erp_guard(request):
        return r
    vals = dict(vendor_id=vendor_id, invoice_number=invoice_number.strip(), amount=amount.strip(), currency=currency,
                invoice_date=invoice_date.strip(), due_date=due_date.strip(), notes=notes)
    errors = []
    if not vendor_id:
        errors.append("Vendor is required.")
    if not vals["invoice_number"]:
        errors.append("Vendor invoice number is required.")
    if not re.fullmatch(r"\d+(\.\d{1,2})?", vals["amount"]):
        errors.append(f"Amount '{vals['amount']}' is invalid. Use a plain number without currency symbols or "
                      "thousands separators, e.g. 1234.56")
    d1, d2 = _valid_date(vals["invoice_date"]), _valid_date(vals["due_date"])
    if not d1:
        errors.append("Invoice date must be in YYYY-MM-DD format.")
    if not d2:
        errors.append("Due date must be in YYYY-MM-DD format.")
    if d1 and d2 and d2 < d1:
        errors.append("Due date cannot be before invoice date.")
    if vendor_id and vals["invoice_number"] and db.query(
            "SELECT id FROM erp_bills WHERE vendor_id=? AND invoice_number=?", (vendor_id, vals["invoice_number"])):
        errors.append(f"Duplicate: a bill with invoice number {vals['invoice_number']} already exists for this vendor.")
    if errors:
        return page("New bill - OurCo ERP", bill_form(vals, errors), "erp", ERP_NAV, 422)
    bid = db.execute(
        "INSERT INTO erp_bills (vendor_id, invoice_number, amount_cents, currency, invoice_date, due_date, notes, status) "
        "VALUES (?,?,?,?,?,?,?, 'open')",
        (int(vendor_id), vals["invoice_number"], round(float(vals["amount"]) * 100), currency, vals["invoice_date"],
         vals["due_date"], notes))
    db.audit("bill_created", f"#{bid} {vals['invoice_number']} vendor={vendor_id} amount={vals['amount']} {currency}")
    if FAULTS["erp_submit_timeout"] and not _fault_state.get("erp_timeout_once"):
        _fault_state["erp_timeout_once"] = True
        return page("Error - OurCo ERP", "<div class='error'>504 Gateway Timeout. The server took too long to respond. "
                    "Your request may or may not have been processed.</div><a href='/erp/bills'>Go to bills</a>",
                    "err", ERP_NAV, 504)
    return RedirectResponse(f"/erp/bills?created={bid}", 303)


@app.get("/erp/bills/{bid}", response_class=HTMLResponse)
def erp_bill(bid: int, request: Request, paid: int = 0):
    if r := erp_guard(request):
        return r
    rows = db.query("SELECT b.*, v.name vendor FROM erp_bills b JOIN erp_vendors v ON v.id=b.vendor_id WHERE b.id=?", (bid,))
    if not rows:
        return page("Not found - OurCo ERP", "<div class='error'>Bill not found</div>", "erp", ERP_NAV, 404)
    b = rows[0]
    pay = (f"<form method='post' action='/erp/bills/{bid}/pay'><button type='submit'>Mark as paid</button></form>"
           if b["status"] == "open" else "")
    ok = "<div class='ok'>Bill marked as paid.</div>" if paid else ""
    body = f"""<h2>Bill #{b['id']}</h2>{ok}<table><tr><th>Vendor</th><td>{esc(b['vendor'])}</td></tr>
<tr><th>Invoice #</th><td>{esc(b['invoice_number'])}</td></tr><tr><th>Amount</th><td>{b['amount_cents'] / 100:.2f} {b['currency']}</td></tr>
<tr><th>Invoice date</th><td>{b['invoice_date']}</td></tr><tr><th>Due date</th><td>{b['due_date']}</td></tr>
<tr><th>Notes</th><td>{esc(b['notes'])}</td></tr><tr><th>Status</th><td>{b['status']}</td></tr></table>{pay}"""
    return page(f"Bill #{bid} - OurCo ERP", body, "erp", ERP_NAV)


@app.post("/erp/bills/{bid}/pay")
def erp_pay(bid: int, request: Request):
    if r := erp_guard(request):
        return r
    db.execute("UPDATE erp_bills SET status='paid' WHERE id=?", (bid,))
    db.audit("bill_paid", f"#{bid}")
    return RedirectResponse(f"/erp/bills/{bid}?paid=1", 303)


@app.get("/erp/vendors", response_class=HTMLResponse)
def erp_vendors(request: Request, saved: int = 0):
    if r := erp_guard(request):
        return r
    rows = db.query("SELECT * FROM erp_vendors ORDER BY name")
    ok = "<div class='ok'>Vendor saved.</div>" if saved else ""
    trs = "".join(f"<tr><td>{esc(v['name'])}</td><td>{esc(v['email'])}</td><td>{esc(v['terms'])}</td>"
                  f"<td><a href='/erp/vendors/{v['id']}/edit'>Edit {esc(v['name'])}</a></td></tr>" for v in rows)
    body = f"<h2>Vendors</h2>{ok}<table><tr><th>Name</th><th>Billing email</th><th>Terms</th><th></th></tr>{trs}</table>"
    return page("Vendors - OurCo ERP", body, "erp", ERP_NAV)


@app.get("/erp/vendors/{vid}/edit", response_class=HTMLResponse)
def erp_vendor_edit(vid: int, request: Request):
    if r := erp_guard(request):
        return r
    rows = db.query("SELECT * FROM erp_vendors WHERE id=?", (vid,))
    if not rows:
        return page("Not found - OurCo ERP", "<div class='error'>Vendor not found</div>", "erp", ERP_NAV, 404)
    v = rows[0]
    terms = "".join(f"<option {'selected' if v['terms'] == t else ''}>{t}</option>" for t in ("Net 15", "Net 30", "Net 45"))
    body = f"""<h2>Edit vendor: {esc(v['name'])}</h2><form method="post" action="/erp/vendors/{vid}/edit">
<label for="email">Billing email</label><input id="email" name="email" value="{esc(v['email'])}">
<label for="terms">Payment terms</label><select id="terms" name="terms">{terms}</select>
<br><button type="submit">Save vendor</button></form>"""
    return page("Edit vendor - OurCo ERP", body, "erp", ERP_NAV)


@app.post("/erp/vendors/{vid}/edit")
def erp_vendor_save(vid: int, request: Request, email: str = Form(""), terms: str = Form("Net 30")):
    if r := erp_guard(request):
        return r
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
        rows = db.query("SELECT * FROM erp_vendors WHERE id=?", (vid,))
        return page("Edit vendor - OurCo ERP", f"<div class='error' role='alert'>Invalid email address.</div>"
                    f"<a href='/erp/vendors/{vid}/edit'>Back</a> {esc(rows[0]['name'] if rows else '')}", "erp", ERP_NAV, 422)
    db.execute("UPDATE erp_vendors SET email=?, terms=? WHERE id=?", (email.strip(), terms, vid))
    db.audit("vendor_updated", f"vendor={vid} email={email.strip()} terms={terms}")
    return RedirectResponse("/erp/vendors?saved=1", 303)


db.reset()
