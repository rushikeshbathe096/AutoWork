"""Value provenance: every value the agent types into a form is checked against what it observed. An unsourced
value is reported back to the model as a warning and a failed step (agent/tools.py); it is not blocked.

WHY: in the first live eval the model left the portal page (which showed "Issued 01 Oct 2026"), the page was
compressed out of its context, and when the ERP form asked for an invoice date it typed today's date. The
fill succeeded, the read-back matched, the auditor only checked the fields the task named: a silent
hallucination that only the ground-truth grader caught.

Sources are the user's task and observations of the environment (pages, files). The agent's own outputs
(`remember`, `fill` read-backs, files it wrote) are deliberately NOT sources, otherwise a made-up value could
be laundered by remembering it first. Dates and amounts are compared in canonical form, so "31 Oct 2026"
sources "2026-10-31" and "$4,250.00" sources "4250.00".
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

_MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
MONTHS = {m: i for i, m in enumerate(_MONTH_NAMES, 1)}
_MON = rf"({'|'.join(_MONTH_NAMES)})[a-z]*\.?"
_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    (re.compile(rf"\b(\d{{1,2}})\s+{_MON},?\s+(\d{{4}})\b", re.I), "dMy"),
    (re.compile(rf"\b{_MON}\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I), "Mdy"),
    (re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b"), "ambiguous"),  # 01.10.2026 or 10/01/2026
]
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:[,.' ]\d{3})+|\d+)(?:[.,]\d{1,2})?(?![\w])")
PROSE_WORDS = 6  # longer values are free text (notes), which legitimately paraphrase


def _iso(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def dates_in(text: str) -> set[str]:
    out: set[str] = set()
    for rx, kind in _DATE_PATTERNS:
        for g in rx.findall(text):
            if kind == "ymd":
                cands = [_iso(int(g[0]), int(g[1]), int(g[2]))]
            elif kind == "dMy":
                cands = [_iso(int(g[2]), MONTHS[g[1][:3].lower()], int(g[0]))]
            elif kind == "Mdy":
                cands = [_iso(int(g[2]), MONTHS[g[0][:3].lower()], int(g[1]))]
            else:  # day-first (EU) and month-first (US) are both plausible readings
                cands = [_iso(int(g[2]), int(g[1]), int(g[0])), _iso(int(g[2]), int(g[0]), int(g[1]))]
            out.update(c for c in cands if c)
    return out


def _amount(token: str) -> str | None:
    t = token.replace("'", "").replace(" ", "")
    m = re.search(r"[.,](\d{1,2})$", t)  # a final 1-2 digit group is decimals; other separators group thousands
    whole, frac = (t[: m.start()], m.group(1)) if m else (t, "0")
    t = re.sub(r"[.,]", "", whole) + "." + frac
    try:
        return f"{Decimal(t).normalize():f}"  # fixed-point: normalize() alone yields "4.25E+3"
    except InvalidOperation:
        return None


def amounts_in(text: str) -> set[str]:
    return {a for a in (_amount(m) for m in _NUMBER.findall(text)) if a is not None}


class Provenance:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.dates: set[str] = set()
        self.amounts: set[str] = set()

    def observe(self, text: str) -> None:
        self.text.append(text.lower())
        self.dates |= dates_in(text)
        self.amounts |= amounts_in(text)

    def is_sourced(self, value: str) -> bool:
        v = value.strip()
        if not v or len(v.split()) > PROSE_WORDS:
            return True
        if d := dates_in(v):
            return bool(d & self.dates)
        if re.fullmatch(r"[$€£]?\s*[-+]?[\d.,' ]+\s*(USD|EUR|GBP)?", v, re.I):
            a = amounts_in(v)
            return bool(a & self.amounts)
        return any(v.lower() in t for t in self.text)
