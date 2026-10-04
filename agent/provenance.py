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

Labels: a value can be sourced and still be the wrong one. In a live eval the model typed an invoice's ISSUE date
into the DUE date field; the date was on the page, so the check above passed. So each observed date/amount also
keeps the label written next to it ("Issued: 2026-09-30", "Payment due  31 Oct 2026"), and `label_conflict` reports
when a field's name matches the label of a DIFFERENT value of the same kind but not the label of the value typed.
Generic word overlap, no domain vocabulary: "Due date" ~ "Payment due", and "Due date" !~ "Issued".
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


_AMOUNT_LIKE = re.compile(r"[$€£]?\s*(USD|EUR|GBP)?\s*[-+]?[\d.,' ]+\s*(USD|EUR|GBP)?", re.I)


def values_match(a: str, b: str) -> bool:
    """Do two renderings name the same value? A backstop for the auditor's own judgement: dates and amounts
    compare canonically ("30 Sep 2026" == "2026-09-30", "USD 640.00" == "640"), identifiers (anything with a
    digit) ignoring case and punctuation ("IN-7002" in "Invoice IN-7002"), other text when one contains the other
    ("Initech" in "Initech LLC"). Not names by similarity: a directory reference (a vendor) is checked by ID."""
    a, b = a.strip(), b.strip()
    da, db = dates_in(a), dates_in(b)
    if da or db:
        return bool(da & db)
    if _AMOUNT_LIKE.fullmatch(a) and _AMOUNT_LIKE.fullmatch(b):
        return bool(amounts_in(a) & amounts_in(b))
    if re.search(r"\d", a + b):
        x, y = re.sub(r"[^a-z0-9]", "", a.lower()), re.sub(r"[^a-z0-9]", "", b.lower())
    else:
        x, y = " ".join(a.lower().split()), " ".join(b.lower().split())
    return bool(x and y) and (x in y or y in x)


_LABEL_STOP = frozenset({"date", "the", "and", "for", "number", "value", "yyyy"})
_LABEL_WORDS = 4  # a label is at most this many words right before its value


def label_words(label: str) -> frozenset[str]:
    """Content words of a label: lowercase, letters only, no format hints in parentheses, no generic words."""
    label = re.sub(r"\([^)]*\)", " ", label)
    return frozenset(w for w in re.findall(r"[a-z]{3,}", label.lower()) if w not in _LABEL_STOP)


def _related(a: frozenset[str], b: frozenset[str]) -> bool:
    """Any word in common, allowing inflection: a shared prefix of 4+ letters ("issued" ~ "issue")."""
    return any(x == y or (len(x) >= 4 and len(y) >= 4 and x[:4] == y[:4]) for x in a for y in b)


def _values_in_line(line: str) -> list[tuple[int, int, str, frozenset[str]]]:
    """(start, end, kind, canonical values) for each date and amount in a line, in order; numbers inside a date
    are not amounts."""
    out = []
    for rx, _ in _DATE_PATTERNS:
        for m in rx.finditer(line):
            if d := dates_in(m.group(0)):
                out.append((m.start(), m.end(), "date", frozenset(d)))
    for m in _NUMBER.finditer(line):
        if any(s <= m.start() < e for s, e, _, _ in out):
            continue
        if a := _amount(m.group(0)):
            out.append((m.start(), m.end(), "amount", frozenset({a})))
    return sorted(out)


def labelled_values(text: str) -> list[tuple[str, frozenset[str], frozenset[str], str]]:
    """(kind, canonical values, label words, label) for every date/amount with a label right before it on its
    line ("Due: 2026-10-30"), or on the line above when the value starts its line (a label above its value)."""
    out = []
    prev = ""
    for line in text.splitlines():
        last_end = 0
        for start, end, kind, vals in _values_in_line(line):
            seg = line[last_end:start]
            if not seg.strip() and last_end == 0 and len(prev.split()) <= _LABEL_WORDS:
                seg = prev
            label = " ".join(re.findall(r"[A-Za-z][A-Za-z#]*", seg)[-_LABEL_WORDS:])
            if words := label_words(label):
                out.append((kind, vals, words, label))
            last_end = end
        if line.strip():
            prev = line.strip()
    return out


class Provenance:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.dates: set[str] = set()
        self.amounts: set[str] = set()
        self.labelled: list[tuple[str, frozenset[str], frozenset[str], str]] = []

    def observe(self, text: str) -> None:
        self.text.append(text.lower())
        self.dates |= dates_in(text)
        self.amounts |= amounts_in(text)
        self.labelled += labelled_values(text)

    def label_conflict(self, field_label: str, value: str) -> str | None:
        """Why `value` looks like the wrong value for a field named `field_label`, or None. Only when the source
        labels this value, those labels don't fit the field, and the field's name does fit the label of another
        value of the same kind (e.g. field "Due date", value labelled "Issued", another date labelled "Due")."""
        fw = label_words(field_label)
        if not fw or len(value.split()) > PROSE_WORDS:
            return None
        if d := dates_in(value):
            kind, vals = "date", d
        elif _AMOUNT_LIKE.fullmatch(value.strip()):
            kind, vals = "amount", amounts_in(value)
        else:
            return None
        own = [lab for k, v, w, lab in self.labelled if k == kind and v & vals]
        if not own or any(_related(fw, label_words(lab)) for lab in own):
            return None
        other = next((lab for k, v, w, lab in self.labelled if k == kind and not v & vals and _related(fw, w)), None)
        if other is None:
            return None
        return (
            f"{value!r} is labelled {', '.join(sorted(set(own)))!r} where you saw it, but the field "
            f"{field_label!r} matches a different value labelled {other!r}"
        )

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
