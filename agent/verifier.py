"""Independent verification of the worker's claim.

WHY a separate auditor: the worker saying "done" is not evidence. The auditor gets a fresh context
(none of the worker's reasoning, so it cannot inherit its mistakes), is told not to trust the claim,
and its browser is read-only AT THE NETWORK LAYER (see BrowserSession.read_only), so it can look but
cannot "fix" what it is checking.

WHY it writes its own checklist: in live runs (2026-10-04) the auditor passed two wrong results. It checked only
the planner's success criteria, which come from the same model as the worker (one plan was empty, one never named
the due date), and it was pointed at the pages the worker had looked at, so it inherited the worker's blind spots:
when the worker never opened the source email, neither did the auditor. Now:
  * a checklist is derived from the TASK TEXT alone (plus the list of apps), naming every field the task implies
    was written and which apps hold the authoritative values; the auditor checks it together with the planner's;
  * the worker's pointers are only where its writes landed (to find the record), never where it read its data;
  * code, not the model, decides whether a pass is acceptable (`enforce`): the auditor must have opened a source
    itself, in one of those apps, that is not the record, and compared every field with values that match each
    other and appear on that source under a fitting label (provenance.label_conflict). A pass that breaks these
    rules becomes a fail (wrong values) or inconclusive (no source / not checked), never a pass.

It shares the worker's browser *context* (cookies) so it doesn't have to log in again. Sharing cookies
does not weaken read-only: that is enforced per request, not per session.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .browser import ApprovalRequired, BrowserSession
from .context import tool_call
from .interfaces import LLM, Browser, Emit, Message
from .llm import LLMError, ProviderConfigError, QuotaExhausted, parse_json
from .provenance import Provenance, label_words, values_match
from .tools import VERDICT, VERIFIER_TOOLS, ToolBox, ToolResult, tool_args_preview
from .vault import Redactor, Vault
from .world import World, default_world

MAX_AUDIT_STEPS = 10


@dataclass
class Claim:
    task: str
    success_criteria: list[str]  # the planner's; checked together with the auditor's own checklist
    summary: str
    evidence: list[str]
    record_locations: list[str] = field(default_factory=list)  # where the worker's writes landed: to FIND the record


@dataclass
class Checklist:
    """What the auditor derives from the task text alone."""

    fields: list[str]  # every value the task implies the worker wrote: each needs both values compared
    conditions: list[str]  # other checkable conditions (no duplicate, a file exists, ...)
    source_apps: list[str]  # where the authoritative values live; empty when they are all in the task
    values_in_task: bool
    references: list[str] = field(default_factory=list)  # fields naming a directory record (vendor): matched by ID


Item = tuple[str, str, str]  # (id "C3", text, kind: "field" | "ref" | "cond")


def validate_checklist(raw: object, apps: list[str]) -> tuple[Checklist | None, str]:
    if not isinstance(raw, dict):
        return None, "expected a JSON object"
    fields_, conds, src = raw.get("fields", []), raw.get("conditions", []), raw.get("source_apps", [])
    refs = raw.get("references", [])
    for name, v in (("fields", fields_), ("conditions", conds), ("source_apps", src), ("references", refs)):
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
            return None, f"'{name}' must be a list of non-empty strings"
    if not fields_ and not conds:
        return None, "'fields' and 'conditions' are both empty: name at least one thing to check"
    unknown = [a for a in src if a not in apps]
    if unknown:
        return None, f"unknown app(s) in 'source_apps': {unknown}; choose from {apps}"
    in_task = raw.get("values_in_task") is True
    if not src and not in_task:
        return None, "'source_apps' is empty but 'values_in_task' is not true: where do the values come from?"
    return Checklist(fields_, conds, src, in_task, refs), ""


def _same_location(a: str, b: str) -> bool:
    return a.rstrip("/") == b.rstrip("/")


def appears(value: str, src: Provenance) -> bool:
    """Is `value` on the source? Dates and amounts in any format, other text as written (case-insensitive)."""
    return src.is_sourced(value)


_ELEMENT_LINE = re.compile(r"^\s*\[\d+\] ")


def _norm_id(x: object) -> str:
    return str(x or "").strip().lstrip("#").lower()


def check_reference(c: dict, source_text: str, seen: dict[str, list[str]]) -> tuple[str, bool] | None:
    """A reference field (e.g. the vendor of a bill) is checked by ID in the directory, never by name similarity
    (2026-10-04: an auditor accepted "Globex Corporation" for "Globex Receivables" by name, which would equally
    accept "Acme Logistics GmbH" for an Acme Supplies invoice). Needs: the directory page the auditor opened,
    the ID the record's entry has there, the ID the source maps to, and the key that maps it: a value shown on
    the source that matches exactly one directory row (an email address, an exact legal name). Returns
    (problem, is_wrong_value) or None when the reference checks out."""
    rec_id, src_id = _norm_id(c.get("record_id")), _norm_id(c.get("source_id"))
    key, directory = str(c.get("source_key") or "").strip(), str(c.get("directory") or "").strip()
    if not (rec_id and src_id and key and directory):
        return "needs record_id, source_id, source_key and directory (match by ID, not by name)", False
    loc = _find(seen, directory)
    if loc is None:
        return f"the directory {directory!r} was not opened by the auditor", False
    text = "\n".join(seen[loc])
    for x in (rec_id, src_id):
        if not re.search(rf"(?<![\w]){re.escape(x)}(?![\w])", text.lower()):
            return f"ID {x!r} does not appear in the directory {loc}", False
    if key.lower() not in source_text.lower():
        return f"the key {key!r} does not appear on the source", False
    rows = [ln for ln in text.splitlines() if key.lower() in ln.lower() and not _ELEMENT_LINE.match(ln)]
    if len(rows) != 1:
        return f"the key {key!r} matches {len(rows)} directory rows, not exactly one", False
    if rec_id != src_id:
        return f"the record belongs to ID {rec_id} but the source maps to ID {src_id}", True
    return None


NOT_IN_RECORD = "not in record"


def _field_shown(label: str, seen: dict[str, list[str]], source_loc: str) -> str | None:
    """A page the auditor opened (other than the source) whose text names this field, or None."""
    words = label_words(label)
    if not words:
        return None
    for loc, texts in seen.items():
        if loc == source_loc:
            continue
        page = " ".join(texts).lower()
        if all(re.search(rf"\b{re.escape(w)}", page) for w in words):
            return loc
    return None


def _find(seen: dict[str, list[str]], where: str) -> str | None:
    return next(
        (
            loc
            for loc in seen
            if where and (_same_location(loc, where) or (where.startswith("/") and loc.endswith(where)))
        ),
        None,
    )


def _infer_source(
    items: list[Item],
    checks: dict[str, dict],
    checklist: Checklist,
    seen: dict[str, list[str]],
    records: list[str],
    world: World,
) -> str:
    """The location the auditor opened, in a source app and not a record, on which every field's source value
    appears; "" when there is no such single location."""
    values = [str(checks.get(i, {}).get("source_value") or "") for i, _, kind in items if kind == "field"]
    fits = []
    for loc, texts in seen.items():
        if any(_same_location(loc, r) for r in records):
            continue
        if checklist.source_apps and world.app_of(loc) not in checklist.source_apps:
            continue
        p = Provenance()
        p.observe("\n".join(texts))
        if values and all(v and appears(v, p) for v in values):
            fits.append(loc)
    return fits[0] if len(fits) == 1 else ""


def enforce(
    v: dict,
    items: list[Item],
    checklist: Checklist,
    seen: dict[str, list[str]],
    records: list[str],
    world: World,
) -> dict:
    """Code-level rules for accepting a pass. `items` are (id, text, kind). `seen` maps each location the
    auditor opened to what it observed there. Returns the verdict, downgraded when a rule is broken: to a
    fail when the values disagree (the worker has something to fix), else to inconclusive."""
    if not v["passed"]:
        return v

    def downgrade(why: str, inconclusive: bool) -> dict:
        out = {**v, "passed": False, "reason": f"{why} (auditor said: {v['reason'][:300]})"}
        if inconclusive:
            out["inconclusive"] = True
        return out

    checks = {str(c.get("id", "")).strip().upper(): c for c in v.get("checks") or [] if isinstance(c, dict)}
    source = str(v.get("source") or "").strip()
    if not source:  # seen live: a complete audit that forgot the field. Infer it, then hold it to the same rules
        source = _infer_source(items, checks, checklist, seen, records, world)
    source_text, source_loc = "", ""
    if source.lower() == "task":
        if not checklist.values_in_task:
            return downgrade("pass rejected: the values are not all in the task, so 'task' is no source", True)
    else:
        loc = _find(seen, source)
        if loc is None:
            return downgrade(f"pass rejected: no source document opened by the auditor (source={source!r})", True)
        if any(_same_location(loc, r) for r in records):
            return downgrade(f"pass rejected: the source {loc} is the record being checked", True)
        app = world.app_of(loc)
        if checklist.source_apps and app not in checklist.source_apps:
            return downgrade(
                f"pass rejected: the source {loc} is in {app or 'no known app'}, but the task's values come "
                f"from {checklist.source_apps}; the auditor never opened the source",
                True,
            )
        source_text = "\n".join(seen[loc])
        source_loc = loc
    missing = [f"{i} ({t})" for i, t, _ in items if i not in checks]
    if missing:
        return downgrade(f"pass rejected: checklist items not checked: {', '.join(missing)}", True)
    failed = [f"{i} ({t})" for i, t, _ in items if checks[i].get("ok") is not True]
    if failed:
        return downgrade(f"pass rejected: checklist items the auditor itself marked not ok: {', '.join(failed)}", False)
    src = Provenance()
    src.observe(source_text)
    for i, t, kind in items:
        c = checks[i]
        if kind == "ref" or (kind == "field" and c.get("record_id") and c.get("source_id")):
            if problem := check_reference(c, source_text, seen):
                why, wrong = problem
                return downgrade(f"{i} ({t}): {why}" if wrong else f"pass rejected: {i} ({t}): {why}", not wrong)
            continue
        if kind != "field":  # conditions: the auditor may describe them in prose (seen live), so no value checks
            continue
        rec, srcv = str(c.get("record_value") or "").strip(), str(c.get("source_value") or "").strip()
        if rec.lower() == NOT_IN_RECORD:
            # Accepted only if no page the auditor opened besides the source shows such a field: it cannot be used
            # to skip a field the record has (seen live: "Description" asked of a bill that only has notes).
            if shown := _field_shown(t, seen, source_loc):
                return downgrade(f"pass rejected: {i} ({t}) said to be not in the record, but {shown} shows it", True)
            continue
        if not (rec and srcv):
            return downgrade(f"pass rejected: {i} ({t}) has no record_value/source_value comparison", True)
        if not values_match(rec, srcv):
            return downgrade(f"{i} ({t}): the record has {rec!r} but the source has {srcv!r}", False)
        if source_text and not appears(srcv, src):
            return downgrade(f"pass rejected: {i} ({t}): {srcv!r} does not appear on the source {source}", True)
        if source_text and (conflict := src.label_conflict(t, srcv)):
            return downgrade(f"{i} ({t}): wrong value: {conflict}", False)
    return v


class Verifier:
    def __init__(
        self,
        llm: LLM,
        emit: Emit,
        workspace: Path,
        shots_dir: Path,
        vault: Vault | None,
        redactor: Redactor,
        over_budget: Callable[[], str | None],
        rel: Callable[[str | None], str | None],
        world: World | None = None,
    ):
        self.llm, self.emit, self.workspace, self.shots_dir = llm, emit, workspace, shots_dir
        self.vault, self.redactor, self.over_budget, self.rel = vault, redactor, over_budget, rel
        self.world = world or default_world()

    def _inconclusive(self, reason: str) -> dict:
        v = {"passed": False, "inconclusive": True, "reason": reason, "evidence": []}
        self.emit("verify_result", v)
        return v

    def checklist(self, task: str) -> Checklist | None:
        """Derived from the task text and the list of apps only: no worker output, no planner output."""
        apps = self.world.app_names()
        msgs: list[Message] = [
            {"role": "system", "content": prompts.VERIFIER_CHECKLIST.replace("{apps}", self.world.describe_apps())},
            {"role": "user", "content": f"TASK:\n{task}"},
        ]
        for attempt in (1, 2):
            try:
                # the retry drops json_mode, as in Agent._plan (a provider's JSON mode can return an empty object)
                r = self.llm.chat(msgs, json_mode=attempt == 1, role="verifier_checklist", step=0)
                cl, problem = validate_checklist(parse_json(r.content), apps)
            except (QuotaExhausted, ProviderConfigError):
                raise
            except (LLMError, ValueError) as e:
                cl, problem, r = None, f"no usable JSON: {e}", None
            if cl:
                return cl
            self.emit("warning", {"message": f"Audit checklist attempt {attempt} invalid: {problem}"})
            if r is not None:
                msgs += [
                    {"role": "assistant", "content": r.content or ""},
                    {"role": "user", "content": f"Invalid: {problem}. Reply with the JSON object again."},
                ]
        return None

    def verify(self, claim: Claim, worker_browser: Browser) -> dict:
        """Returns {'passed', 'reason', 'evidence'} plus 'inconclusive' when no acceptable verdict was reached."""
        self.emit("verify_start", {"summary": claim.summary})
        checklist = self.checklist(claim.task)
        if checklist is None:
            return self._inconclusive("The auditor could not derive a checklist from the task")
        items = self._items(checklist, claim.success_criteria)
        self.emit(
            "verify_checklist",
            {"items": [f"{i} [{kind}]: {t}" for i, t, kind in items], "source_apps": checklist.source_apps},
        )
        login_paths = frozenset(self.vault.login_paths()) if self.vault else frozenset()
        vb = BrowserSession(
            self.shots_dir,
            allowlist=self.world.allowlist(),
            risk=self.world.risk,
            base_url=self.world.start_url,
            read_only=True,
            login_paths=login_paths,
        ).start(shared_context=worker_browser.context)
        tools = ToolBox(vb, self.workspace, vault=self.vault, redactor=self.redactor)
        msgs = self._opening(claim, items, checklist)
        try:
            final_retry = True
            step = 0
            while step < MAX_AUDIT_STEPS:
                step += 1
                if why := self.over_budget():
                    self.emit("warning", {"message": f"Verification stopped: {why}"})
                    break
                # WHY force on the last step: a weaker model that wanders (seen live: it read an unrelated CSV)
                # otherwise ends "inconclusive", and a correct run is reported as unverified.
                last = step == MAX_AUDIT_STEPS
                turn = msgs + [
                    {
                        "role": "user",
                        "content": f"[audit step {step}/{MAX_AUDIT_STEPS}] "
                        + (
                            "This is your LAST step: call verdict now, based only on what you have seen. If you "
                            "could not confirm an item, that is passed=false."
                            if last
                            else "2 steps left: finish checking and call verdict."
                            if step == MAX_AUDIT_STEPS - 2
                            else "Check the next item, or call verdict once all are checked."
                        ),
                    }
                ]
                r = self.llm.chat(
                    turn,
                    tools=[VERDICT] if last else VERIFIER_TOOLS,
                    require_tool=True,
                    role="verifier",
                    step=step,
                )
                if not r.tool_calls:
                    self.emit(
                        "verify_step",
                        {"tool": "(no tool call)", "args": "", "ok": False, "text": (r.content or "")[:500]},
                    )
                    msgs.append({"role": "user", "content": "Call a tool (verdict when done)."})
                    if last and final_retry:  # seen live: nemotron answered the forced verdict with nothing, once
                        final_retry = False
                        step -= 1
                    continue
                c = r.tool_calls[0]
                if c.name == "verdict":
                    v = {
                        "passed": c.arguments.get("passed") is True,  # anything but an explicit true fails
                        "reason": c.arguments.get("reason", ""),
                        "evidence": c.arguments.get("evidence", []),
                        "source": c.arguments.get("source", ""),
                        "checks": c.arguments.get("checks", []),
                    }
                    if last and v["passed"] and not v["evidence"]:
                        # A verdict forced by the step budget must not default towards passing: a pass needs
                        # cited evidence. Errs towards "unverified", which the honesty metric scores as safe.
                        v["passed"] = False
                        v["reason"] = f"forced verdict claimed a pass without evidence ({v['reason']})"
                    v = enforce(v, items, checklist, tools.seen, claim.record_locations, self.world)
                    self.emit("verify_result", v)
                    return v
                try:
                    res = tools.run(c.name, c.arguments, step)
                except ApprovalRequired as e:  # cannot happen in read-only mode; handled for completeness
                    res = ToolResult(f"BLOCKED: {e}", "blocked", ok=False)
                self.emit(
                    "verify_step",
                    {
                        "tool": c.name,
                        "args": tool_args_preview(c.arguments),
                        "ok": res.ok,
                        "text": res.text[:1500],
                        "screenshot": self.rel(res.screenshot),
                    },
                )
                msgs.append(
                    {
                        "role": "assistant",
                        "content": r.content or None,
                        "tool_calls": [tool_call(c.id, c.name, c.raw_arguments)],
                    }
                )
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": res.text[:5000]})
        finally:
            vb.close()
        return self._inconclusive("Auditor did not reach a verdict")

    @staticmethod
    def _items(checklist: Checklist, criteria: list[str]) -> list[Item]:
        """The union, numbered C1..Cn: the auditor's references and fields, its conditions, then the planner's
        criteria."""
        refs = {" ".join(r.lower().split()) for r in checklist.references}
        out: list[Item] = []
        seen: set[str] = set()
        for text, kind in (
            [(r, "ref") for r in checklist.references]
            + [(f, "ref" if " ".join(f.lower().split()) in refs else "field") for f in checklist.fields]
            + [(c, "cond") for c in checklist.conditions]
            + [(c, "cond") for c in criteria]
        ):
            key = " ".join(text.lower().split())
            if key not in seen:
                seen.add(key)
                out.append((f"C{len(out) + 1}", text, kind))
        return out

    def _opening(self, claim: Claim, items: list[Item], checklist: Checklist) -> list[Message]:
        tag = {"field": " [FIELD]", "ref": " [REFERENCE]", "cond": ""}
        lines = "\n".join(f"- {i}{tag[kind]}: {t}" for i, t, kind in items)
        where = (
            "every value is stated in the task itself (source='task' is allowed)"
            if checklist.values_in_task and not checklist.source_apps
            else f"apps {checklist.source_apps}" + (" (or the task itself)" if checklist.values_in_task else "")
        )
        records = "\n".join(f"- {u}" for u in claim.record_locations[-10:]) or "- (none: no form was submitted)"
        return [
            {"role": "system", "content": prompts.VERIFIER},
            {
                "role": "user",
                "content": f"USER TASK:\n{claim.task}\n\nSUCCESS CRITERIA (checklist, check EVERY item):\n{lines}\n\n"
                f"WORKER'S CLAIM:\n{claim.summary}\nEvidence claimed: {json.dumps(claim.evidence)}\n\n"
                f"RECORD LOCATIONS (pages the worker's form submissions landed on; use ONLY to find the record, "
                f"never as the source of the values):\n{records}\n\n"
                f"WHERE THE AUTHORITATIVE VALUES ARE: {where}. Find and open that source yourself.\n\n"
                f"Apps:\n{self.world.describe_apps()}\nStart page: {self.world.start_url}",
            },
        ]
