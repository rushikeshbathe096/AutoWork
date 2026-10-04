"""Tool definitions (JSON schema, as sent to the model) and their implementations.

Tools are generic (browser, files, memory, human). Nothing here knows about invoices or the
ERP, which is what lets the same agent take on a different task unchanged.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

from .browser import ApprovalRequired, BrowserError, wrap_untrusted
from .interfaces import Browser
from .memory import WorkingMemory
from .provenance import Provenance
from .vault import Redactor, Vault

MAX_READ_BYTES = 200_000  # refuse to load bigger files at all
MAX_READ_CHARS = 8_000  # what the model sees
MAX_WRITE_CHARS = 100_000


def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


EID = {"type": "integer", "description": "[n] from the latest observation"}

BROWSER_GOTO = fn(
    "browser_goto",
    "Open a URL.",
    {"url": {"type": "string"}},
    ["url"],
)
BROWSER_CLICK = fn(
    "browser_click",
    "Click a link or button.",
    {"element_id": EID},
    ["element_id"],
)
BROWSER_FILL = fn(
    "browser_fill",
    "Fill form fields (selects: visible option text). Values are read back. Does NOT submit.",
    {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"element_id": EID, "value": {"type": "string"}},
                "required": ["element_id", "value"],
            },
        }
    },
    ["fields"],
)
BROWSER_READ = fn(
    "browser_read",
    "Full text of the current page (observations show an excerpt).",
    {"offset": {"type": "integer", "description": "start character"}},
    [],
)
BROWSER_BACK = fn("browser_back", "Go back.", {}, [])
LOGIN = fn(
    "login",
    "Sign in from the credential vault (password never shown). Use on any login page.",
    {"site": {"type": "string"}},
    ["site"],
)
LIST_FILES = fn("list_files", "List workspace files.", {}, [])
READ_FILE = fn("read_file", "Read a workspace file.", {"path": {"type": "string"}}, ["path"])
WRITE_FILE = fn(
    "write_file",
    "Write a workspace file (e.g. a report).",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)
REMEMBER = fn(
    "remember",
    "Save a fact to working memory (kept when old observations are dropped).",
    {"key": {"type": "string"}, "value": {"type": "string"}},
    ["key", "value"],
)
ASK_HUMAN = fn(
    "ask_human",
    "Ask the user and wait; only if you cannot safely proceed after investigating.",
    {
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
    },
    ["question"],
)
FINISH = fn(
    "finish",
    "End the task: done (achieved and checked), failed (impossible) or needs_user (a human must act).",
    {
        "status": {"type": "string", "enum": ["done", "failed", "needs_user"]},
        "summary": {"type": "string", "description": "1-4 sentences for the user"},
        "evidence": {"type": "array", "items": {"type": "string"}, "description": "Facts you saw (ids, values)"},
    },
    ["status", "summary", "evidence"],
)
VERDICT = fn(
    "verdict",
    "Report your verification result. A pass is accepted only with a source you opened yourself and one entry "
    "in `checks` per checklist item (C1, C2, ...), with both values for every FIELD item.",
    {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "source": {
            "type": "string",
            "description": "URL (or 'workspace file <path>') of the source document you opened and compared "
            "against, or 'task' if every value is stated in the task itself. Never the record being checked.",
        },
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "checklist id, e.g. C3"},
                    "ok": {"type": "boolean"},
                    "record_value": {"type": "string", "description": "value in the record, as shown there"},
                    "source_value": {"type": "string", "description": "value in the source, as shown there"},
                    "directory": {"type": "string", "description": "[REFERENCE] items: the directory page URL"},
                    "record_id": {"type": "string", "description": "[REFERENCE]: ID of the record's entry"},
                    "source_key": {"type": "string", "description": "[REFERENCE]: source value naming one entry"},
                    "source_id": {"type": "string", "description": "[REFERENCE]: ID of that entry"},
                },
                "required": ["id", "ok"],
            },
        },
    },
    ["passed", "reason", "evidence", "source", "checks"],
)

WORKER_TOOLS = [
    BROWSER_GOTO,
    BROWSER_CLICK,
    BROWSER_FILL,
    BROWSER_READ,
    BROWSER_BACK,
    LOGIN,
    LIST_FILES,
    READ_FILE,
    WRITE_FILE,
    REMEMBER,
    ASK_HUMAN,
    FINISH,
]
VERIFIER_TOOLS = [BROWSER_GOTO, BROWSER_CLICK, BROWSER_READ, BROWSER_BACK, LOGIN, LIST_FILES, READ_FILE, VERDICT]


@dataclass
class ToolResult:
    text: str  # full observation shown to the model
    short: str  # what the model sees once this observation is old (context compression)
    ok: bool = True
    screenshot: str | None = None


class WorkspaceError(Exception):
    pass


def confine(root: Path, rel: str) -> Path:
    """Resolve `rel` inside `root`, following symlinks, and refuse anything that lands outside.
    WHY realpath + is_relative_to: string checks miss `a/../../x` and symlinks pointing out."""
    if not rel or Path(rel).is_absolute():
        raise WorkspaceError(f"Path {rel!r} must be relative to the workspace")
    base = root.resolve()
    full = (base / rel).resolve()
    if not full.is_relative_to(base) or full == base:
        raise WorkspaceError(f"Path {rel!r} is outside the workspace")
    return full


# Tools whose output is an observation of the environment, i.e. a legitimate source of data values.
# Not remember / browser_fill / write_file: echoing the agent's own output would launder invented values.
SOURCE_TOOLS = {"browser_goto", "browser_click", "browser_read", "browser_back", "login", "read_file", "list_files"}


class ToolBox:
    def __init__(
        self,
        browser: Browser,
        workspace: Path,
        memory: WorkingMemory | None = None,
        vault: Vault | None = None,
        redactor: Redactor | None = None,
        provenance: Provenance | None = None,
    ):
        self.browser = browser
        self.provenance = provenance
        self.sources: list[str] = []  # pages and files observed, in order
        self.records: list[str] = []  # pages reached by submitting a form: where the worker's writes landed
        self.seen: dict[str, list[str]] = {}  # location -> observation texts (for the auditor's value checks)
        self.last_observation: str | None = None  # environment text the last call returned (None: not a reading)
        self.last_changed_state = False  # the last call submitted a form that landed, or wrote a file
        self.workspace = workspace
        self.memory = memory
        self.vault = vault
        self.redactor = redactor or Redactor([])

    def run(self, name: str, args: dict, step: int) -> ToolResult:
        if "__invalid_json__" in args:
            return ToolResult(
                f"ERROR: arguments were not valid JSON: {args['__invalid_json__'][:200]}", "invalid arguments", ok=False
            )
        self.last_observation, self.last_changed_state = None, False
        impl = getattr(self, f"_t_{name}", None)
        if impl is None:
            return ToolResult(f"ERROR: unknown tool {name!r}", "unknown tool", ok=False)
        try:
            records = len(self.records)
            result = self._redact(impl(step=step, **args))
            self.last_changed_state = len(self.records) > records or (name == "write_file" and result.ok)
            if name in SOURCE_TOOLS:
                self.last_observation = result.text
                if self.provenance:
                    self.provenance.observe(result.text)
                where = f"workspace file {args.get('path')}" if name == "read_file" else None
                if name.startswith("browser_") or name == "login":
                    where = self._current_url()
                if where and where not in self.sources:
                    self.sources.append(where)
                if where:
                    self.seen.setdefault(where, []).append(result.text)
            return result
        except ApprovalRequired:
            raise  # the agent loop handles this: it asks a human
        except WorkspaceError as e:
            return ToolResult(f"ERROR: {e}", f"ERROR: {e}", ok=False)
        except TypeError as e:
            return ToolResult(f"ERROR: bad arguments for {name}: {e}", "bad arguments", ok=False)
        except BrowserError as e:
            snap = ""
            with contextlib.suppress(Exception):  # the error report must not fail because the page is gone
                snap = "\n\nCURRENT PAGE:\n" + self.browser.snapshot("error").render(900)
            return self._redact(
                ToolResult(
                    f"ERROR: {e}{snap}",
                    f"ERROR: {str(e)[:150]}",
                    ok=False,
                    screenshot=self.browser.last.screenshot if self.browser.last else None,
                )
            )

    def _current_url(self) -> str | None:
        return getattr(getattr(self.browser, "last", None), "url", None)

    def _redact(self, r: ToolResult) -> ToolResult:
        """Defence in depth: even if a page echoes a secret back, it never reaches the prompt."""
        r.text, r.short = self.redactor.text(r.text), self.redactor.text(r.short)
        return r

    # -- browser
    def _obs(self, snap, verb: str) -> ToolResult:
        failed = bool(snap.status and snap.status >= 400)
        short = (
            f"[{verb} -> {snap.url} | {snap.title}"
            + (f" | HTTP {snap.status}" if failed else "")
            + (f" | alerts: {'; '.join(snap.alerts)[:160]}" if snap.alerts else "")
            + "] (old observation elided)"
        )
        return ToolResult(snap.render(), short, ok=not failed, screenshot=snap.screenshot)

    def _t_browser_goto(self, url: str, step: int):
        return self._obs(self.browser.goto(url), "opened")

    def _t_browser_click(self, element_id: int, step: int):
        before = getattr(self.browser, "last", None)
        el = before.element(element_id) if before else None
        res = self._obs(self.browser.click(element_id), f"clicked [{element_id}]")
        submitted = el is not None and (el.get("tag") == "button" or el.get("type") == "submit")
        url = self._current_url()
        if submitted and res.ok and url and before and url != before.url and url not in self.records:
            self.records.append(url)
        return res

    def _t_browser_fill(self, fields: list, step: int):
        before = getattr(self.browser, "last", None)
        out = self.browser.fill(fields)
        unsourced, conflicts = [], []
        for f in fields:
            if not (isinstance(f, dict) and self.provenance):
                continue
            value = str(f.get("value", ""))
            if not self.provenance.is_sourced(value):
                unsourced.append(f"[{f.get('element_id')}] {value!r}")
                continue
            el = before.element(int(f["element_id"])) if before and str(f.get("element_id", "")).isdigit() else None
            if el and (why := self.provenance.label_conflict(el.get("label", ""), value)):
                conflicts.append(f"[{f.get('element_id')}] {why}")
        if unsourced:
            out += (
                "\nUNSOURCED VALUES: " + ", ".join(unsourced) + " do not appear in the task or in anything you "
                "observed this run (dates and amounts compared in any format). Do not invent data: find the source "
                "and correct the field, or clear it if it is optional. Ask the human if the data does not exist."
            )
        if conflicts:
            out += (
                "\nLABEL CONFLICT: " + "; ".join(conflicts) + ". You may have copied the wrong value: re-read the "
                "source and use the value labelled like this field."
            )
        return ToolResult(out, out[:300], ok="MISMATCH" not in out and not unsourced and not conflicts)

    def _t_browser_read(self, step: int, offset: int = 0):
        out = self.browser.read(int(offset or 0))
        return ToolResult(out, f"[read full text of {self.browser.page.url}] (old observation elided)")

    def _t_browser_back(self, step: int):
        return self._obs(self.browser.back(), "went back")

    def _t_login(self, site: str, step: int):
        cred = self.vault.get(site) if self.vault else None
        if cred is None:
            sites = self.vault.sites() if self.vault else []
            return ToolResult(
                f"ERROR: no credentials for site {site!r}. Known sites: {sites}", "unknown site", ok=False
            )
        return self._obs(self.browser.login(cred), f"logged in to {site}")

    # -- files
    def _t_list_files(self, step: int):
        base = self.workspace.resolve()
        files = sorted(
            str(p.relative_to(self.workspace))
            for p in self.workspace.rglob("*")
            if p.is_file() and p.resolve().is_relative_to(base)
        )
        return ToolResult("Workspace files:\n" + "\n".join(files or ["(empty)"]), "listed workspace files")

    def _t_read_file(self, path: str, step: int):
        p = confine(self.workspace, path)
        if not p.is_file():
            return ToolResult(f"ERROR: {path} not found. Use list_files.", "file not found", ok=False)
        if p.stat().st_size > MAX_READ_BYTES:
            return ToolResult(f"ERROR: {path} is larger than {MAX_READ_BYTES} bytes", "file too large", ok=False)
        text = p.read_text(errors="replace")
        more = f"\n... [{len(text) - MAX_READ_CHARS} more chars not shown]" if len(text) > MAX_READ_CHARS else ""
        return ToolResult(
            wrap_untrusted("FILE", text[:MAX_READ_CHARS] + more), f"[read file {path}] (old observation elided)"
        )

    def _t_write_file(self, path: str, content: str, step: int):
        if len(content) > MAX_WRITE_CHARS:
            return ToolResult(f"ERROR: content exceeds {MAX_WRITE_CHARS} chars", "write too large", ok=False)
        p = confine(self.workspace, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return ToolResult(f"Wrote {len(content)} chars to {path}", f"wrote {path}")

    # -- memory
    def _t_remember(self, key: str, value: str, step: int):
        if self.memory is None:  # e.g. the verifier's toolbox: it has no working memory
            return ToolResult("ERROR: memory is not available here", "no memory", ok=False)
        r = self.memory.remember(key, value, step)
        return ToolResult(r, r)


def tool_args_preview(args: dict) -> str:
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) < 300 else s[:300] + "…"
