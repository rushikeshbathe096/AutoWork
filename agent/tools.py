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


EID = {"type": "integer", "description": "Element number from the latest observation, e.g. 7 for [7]"}

BROWSER_GOTO = fn(
    "browser_goto",
    "Open a URL in the browser. Returns an observation of the page.",
    {"url": {"type": "string"}},
    ["url"],
)
BROWSER_CLICK = fn(
    "browser_click",
    "Click a link or button by element number. Returns the resulting page.",
    {"element_id": EID},
    ["element_id"],
)
BROWSER_FILL = fn(
    "browser_fill",
    "Fill one or more form fields (text inputs, textareas, selects) on the current page in one go. "
    "For selects pass the visible option text. Values are read back so you can confirm them. "
    "Does NOT submit; click the submit button afterwards.",
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
    "Read the full text of the current page (observations only show an excerpt).",
    {"offset": {"type": "integer", "description": "Character offset, default 0"}},
    [],
)
BROWSER_BACK = fn("browser_back", "Go back to the previous page.", {}, [])
LOGIN = fn(
    "login",
    "Sign in to a site using the company credential vault. You never see the password; the browser fills it. "
    "Use this whenever you reach a login page.",
    {"site": {"type": "string", "description": "Vault site name, e.g. 'erp' or 'acme'"}},
    ["site"],
)
LIST_FILES = fn("list_files", "List files in the shared workspace folder.", {}, [])
READ_FILE = fn("read_file", "Read a text file from the workspace.", {"path": {"type": "string"}}, ["path"])
WRITE_FILE = fn(
    "write_file",
    "Write a text file into the workspace (e.g. a report).",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)
REMEMBER = fn(
    "remember",
    "Save an important fact to working memory (amounts, ids, dates, what you have completed). Old page "
    "observations are dropped from your context, memory is not. Use it for anything you will need later.",
    {"key": {"type": "string"}, "value": {"type": "string"}},
    ["key", "value"],
)
ASK_HUMAN = fn(
    "ask_human",
    "Ask the user a question and wait for the answer. Use ONLY when you cannot safely proceed: the request "
    "is genuinely ambiguous after checking the available systems, information is missing, or something "
    "looks suspicious. Do not ask for things you can look up yourself.",
    {
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}, "description": "Optional suggested answers"},
    },
    ["question"],
)
FINISH = fn(
    "finish",
    "End the task. Call this only after you have checked that the outcome is in place, e.g. by viewing the "
    "saved record. status: 'done' if the goal is achieved, 'failed' if it cannot be achieved, 'needs_user' "
    "if a human must act.",
    {
        "status": {"type": "string", "enum": ["done", "failed", "needs_user"]},
        "summary": {"type": "string", "description": "Concise answer/summary for the user, 1-4 sentences"},
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete facts proving the outcome (record ids, URLs, values seen)",
        },
    },
    ["status", "summary", "evidence"],
)
VERDICT = fn(
    "verdict",
    "Report your verification result.",
    {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    ["passed", "reason", "evidence"],
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


class ToolBox:
    def __init__(
        self,
        browser: Browser,
        workspace: Path,
        memory: WorkingMemory | None = None,
        vault: Vault | None = None,
        redactor: Redactor | None = None,
    ):
        self.browser = browser
        self.workspace = workspace
        self.memory = memory
        self.vault = vault
        self.redactor = redactor or Redactor([])

    def run(self, name: str, args: dict, step: int) -> ToolResult:
        if "__invalid_json__" in args:
            return ToolResult(
                f"ERROR: arguments were not valid JSON: {args['__invalid_json__'][:200]}", "invalid arguments", ok=False
            )
        impl = getattr(self, f"_t_{name}", None)
        if impl is None:
            return ToolResult(f"ERROR: unknown tool {name!r}", "unknown tool", ok=False)
        try:
            return self._redact(impl(step=step, **args))
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
        return self._obs(self.browser.click(element_id), f"clicked [{element_id}]")

    def _t_browser_fill(self, fields: list, step: int):
        out = self.browser.fill(fields)
        return ToolResult(out, out[:300], ok="MISMATCH" not in out)

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
