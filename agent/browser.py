"""Playwright browser wrapper that turns web pages into compact, LLM-readable observations.

Design: instead of screenshots-to-vision (slow, expensive, imprecise) the agent sees a
text snapshot: URL, HTTP status, alerts, a numbered list of interactive elements and a
trimmed text excerpt. It acts by element number. Screenshots are still captured on every
step, for humans and as evidence.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Error as PWError
from playwright.sync_api import sync_playwright

SNAPSHOT_JS = r"""
() => {
  document.querySelectorAll('[data-aw-id]').forEach(e => e.removeAttribute('data-aw-id'));
  const visible = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const clean = (t, n = 80) => (t || '').replace(/\s+/g, ' ').trim().slice(0, n);
  const labelFor = el => {
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    if (el.id) { const l = document.querySelector(`label[for="${el.id}"]`); if (l) return l.innerText; }
    const pl = el.closest('label'); if (pl) return pl.innerText;
    return el.getAttribute('placeholder') || el.getAttribute('name') || '';
  };
  const out = []; let n = 0;
  const sel = 'a[href], button, input:not([type=hidden]), select, textarea, [role=button], [onclick]';
  for (const el of document.querySelectorAll(sel)) {
    if (!visible(el)) continue;
    n += 1; el.setAttribute('data-aw-id', String(n));
    const tag = el.tagName.toLowerCase();
    const item = {id: n, tag};
    if (tag === 'a') { item.text = clean(el.innerText); item.href = el.getAttribute('href'); }
    else if (tag === 'button' || el.getAttribute('role') === 'button') { item.text = clean(el.innerText || el.value); }
    else if (tag === 'select') {
      item.label = clean(labelFor(el));
      item.value = el.options[el.selectedIndex] ? clean(el.options[el.selectedIndex].text) : '';
      item.options = Array.from(el.options).slice(0, 25).map(o => clean(o.text));
    } else {
      item.type = el.type || 'text'; item.label = clean(labelFor(el));
      item.value = item.type === 'password' ? (el.value ? '••••' : '') : clean(el.value);
      if (['submit','button'].includes(item.type)) item.text = clean(el.value);
    }
    out.push(item);
  }
  const alerts = Array.from(document.querySelectorAll('.error,.alert,[role=alert],.ok,.success'))
    .filter(visible).map(e => clean(e.innerText, 240)).filter(Boolean);
  return {elements: out, alerts, text: (document.body ? document.body.innerText : '').replace(/\n{3,}/g, '\n\n'),
          hasPassword: !!document.querySelector('input[type=password]')};
}
"""


@dataclass
class Snapshot:
    url: str
    title: str
    status: int | None
    elements: list[dict]
    alerts: list[str]
    text: str
    has_password: bool
    screenshot: str | None = None

    def fingerprint(self) -> str:
        return hashlib.md5((self.url + self.text[:3000]).encode()).hexdigest()[:10]

    def element(self, eid: int) -> dict | None:
        return next((e for e in self.elements if e["id"] == eid), None)

    def render(self, text_chars: int = 1800) -> str:
        lines = [f"URL: {self.url}", f"TITLE: {self.title}"]
        if self.status and self.status >= 400:
            lines.append(f"HTTP STATUS: {self.status}  <-- the last request FAILED")
        if self.alerts:
            lines.append("ALERTS ON PAGE: " + " | ".join(self.alerts))
        lines.append("INTERACTIVE ELEMENTS:")
        for e in self.elements[:70]:
            lines.append("  " + describe(e))
        if len(self.elements) > 70:
            lines.append(f"  ... {len(self.elements) - 70} more elements (use browser_read)")
        body = self.text.strip()
        more = f"\n  ... [{len(body) - text_chars} more chars, use browser_read]" if len(body) > text_chars else ""
        lines.append("PAGE TEXT:\n" + body[:text_chars] + more)
        return "\n".join(lines)


def describe(e: dict) -> str:
    t = e["tag"]
    if t == "a":
        return f"[{e['id']}] link \"{e.get('text', '')}\" -> {e.get('href', '')}"
    if t == "select":
        return f"[{e['id']}] select \"{e.get('label', '')}\" = \"{e.get('value', '')}\" options={e.get('options')}"
    if t == "button" or e.get("type") in ("submit", "button"):
        return f"[{e['id']}] button \"{e.get('text', '')}\""
    return f"[{e['id']}] {t}[{e.get('type', 'text')}] \"{e.get('label', '')}\" value=\"{e.get('value', '')}\""


class BrowserError(Exception):
    pass


@dataclass
class BrowserSession:
    shots_dir: Path
    allowed_hosts: list[str] = field(default_factory=lambda: ["localhost:8001", "127.0.0.1:8001"])
    headless: bool = True
    read_only: bool = False
    last: Snapshot | None = None

    def start(self, shared_context=None):
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        self._own = shared_context is None
        if self._own:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=self.headless)
            self.context = self._browser.new_context(viewport={"width": 1280, "height": 860})
        else:
            self.context = shared_context
        self.page = self.context.new_page()
        self.page.set_default_timeout(10000)
        self._status: int | None = None
        self.page.on("response", self._on_response)
        if self.read_only:
            # Verifier guarantee enforced at the network layer, not by prompt: no state-changing requests.
            self.page.route("**/*", self._read_only_route)
        self._shot_n = 0
        self._blocked = None
        return self

    def _read_only_route(self, route):
        req = route.request
        if req.method not in ("GET", "HEAD") and not urlparse(req.url).path.endswith("/login"):
            self._blocked = f"{req.method} {urlparse(req.url).path}"
            route.abort("blockedbyclient")
        else:
            route.continue_()

    def _on_response(self, resp):
        if resp.request.is_navigation_request() and resp.frame == self.page.main_frame:
            self._status = resp.status

    def close(self):
        try:
            self.page.close()
            if self._own:
                self._browser.close()
                self._pw.stop()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ observation
    def snapshot(self, label: str = "") -> Snapshot:
        try:
            self.page.wait_for_load_state("load", timeout=8000)
        except PWError:
            pass
        data = self.page.evaluate(SNAPSHOT_JS)
        self._shot_n += 1
        shot = self.shots_dir / f"{self._shot_n:03d}{'-' + label if label else ''}.png"
        try:
            self.page.screenshot(path=str(shot), full_page=True)
        except PWError:
            shot = None
        self.last = Snapshot(self.page.url, self.page.title(), self._status, data["elements"], data["alerts"],
                             data["text"], data["hasPassword"], str(shot) if shot else None)
        return self.last

    # ------------------------------------------------------------------ actions
    def _check_url(self, url: str):
        p = urlparse(url)
        if p.scheme not in ("http", "https") or p.netloc not in self.allowed_hosts:
            raise BrowserError(f"URL {url!r} is outside the allowed hosts {self.allowed_hosts}")
        if p.path.startswith("/admin"):
            raise BrowserError("Access to /admin is forbidden for the agent")

    def goto(self, url: str) -> Snapshot:
        if url.startswith("/"):
            url = "http://localhost:8001" + url
        self._check_url(url)
        self._status = None
        try:
            self.page.goto(url, wait_until="load")
        except PWError as e:
            raise BrowserError(f"Navigation failed: {_short(e)}") from e
        return self.snapshot("goto")

    def _locate(self, eid: int):
        if self.last is None:
            raise BrowserError("No page loaded yet; call browser_goto first")
        loc = self.page.locator(f'[data-aw-id="{int(eid)}"]')
        if loc.count() == 0:
            raise BrowserError(f"Element [{eid}] does not exist on the current page (the page may have changed). "
                               "Use element ids from the latest observation.")
        return loc.first

    def click(self, eid: int) -> Snapshot:
        loc = self._locate(eid)
        el = self.last.element(int(eid)) if self.last else None
        if el and el.get("tag") == "a" and el.get("href"):
            href = el["href"]
            if href.startswith("http"):
                self._check_url(href)
        self._status = None
        for attempt in (1, 2):
            try:
                loc.click(timeout=5000)
                break
            except PWError as e:
                if attempt == 2:
                    raise BrowserError(f"Click on [{eid}] failed: {_short(e)}") from e
                self.page.wait_for_timeout(500)  # transient (overlay, animation) -> one automatic retry
        try:
            self.page.wait_for_load_state("load", timeout=8000)
        except PWError:
            pass
        self.page.wait_for_timeout(250)
        if self._blocked:
            blocked, self._blocked = self._blocked, None
            if self.page.url.startswith("chrome-error"):
                self.page.go_back()
            raise BrowserError(f"Read-only mode: the state-changing request {blocked} was blocked")
        return self.snapshot("click")

    def fill(self, fields: list[dict]) -> str:
        """Fill several inputs/selects; reads every value back so the agent can verify its own input."""
        report = []
        for f in fields:
            eid, value = f.get("element_id"), str(f.get("value", ""))
            loc = self._locate(eid)
            tag = loc.evaluate("e => e.tagName.toLowerCase()")
            try:
                if tag == "select":
                    try:
                        loc.select_option(label=value, timeout=3000)
                    except PWError:
                        opts = loc.evaluate("e => Array.from(e.options).map(o => o.text)")
                        match = [o for o in opts if value.lower() in o.lower()]
                        if len(match) != 1:
                            raise BrowserError(f"[{eid}] has no unique option matching {value!r}; options: {opts}")
                        loc.select_option(label=match[0])
                    actual = loc.evaluate("e => e.options[e.selectedIndex].text")
                else:
                    loc.fill(value, timeout=3000)
                    actual = loc.input_value()
            except PWError as e:
                raise BrowserError(f"Could not fill [{eid}]: {_short(e)}") from e
            ok = actual.strip() == value.strip() or (tag == "select" and value.lower() in actual.lower())
            shown = "••••" if (self.last and (self.last.element(int(eid)) or {}).get("type") == "password") else actual
            report.append(f"[{eid}] now = {shown!r}" + ("" if ok else f"  <-- MISMATCH, wanted {value!r}"))
        return "Filled fields (read back from page):\n" + "\n".join(report)

    def back(self) -> Snapshot:
        self.page.go_back()
        return self.snapshot("back")

    def read(self, offset: int = 0, chars: int = 6000) -> str:
        if self.last is None:
            raise BrowserError("No page loaded yet")
        text = self.page.evaluate("() => document.body.innerText")
        chunk = text[offset:offset + chars]
        tail = f"\n... [{len(text) - offset - chars} more chars; call again with offset={offset + chars}]" \
            if len(text) > offset + chars else ""
        return f"URL: {self.page.url}\nFULL PAGE TEXT (from char {offset}):\n{chunk}{tail}"


def _short(e: Exception) -> str:
    return re.sub(r"\s+", " ", str(e)).split("Call log")[0][:300]
