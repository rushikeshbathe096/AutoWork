"""Network-level policy, applied to EVERY request the browser makes (navigations, redirects,
form posts, fetch/XHR, images...) through Playwright request interception.

WHY at the network layer: checking only the URL the model passes to `browser_goto` was
bypassable in three ways, all reproduced before this fix:
  * percent-encoding:   /%61dmin/state   (the server decodes it to /admin/state)
  * open redirect:      /erp/login?next=/erp/../admin/state  -> 303 -> /admin/state
  * page JavaScript:    fetch('/admin/state')
Every request is normalised the same way the server would interpret it, then checked.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

DEFAULT_PORTS = {"http": 80, "https": 443}
LOOPBACK_ALIASES = {"localhost", "127.0.0.1", "::1", "[::1]"}

# Paths whose non-GET requests move money, destroy data or change payment details.
HIGH_RISK_PATH = re.compile(
    r"/(pay|payment|payments|delete|remove|transfer|wire|refund|approve|cancel)(/|$)"
    r"|bank|iban|payout",
    re.I,
)


@dataclass(frozen=True)
class NormalizedURL:
    scheme: str
    host: str  # lowercase; loopback aliases collapsed to "localhost"
    port: int
    path: str  # fully percent-decoded, dot-segments resolved, slashes collapsed


def normalize(url: str) -> NormalizedURL | None:
    """Returns None for URLs that cannot be interpreted safely (non-http(s), garbage)."""
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme.lower()
        if scheme not in DEFAULT_PORTS:
            return None
        host = (parts.hostname or "").lower().rstrip(".")
        if not host:
            return None
        port = parts.port or DEFAULT_PORTS[scheme]
    except ValueError:
        return None
    if host in LOOPBACK_ALIASES:
        host = "localhost"
    path = parts.path or "/"
    for _ in range(5):  # decode repeatedly: %2561 -> %61 -> a
        decoded = unquote(path)
        if decoded == path:
            break
        path = decoded
    path = path.replace("\\", "/")
    path = re.sub(r"/{2,}", "/", path)
    path = posixpath.normpath(path)
    if not path.startswith("/"):
        path = "/" + path
    return NormalizedURL(scheme, host, port, path)


@dataclass(frozen=True)
class AllowList:
    origins: frozenset[tuple[str, str, int]] = frozenset({("http", "localhost", 8001)})
    blocked_prefixes: tuple[str, ...] = ("/admin",)

    def check(self, url: str) -> str | None:
        """Returns a human-readable reason if the URL is NOT allowed, else None."""
        n = normalize(url)
        if n is None:
            return f"scheme or URL not allowed: {url[:80]!r}"
        if (n.scheme, n.host, n.port) not in self.origins:
            return f"origin {n.scheme}://{n.host}:{n.port} is not in the allowlist"
        low = n.path.lower()
        for p in self.blocked_prefixes:
            if low == p or low.startswith(p + "/"):
                return f"path {n.path} is forbidden for the agent"
        return None


HIGH_RISK_FIELD = re.compile(r"(^|&)[^=&]*(bank|iban|swift|routing|account_number|payout)[^=&]*=", re.I)


def is_high_risk(method: str, url: str, body: str | None = None) -> bool:
    """Non-GET requests to payment / deletion endpoints, or that submit bank-detail fields.
    Independent of how the request was triggered (button, Enter key, page JavaScript)."""
    if method.upper() in ("GET", "HEAD", "OPTIONS"):
        return False
    n = normalize(url)
    if n is None or HIGH_RISK_PATH.search(n.path):
        return True
    return bool(body and HIGH_RISK_FIELD.search(unquote(body)))


def request_key(method: str, url: str) -> str:
    n = normalize(url)
    return f"{method.upper()} {n.path if n else url}"
