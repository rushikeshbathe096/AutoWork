"""Credential vault + redaction.

WHY: credentials used to live in a workspace file the model read, which put passwords into
prompts (sent to the LLM provider), into the event log and into report.json. Now the model only
ever names a site (`login(site="erp")`); the browser layer fills the secret itself, and every
string leaving the agent (events, reports, observations) passes through `Redactor`.

Source of secrets, first match wins:
  1. AUTOWORK_VAULT_FILE (path to a JSON file), else
  2. config/vault.json (git-ignored, for your own values), else
  3. config/vault.example.json, the demo credentials of the *simulated* apps, so a fresh
     clone works. A warning is logged when this fallback is used.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("autowork.vault")
ROOT = Path(__file__).resolve().parent.parent
REDACTED = "[REDACTED]"
_warned_demo = False


@dataclass(frozen=True)
class SiteCredential:
    site: str
    login_url: str
    username: str
    password: str


class Vault:
    def __init__(self, creds: dict[str, SiteCredential]):
        self._creds = creds

    @classmethod
    def load(cls) -> Vault:
        path = _vault_path()
        raw = json.loads(path.read_text())
        creds = {}
        for site, c in raw.items():
            missing = {"login_url", "username", "password"} - set(c)
            if missing:
                raise ValueError(f"vault entry {site!r} in {path} is missing {sorted(missing)}")
            creds[site] = SiteCredential(site, c["login_url"], c["username"], c["password"])
        return cls(creds)

    def sites(self) -> list[str]:
        return sorted(self._creds)

    def get(self, site: str) -> SiteCredential | None:
        return self._creds.get(site)

    def secret_values(self) -> list[str]:
        """Values that must never appear in logs or prompts. Usernames are included too: cheap and safer."""
        vals = {c.password for c in self._creds.values()} | {c.username for c in self._creds.values()}
        return sorted((v for v in vals if len(v) >= 4), key=len, reverse=True)

    def login_paths(self) -> set[str]:
        """Exact login form paths; the read-only verifier may POST only to these."""
        from urllib.parse import urlparse

        return {urlparse(c.login_url).path for c in self._creds.values()}


def _vault_path() -> Path:
    env = os.environ.get("AUTOWORK_VAULT_FILE")
    if env:
        return Path(env)
    own = ROOT / "config" / "vault.json"
    if own.exists():
        return own
    global _warned_demo
    if not _warned_demo:
        log.warning("Using demo credentials from config/vault.example.json (simulated apps only)")
        _warned_demo = True
    return ROOT / "config" / "vault.example.json"


class Redactor:
    """Replaces known secret values anywhere inside strings / nested dicts / lists."""

    def __init__(self, secrets: list[str]):
        self._secrets = [s for s in secrets if s]

    def text(self, s: str) -> str:
        for secret in self._secrets:
            if secret in s:
                s = s.replace(secret, REDACTED)
        return s

    def obj(self, o: Any) -> Any:
        if isinstance(o, str):
            return self.text(o)
        if isinstance(o, dict):
            return {k: self.obj(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.obj(v) for v in o]
        return o
