"""The environment the agent works in, loaded from config/world.json (or AUTOWORK_WORLD_FILE).

WHY a file and not code: the start page, which origins the browser may reach, and which actions count as high-risk
are facts about the deployment's applications, not about the agent. Keeping them here means a new application is
added by editing configuration (and declaring its risky actions), without touching agent code.
The lists are compiled into the same regular expressions the gates used before they were configurable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .netpolicy import AllowList, normalize

DEFAULT_WORLD_FILE = Path(__file__).resolve().parent.parent / "config" / "world.json"


class WorldError(ValueError):
    pass


@dataclass(frozen=True)
class RiskRules:
    button_label: re.Pattern[str]  # a button whose label matches needs approval (agent/policy.py)
    path: re.Pattern[str]  # a non-GET request whose normalized path matches needs approval (agent/netpolicy.py)
    form_field: re.Pattern[str]  # ...or whose form body has a matching field name


WORKSPACE_APP = "workspace"  # the shared files; sources there are recorded as "workspace file <path>"


@dataclass(frozen=True)
class App:
    name: str
    path: str  # URL path prefix, e.g. "/erp"
    about: str


@dataclass(frozen=True)
class World:
    start_url: str
    description: str  # one line for the planner, e.g. what the start page links to
    allowed_origins: frozenset[tuple[str, str, int]]
    blocked_path_prefixes: tuple[str, ...]
    risk: RiskRules
    apps: tuple[App, ...] = ()

    def allowlist(self) -> AllowList:
        return AllowList(self.allowed_origins, self.blocked_path_prefixes)

    def app_names(self) -> list[str]:
        return [a.name for a in self.apps] + [WORKSPACE_APP]

    def describe_apps(self) -> str:
        lines = [f"- {a.name}: {a.about} (URLs under {a.path})" for a in self.apps]
        return "\n".join(lines + [f"- {WORKSPACE_APP}: the shared workspace folder (files)"])

    def app_of(self, location: str) -> str | None:
        """Which app a location (URL, or "workspace file <path>") belongs to, by longest path prefix."""
        if location.startswith("workspace file "):
            return WORKSPACE_APP
        n = normalize(location) if "://" in location else None
        path = n.path if n else location.split("?")[0]
        best = None
        for a in self.apps:
            if (path == a.path or path.startswith(a.path.rstrip("/") + "/")) and (
                best is None or len(a.path) > len(best.path)
            ):
                best = a
        return best.name if best else None


def _words(raw: dict, key: str) -> str:
    items = raw.get(key)
    if not isinstance(items, list) or not items or not all(isinstance(w, str) and w.strip() for w in items):
        raise WorldError(f"high_risk.{key} must be a non-empty list of strings")
    return "|".join(re.escape(w.strip()) for w in items)


def parse_world(data: dict) -> World:
    try:
        start_url, description = data["start_url"], data["description"]
        origins_raw, risk_raw = data["allowed_origins"], data["high_risk"]
    except (KeyError, TypeError) as e:
        raise WorldError(f"missing key {e}") from None
    origins = set()
    for o in origins_raw:
        n = normalize(str(o))
        if n is None:
            raise WorldError(f"allowed_origins: {o!r} is not an http(s) URL")
        origins.add((n.scheme, n.host, n.port))
    start = normalize(str(start_url))
    if start is None or (start.scheme, start.host, start.port) not in origins:
        raise WorldError(f"start_url {start_url!r} must be an http(s) URL on one of allowed_origins")
    if not isinstance(risk_raw, dict):
        raise WorldError("high_risk must be an object")
    risk = RiskRules(
        button_label=re.compile(rf"\b({_words(risk_raw, 'button_words')})\b", re.I),
        path=re.compile(rf"/({_words(risk_raw, 'path_segments')})(/|$)|{_words(risk_raw, 'path_substrings')}", re.I),
        form_field=re.compile(rf"(^|&)[^=&]*({_words(risk_raw, 'field_substrings')})[^=&]*=", re.I),
    )
    apps = []
    for name, app in (data.get("apps") or {}).items():
        if name.startswith("_"):
            continue
        if not isinstance(app, dict) or not str(app.get("path", "")).startswith("/"):
            raise WorldError(f"apps.{name} needs a 'path' starting with '/'")
        apps.append(App(name, str(app["path"]), str(app.get("about", ""))))
    return World(
        start_url=str(start_url),
        description=str(description).replace("{start_url}", str(start_url)),
        allowed_origins=frozenset(origins),
        blocked_path_prefixes=tuple(str(p) for p in data.get("blocked_path_prefixes", [])),
        risk=risk,
        apps=tuple(apps),
    )


def load_world(path: Path) -> World:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise WorldError(f"cannot read {path}: {e}") from None
    return parse_world(data)


@lru_cache(maxsize=1)
def default_world() -> World:
    """The checked-in world. Used when a caller (mostly tests) doesn't pass one explicitly."""
    return load_world(DEFAULT_WORLD_FILE)
