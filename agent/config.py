"""All configuration in one place, read from the environment (and .env), validated up front.

WHY fail fast: a typo like AUTOWORK_MAX_STEPS=4o should stop the program at startup with a clear
message, not surface 20 minutes into a run as an obscure TypeError.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

WORKSPACE = Path(os.environ.get("AUTOWORK_WORKSPACE", ROOT / "workspace"))
WORKSPACE_SEED = ROOT / "workspace_seed"
RUNS_DIR = ROOT / "runs"
PLAYBOOK_PATH = ROOT / "data" / "playbook.json"
WORLD_URL = os.environ.get("WORLD_URL", "http://localhost:8001")

MODES = ("autonomous", "balanced", "supervised")


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    llm_api_key: str  # empty is allowed at load time; LLMClient refuses to start without one
    llm_base_url: str
    llm_model: str
    llm_timeout_s: float
    mode: str
    max_steps: int
    max_tokens_total: int
    max_active_seconds: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Parse and validate. Collects ALL problems before raising, so one restart fixes them all."""
        e = os.environ if env is None else env
        errors: list[str] = []

        def num(name: str, default: str, cast: type, lo: float, hi: float):
            raw = e.get(name, default)
            try:
                v = cast(raw)
            except ValueError:
                errors.append(f"{name}={raw!r} is not a valid {cast.__name__}")
                return cast(default)
            if not lo <= v <= hi:
                errors.append(f"{name}={v} must be between {lo} and {hi}")
            return v

        base_url = e.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        if urlparse(base_url).scheme not in ("http", "https") or not urlparse(base_url).netloc:
            errors.append(f"LLM_BASE_URL={base_url!r} must be an http(s) URL")
        model = e.get("LLM_MODEL", "openai/gpt-oss-120b").strip()
        if not model:
            errors.append("LLM_MODEL must not be empty")
        mode = e.get("AUTOWORK_MODE", "balanced")
        if mode not in MODES:
            errors.append(f"AUTOWORK_MODE={mode!r} must be one of {MODES}")
        s = cls(
            # LLM_API_KEY is provider-neutral; GROQ_API_KEY kept as a fallback for the default provider
            llm_api_key=e.get("LLM_API_KEY") or e.get("GROQ_API_KEY") or "",
            llm_base_url=base_url,
            llm_model=model,
            llm_timeout_s=num("LLM_TIMEOUT_S", "90", float, 1, 600),
            mode=mode,
            max_steps=num("AUTOWORK_MAX_STEPS", "40", int, 1, 200),
            max_tokens_total=num("AUTOWORK_MAX_TOKENS", "400000", int, 1000, 10_000_000),
            max_active_seconds=num("AUTOWORK_MAX_ACTIVE_SECONDS", "1800", float, 10, 86_400),
        )
        if errors:
            raise SettingsError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        return s


def reset_workspace() -> None:
    """Restore the shared workspace to its seed contents (outputs from earlier runs are removed)."""
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(WORKSPACE_SEED, WORKSPACE)


def admin_headers() -> dict[str, str]:
    """Header for simworld's /admin endpoints (eval harness and world reset only, never the agent)."""
    return {"X-Admin-Token": os.environ.get("SIMWORLD_ADMIN_TOKEN", "")}
