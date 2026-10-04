"""All configuration in one place, read from the environment (and .env), validated up front.

WHY fail fast: a typo like AUTOWORK_MAX_STEPS=4o should stop the program at startup with a clear
message, not surface 20 minutes into a run as an obscure TypeError.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from .world import DEFAULT_WORLD_FILE, World, WorldError, default_world, load_world

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

WORKSPACE = Path(os.environ.get("AUTOWORK_WORKSPACE", ROOT / "workspace"))
WORKSPACE_SEED = ROOT / "workspace_seed"
RUNS_DIR = ROOT / "runs"
PLAYBOOK_PATH = ROOT / "data" / "playbook.json"
WORLD_URL = os.environ.get("WORLD_URL", "http://localhost:8001")

MODES = ("autonomous", "balanced", "supervised")
CONTEXT_SCHEMES = ("classic", "digest")  # see agent/context.py


class SettingsError(ValueError):
    pass


# Extra OpenAI-compatible providers, selected per model with a "provider:" prefix (e.g. gemini:gemini-3.8-flash).
# A colon, because provider model names already contain slashes (qwen/qwen3.8-27b). Unprefixed models use
# LLM_BASE_URL / LLM_API_KEY. Each provider has its own key and its own free-tier quota.
PROVIDERS: dict[str, tuple[str, str]] = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),  # free models end in ":free"
}


def provider_of(model: str, base_url: str) -> str:
    """Short provider name, for key labels: the model's prefix, else the base URL's host ("groq")."""
    prefix, sep, _ = model.partition(":")
    if sep and prefix in PROVIDERS:
        return prefix
    host = urlparse(base_url).hostname or "custom"
    return next((name for name in ("groq", "openai", "together") if name in host), host)


def api_model_of(model: str) -> str:
    """The model id the provider expects: without our "provider:" prefix."""
    prefix, sep, rest = model.partition(":")
    return rest if sep and prefix in PROVIDERS else model


@dataclass(frozen=True)
class Fallback:
    """A model to switch to when the ones before it are out of quota or unavailable (LLM_FALLBACK_MODELS)."""

    model: str
    base_url: str
    api_key: str
    key_label: str = ""  # e.g. "groq#2": which key this is, for the quota ledger (never the key itself)


@dataclass(frozen=True)
class Settings:
    llm_api_key: str  # empty is allowed at load time; LLMClient refuses to start without one
    llm_base_url: str
    llm_model: str
    llm_timeout_s: float
    llm_max_output_tokens: int
    llm_reasoning_effort: str  # "" = provider default (and the parameter is not sent at all)
    mode: str
    max_steps: int
    max_tokens_total: int
    max_active_seconds: float
    llm_fallbacks: tuple[Fallback, ...] = ()  # tried in order once the primary model can't serve us
    llm_key_label: str = ""  # label of the primary key, e.g. "groq#1"
    max_run_tokens: int = 0  # AUTOWORK_MAX_RUN_TOKENS: budget mode (0 = off); when set it is the per-run token cap
    context_scheme: str = "classic"  # how past turns are put in the prompt: classic | digest (agent/context.py)
    world: World = field(default_factory=default_world)  # apps, start page, high-risk actions (config/world.json)
    no_progress_steps: int = 8  # AUTOWORK_NO_PROGRESS_STEPS: stop after this many steps without progress (0 = never)
    verifier_model: str = ""  # AUTOWORK_VERIFIER_MODEL: the auditor's model; empty = the worker's model

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
        model = e.get("LLM_MODEL", "qwen/qwen3.8-27b").strip()  # see docs/decisions/0006 for why not gpt-oss-120b
        if not model:
            errors.append("LLM_MODEL must not be empty")
        effort = e.get("LLM_REASONING_EFFORT", "").strip().lower()
        if effort not in ("", "low", "medium", "high"):
            errors.append(f"LLM_REASONING_EFFORT={effort!r} must be low, medium or high (or unset)")
        context = e.get("AUTOWORK_CONTEXT", "classic").strip().lower()
        if context not in CONTEXT_SCHEMES:
            errors.append(f"AUTOWORK_CONTEXT={context!r} must be one of {CONTEXT_SCHEMES}")
        world_file = Path(e.get("AUTOWORK_WORLD_FILE") or DEFAULT_WORLD_FILE)
        try:
            world = load_world(world_file)
        except WorldError as err:
            errors.append(f"AUTOWORK_WORLD_FILE={str(world_file)!r}: {err}")
            world = default_world()
        mode = e.get("AUTOWORK_MODE", "balanced")
        if mode not in MODES:
            errors.append(f"AUTOWORK_MODE={mode!r} must be one of {MODES}")
        api_key = e.get("LLM_API_KEY") or e.get("GROQ_API_KEY") or ""  # GROQ_API_KEY: fallback for the default
        default_url, default_key = base_url, api_key

        def resolve(m: str, var: str) -> tuple[str, list[str]]:
            """Base URL and key(s) for model m. A key variable may hold several comma-separated keys (key
            rotation): each has its own quota, and the client moves to the next when one is exhausted."""
            prefix, sep, rest = m.partition(":")
            if sep and prefix in PROVIDERS:
                url, key_var = PROVIDERS[prefix]
                if not e.get(key_var):
                    errors.append(f"{var}={m!r} needs {key_var} in .env")
                raw = e.get(key_var, "")
            else:
                url, raw = default_url, default_key
            return url, [k.strip() for k in raw.split(",") if k.strip()] or [""]

        def with_rotation(m: str, var: str) -> tuple[Fallback, ...]:
            url, keys = resolve(m, var)
            prov = provider_of(m, url)
            return tuple(Fallback(m, url, k, f"{prov}#{i}") for i, k in enumerate(keys, 1))

        primary, *rotation = with_rotation(model, "LLM_MODEL")
        base_url, api_key = primary.base_url, primary.api_key
        fallbacks = tuple(rotation) + tuple(
            fb
            for m in (x.strip() for x in e.get("LLM_FALLBACK_MODELS", "").split(","))
            if m and m != model
            for fb in with_rotation(m, "LLM_FALLBACK_MODELS")
        )
        verifier_model = e.get("AUTOWORK_VERIFIER_MODEL", "").strip()
        if verifier_model:
            resolve(verifier_model, "AUTOWORK_VERIFIER_MODEL")  # fail fast on a missing provider key
        max_run = num("AUTOWORK_MAX_RUN_TOKENS", "0", int, 0, 10_000_000)
        s = cls(
            llm_key_label=primary.key_label,
            max_run_tokens=max_run,
            llm_api_key=api_key,
            llm_base_url=base_url,
            llm_model=model,
            llm_timeout_s=num("LLM_TIMEOUT_S", "90", float, 1, 600),
            llm_max_output_tokens=num("LLM_MAX_OUTPUT_TOKENS", "4096", int, 256, 65_536),
            llm_reasoning_effort=effort,
            mode=mode,
            max_steps=num("AUTOWORK_MAX_STEPS", "40", int, 1, 200),
            # AUTOWORK_MAX_RUN_TOKENS, when set, is the cap (budget mode); AUTOWORK_MAX_TOKENS is the older name
            max_tokens_total=max_run or num("AUTOWORK_MAX_TOKENS", "400000", int, 1000, 10_000_000),
            max_active_seconds=num("AUTOWORK_MAX_ACTIVE_SECONDS", "1800", float, 10, 86_400),
            llm_fallbacks=fallbacks,
            context_scheme=context,
            world=world,
            verifier_model=verifier_model,
            no_progress_steps=num("AUTOWORK_NO_PROGRESS_STEPS", "8", int, 0, 200),
        )
        if errors:
            raise SettingsError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        return s

    def for_model(self, model: str, env: Mapping[str, str] | None = None) -> Settings:
        """These settings with another model, resolving its provider prefix (key and base URL) the same way.
        No model fallbacks: the eval harness grades each model separately, so a run must not silently switch
        model. Key rotation (more keys for the SAME model) is kept."""
        e = dict(os.environ if env is None else env)
        e["LLM_MODEL"] = model
        e.pop("LLM_FALLBACK_MODELS", None)
        e["AUTOWORK_VERIFIER_MODEL"] = self.verifier_model
        return dataclasses.replace(
            Settings.from_env(e),
            **{
                k: getattr(self, k)
                for k in (
                    "mode",
                    "max_steps",
                    "max_tokens_total",
                    "max_active_seconds",
                    "context_scheme",
                    "no_progress_steps",
                    "max_run_tokens",
                )
            },
        )

    def verifier_settings(self, env: Mapping[str, str] | None = None) -> Settings | None:
        """Settings for the auditor's own LLM client, or None when it shares the worker's model."""
        if not self.verifier_model or self.verifier_model == self.llm_model:
            return None
        return self.for_model(self.verifier_model, env)

    @property
    def api_model(self) -> str:
        """The model id the provider expects: without our "provider:" prefix."""
        return api_model_of(self.llm_model)


def reset_workspace() -> None:
    """Restore the shared workspace to its seed contents (outputs from earlier runs are removed)."""
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(WORKSPACE_SEED, WORKSPACE)


def admin_headers() -> dict[str, str]:
    """Header for simworld's /admin endpoints (eval harness and world reset only, never the agent)."""
    return {"X-Admin-Token": os.environ.get("SIMWORLD_ADMIN_TOKEN", "")}


def save_world_snapshot(run_dir: Path) -> Path | None:
    """Copy the simulated world's database into the run directory (harness only: uses the admin token).
    Returns the file, or None when the world is not reachable (e.g. a run against real applications)."""
    import httpx  # local: the agent itself never talks to the admin API

    try:
        r = httpx.get(f"{WORLD_URL}/admin/snapshot", headers=admin_headers(), timeout=10)
        r.raise_for_status()
    except httpx.HTTPError:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "world.db"
    out.write_bytes(r.content)
    return out


def restore_world_snapshot(path: Path) -> None:
    import httpx

    r = httpx.post(f"{WORLD_URL}/admin/restore", content=path.read_bytes(), headers=admin_headers(), timeout=10)
    r.raise_for_status()
