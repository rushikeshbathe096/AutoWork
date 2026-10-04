"""Budget mode for free tiers: a local ledger of each Groq key's daily token quota, and a pre-run check.

WHY a ledger: Groq's response headers report tokens per MINUTE and requests per day, not tokens per day (checked on
recorded qwen/qwen3.8-27b responses: x-ratelimit-limit-tokens: 8000). The daily figure only appears in a 429 body:
"tokens per day (TPD): Limit 200000, Used 198881, Requested 2564. Please try again in 10m24.24s". So the remaining
daily quota is estimated from our own usage logs (runs/*/llm_usage.jsonl), per key label (groq#1, groq#2, ...) and
model, and corrected from the provider's own numbers whenever a 429 reports them.

Model: a bucket of `limit` tokens that refills continuously at limit / 24 h (200k/day = ~8,333 tokens/hour, which
matches the 10m24s wait above for the ~1.4k tokens missing). Cached prompt tokens are not counted: Groq does not
count them towards rate limits.

Not visible to the ledger: usage of the same key from another machine, another checkout, or the Groq playground.
A 429 then corrects it on the next run. Two keys of the same Groq organization share one quota; the ledger treats
labels as separate, so give each label a key from a different account.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Fallback, Settings, provider_of

DEFAULT_TPD = 200_000  # Groq free tier, tokens per day per model (measured: every 429 body so far)
DAY_S = 86_400
TPD_RE = re.compile(r"tokens per day \(TPD\): Limit (\d+), Used (\d+)")
LEGACY_LABEL = "groq#1"  # usage rows written before key labels existed: there was one key


class QuotaShortage(Exception):
    """Not enough daily quota left on any key for one capped run. The message says when there will be."""


@dataclass(frozen=True)
class KeyQuota:
    label: str
    remaining: int
    limit: int

    def wait_s(self, need: int) -> float:
        return max(0, need - self.remaining) / (self.limit / DAY_S)


def _rows(runs_dir: Path):
    for log in runs_dir.glob("*/llm_usage.jsonl"):
        try:
            lines = log.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def remaining(runs_dir: Path | list[Path], label: str, model: str, now: float | None = None) -> KeyQuota:
    """Estimated tokens left today for this key label and model. Several runs directories (e.g. this checkout's
    and a frozen demo copy's) are read together."""
    events = []
    dirs = runs_dir if isinstance(runs_dir, list) else [runs_dir]
    for r in (row for d in dirs for row in _rows(d)):
        if r.get("provider") != "groq" or r.get("model") != model or (r.get("key") or LEGACY_LABEL) != label:
            continue
        t = float(r.get("timestamp_wall") or 0)
        if m := TPD_RE.search(r.get("error_body") or ""):
            events.append((t, "correct", int(m.group(2)), int(m.group(1))))
        used = (r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0) + (r.get("completion_tokens") or 0)
        if used > 0:
            events.append((t, "use", used, 0))
    limit = DEFAULT_TPD
    left, last = float(limit), None
    for t, kind, n, lim in sorted(events):
        if last is not None:
            left = min(limit, left + (t - last) * limit / DAY_S)
        if kind == "correct":
            limit, left = lim, float(lim - n)
        else:
            left -= n
        last = t
    now = time.time() if now is None else now
    if last is not None:
        left = min(limit, left + max(0.0, now - last) * limit / DAY_S)
    return KeyQuota(label, int(left), limit)


def _keys(s: Settings) -> list[Fallback]:
    """The primary key and the rotation keys for the same model, in order."""
    first = Fallback(s.llm_model, s.llm_base_url, s.llm_api_key, s.llm_key_label)
    return [first, *(f for f in s.llm_fallbacks if f.model == s.llm_model)]


def preflight(s: Settings, runs_dir: Path, now: float | None = None) -> Settings:
    """Before a run in budget mode on a Groq model: start on a key with at least the run's cap left, or raise
    QuotaShortage saying when one will have it. Other providers and runs without a cap pass through."""
    if not s.max_run_tokens or provider_of(s.llm_model, s.llm_base_url) != "groq":
        return s
    keys = _keys(s)
    quotas = [remaining(runs_dir, k.key_label or LEGACY_LABEL, s.llm_model, now) for k in keys]
    ok = [i for i, q in enumerate(quotas) if q.remaining >= s.max_run_tokens]
    if ok:
        i = ok[0]
        if i == 0:
            return s
        others = [k for j, k in enumerate(keys) if j != i]
        rest = tuple(others) + tuple(f for f in s.llm_fallbacks if f.model != s.llm_model)
        k = keys[i]
        return dataclasses.replace(s, llm_api_key=k.api_key, llm_key_label=k.key_label, llm_fallbacks=rest)
    best = min(quotas, key=lambda q: q.wait_s(s.max_run_tokens))
    wait = best.wait_s(s.max_run_tokens)
    at = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime((time.time() if now is None else now) + wait))
    left = ", ".join(f"{q.label}: ~{q.remaining:,} of {q.limit:,}" for q in quotas)
    raise QuotaShortage(
        f"Not starting {s.llm_model}: the run cap is {s.max_run_tokens:,} tokens and the daily quota left is "
        f"{left} (local ledger; usage from other machines is not visible to it). {best.label} should have enough "
        f"by {at} (in {wait / 3600:.1f} h at ~{best.limit // 24:,} tokens/hour). Start again then"
        " (evals: rerun the same command with --resume)."
    )


def main() -> None:
    """python -m agent.quota [--model M] [--cap N] [--runs-dir DIR ...]: the ledger's estimate per key, and when a
    run with that cap could start."""
    import argparse

    from .config import RUNS_DIR

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--model", help="default: LLM_MODEL")
    ap.add_argument("--cap", type=int, default=0, help="tokens one run needs (default: AUTOWORK_MAX_RUN_TOKENS)")
    ap.add_argument("--runs-dir", action="append", type=Path, help="extra runs directories to count (repeatable)")
    a = ap.parse_args()
    base = Settings.from_env()
    s = base.for_model(a.model) if a.model else base
    cap = a.cap or s.max_run_tokens
    dirs = [RUNS_DIR, *(a.runs_dir or [])]
    for k in _keys(s):
        q = remaining(dirs, k.key_label or LEGACY_LABEL, s.llm_model)
        line = f"{s.llm_model} {q.label}: ~{q.remaining:,} of {q.limit:,} tokens left today"
        if cap:
            w = q.wait_s(cap)
            line += f"; a {cap:,}-token run " + (
                "can start now"
                if not w
                else f"at {time.strftime('%H:%M %Z', time.localtime(time.time() + w))} (in {w / 3600:.1f} h)"
            )
        print(line)
    print("Local ledger: usage from other machines (or a checkout whose runs/ is not listed) is not visible.")


if __name__ == "__main__":
    main()
