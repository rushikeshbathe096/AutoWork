"""Budget mode: the per-key daily-quota ledger, the pre-run check, the run-token cap and key labels."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from agent.config import Settings
from agent.core import Agent
from agent.human import ScriptedHuman
from agent.llm import LLMClient
from agent.quota import DAY_S, QuotaShortage, preflight, remaining
from agent.usage import UsageLogger
from conftest import PLAN, FakeLLM, W, admin_post

QWEN = "qwen/qwen3.8-27b"
T0 = 1_791_000_000.0
TPD_429 = (
    '{"error":{"message":"Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier '
    "`on_demand` on tokens per day (TPD): Limit 200000, Used 198881, Requested 2564. Please try again in 10m24.24s."
    '"}}'
)


def write_usage(runs, run_id, rows):
    d = runs / run_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "llm_usage.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps({"provider": "groq", "model": QWEN, **r}) + "\n")


def use(t, prompt, completion=0, cached=0, key="groq#1"):
    return {
        "timestamp_wall": t,
        "key": key,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
    }


def test_ledger_counts_uncached_tokens_and_refills_at_a_day_rate(tmp_path):
    write_usage(tmp_path, "r1", [use(T0, 50_000, 10_000, cached=20_000), use(T0 + 60, 30_000)])
    q = remaining(tmp_path, "groq#1", QWEN, now=T0 + 60)
    assert q.limit == 200_000 and abs(q.remaining - (200_000 - 40_000 - 30_000)) <= 200  # cached not counted
    later = remaining(tmp_path, "groq#1", QWEN, now=T0 + 60 + 3600)
    assert abs((later.remaining - q.remaining) - 8_333) <= 1  # ~8,333 tokens/hour
    assert remaining(tmp_path, "groq#1", QWEN, now=T0 + 2 * DAY_S).remaining == 200_000  # full, never above


def test_ledger_is_per_key_label_and_model_and_reads_old_rows_as_key_1(tmp_path):
    write_usage(tmp_path, "r1", [use(T0, 100_000, key="groq#2")])
    write_usage(tmp_path, "r2", [{"timestamp_wall": T0, "prompt_tokens": 70_000, "completion_tokens": 0}])  # no key
    write_usage(tmp_path, "r3", [{**use(T0, 90_000), "model": "openai/gpt-oss-20b"}])
    write_usage(tmp_path, "r4", [{**use(T0, 90_000), "provider": "nvidia"}])
    assert remaining(tmp_path, "groq#2", QWEN, now=T0).remaining == 100_000
    assert remaining(tmp_path, "groq#1", QWEN, now=T0).remaining == 130_000
    assert remaining(tmp_path, "groq#3", QWEN, now=T0).remaining == 200_000


def test_ledger_reads_several_runs_directories(tmp_path):
    write_usage(tmp_path / "main", "r1", [use(T0, 60_000)])
    write_usage(tmp_path / "demo", "r2", [use(T0, 50_000)])  # e.g. a frozen demo copy's runs/
    assert remaining([tmp_path / "main", tmp_path / "demo"], "groq#1", QWEN, now=T0).remaining == 90_000


def test_ledger_is_corrected_from_a_429_tpd_body(tmp_path):
    # Our own log saw only 20k, but the key was also used elsewhere: Groq's own count wins from then on
    write_usage(tmp_path, "r1", [use(T0, 20_000)])
    write_usage(
        tmp_path, "r2", [{"timestamp_wall": T0 + 10, "key": "groq#1", "status_code": 429, "error_body": TPD_429}]
    )
    q = remaining(tmp_path, "groq#1", QWEN, now=T0 + 10)
    assert q.remaining == 200_000 - 198_881 and q.limit == 200_000
    write_usage(tmp_path, "r3", [use(T0 + 3600, 1_000)])  # usage after the correction still counts
    assert (
        abs(remaining(tmp_path, "groq#1", QWEN, now=T0 + 3600).remaining - (1_119 + 8_310 - 1_000)) <= 2
    )  # 3,590 s of refill


ENV = {"LLM_API_KEY": "k1,k2,k3", "LLM_MODEL": QWEN}


def test_keys_get_labels_and_the_run_cap_is_the_token_budget():
    s = Settings.from_env({**ENV, "AUTOWORK_MAX_RUN_TOKENS": "150000"})
    assert s.llm_key_label == "groq#1" and [f.key_label for f in s.llm_fallbacks] == ["groq#2", "groq#3"]
    assert s.max_run_tokens == 150_000 and s.max_tokens_total == 150_000
    off = Settings.from_env(ENV)
    assert off.max_run_tokens == 0 and off.max_tokens_total == 400_000
    assert Settings.from_env({"NVIDIA_API_KEY": "a,b", "LLM_MODEL": "nvidia:m"}).llm_key_label == "nvidia#1"


def test_preflight_starts_on_a_key_with_enough_quota(tmp_path):
    s = Settings.from_env({**ENV, "AUTOWORK_MAX_RUN_TOKENS": "100000"})
    write_usage(tmp_path, "r1", [use(T0, 150_000, key="groq#1")])
    chosen = preflight(s, tmp_path, now=T0)
    assert chosen.llm_key_label == "groq#2" and chosen.llm_api_key == "k2"
    assert [f.key_label for f in chosen.llm_fallbacks] == ["groq#1", "groq#3"]  # rotation still possible
    assert preflight(s, tmp_path, now=T0 + DAY_S).llm_key_label == "groq#1"  # refilled: keep the order


def test_preflight_refuses_and_says_when(tmp_path):
    s = Settings.from_env({"LLM_API_KEY": "k1,k2", "LLM_MODEL": QWEN, "AUTOWORK_MAX_RUN_TOKENS": "100000"})
    write_usage(tmp_path, "r1", [use(T0, 150_000, key="groq#1"), use(T0, 180_000, key="groq#2")])
    with pytest.raises(QuotaShortage) as e:
        preflight(s, tmp_path, now=T0)
    msg = str(e.value)
    # groq#1 has 50k and needs 50k more: 6.0 h at 8,333/h; groq#2 needs 80k more
    assert "groq#1: ~50,000 of 200,000" in msg and "groq#1 should have enough" in msg and "in 6.0 h" in msg
    assert "--resume" in msg and "other machines is not visible" in msg


def test_preflight_is_off_without_a_cap_or_off_groq(tmp_path):
    write_usage(tmp_path, "r1", [use(T0, 199_000)])
    assert preflight(Settings.from_env(ENV), tmp_path, now=T0).llm_key_label == "groq#1"  # no cap: no check
    nv = Settings.from_env({"NVIDIA_API_KEY": "n", "LLM_MODEL": "nvidia:m", "AUTOWORK_MAX_RUN_TOKENS": "100000"})
    assert preflight(nv, tmp_path, now=T0) is nv


def test_usage_rows_record_the_key_label_through_rotation(tmp_path):
    """The 429 on key 1 is logged under groq#1 (and corrects its ledger); the retried call under groq#2."""

    def response():
        return NS(
            choices=[NS(message=NS(content="hi", tool_calls=None))],
            usage=NS(prompt_tokens=10, completion_tokens=5, model_dump=lambda: {}),
        )

    class Completions:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)

        def create(self, **kw):
            o = self.outcomes.pop(0)
            if isinstance(o, Exception):
                raise o
            return o

    def client(outcomes):
        return NS(base_url="https://api.groq.com/openai/v1/", chat=NS(completions=Completions(outcomes)))

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    daily = openai.RateLimitError(TPD_429, response=httpx.Response(429, request=req, text=TPD_429), body=None)
    s = Settings.from_env({"LLM_API_KEY": "k1,k2", "LLM_MODEL": QWEN})
    c = LLMClient(s, client=client([daily]), sleep=lambda w: None, fallback_clients=[client([response()])])
    c.set_usage_logger(UsageLogger(tmp_path / "run", "run"))
    assert c.chat([{"role": "user", "content": "x"}]).content == "hi"
    rows = [json.loads(x) for x in (tmp_path / "run" / "llm_usage.jsonl").read_text().splitlines()]
    assert [(r["key"], r["success"]) for r in rows] == [("groq#1", False), ("groq#2", True)]
    assert "k1" not in json.dumps(rows) and "k2" not in json.dumps(rows)  # labels, never keys
    assert remaining(tmp_path, "groq#1", QWEN).remaining < 10_000  # corrected from the 429 body


def test_run_cap_stops_the_run_cleanly(tmp_path, ws):
    admin_post("/admin/reset")
    s = Settings.from_env({**ENV, "AUTOWORK_MAX_RUN_TOKENS": "12000"})
    llm = FakeLLM([PLAN] + [("browser_goto", {"url": W + f"/mail/{i}"}) for i in range(1, 9)])
    llm.stats["prompt_tokens"] = 0
    orig = llm.chat

    def costly(*a, **k):
        llm.stats["prompt_tokens"] += 5000
        return orig(*a, **k)

    llm.chat = costly  # type: ignore[method-assign]
    agent = Agent(llm, ScriptedHuman(), ws, tmp_path / "runs", None, max_tokens_total=s.max_tokens_total)
    r = agent.run("x")
    assert r.status == "budget_exhausted" and "token budget exhausted (15000 > 12000)" in r.summary


def test_ui_run_cap_applies_and_a_short_quota_is_refused_with_the_wait(tmp_path, monkeypatch):
    """The demo sets a cap per run in the UI (150k for the Acme run, 60k for the payment approval)."""
    import time

    from fastapi.testclient import TestClient

    import server.app as sa

    groq = Settings.from_env({"LLM_API_KEY": "k1", "LLM_MODEL": QWEN})  # no cap in .env: the UI sets it
    write_usage(tmp_path, "r1", [use(time.time(), 150_000)])  # 50k left now
    monkeypatch.setattr(sa, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(sa.Settings, "from_env", staticmethod(lambda env=None: groq))
    c = TestClient(sa.app, base_url="http://localhost:8000")
    h = {"X-AutoWork-Token": sa.CONTROL_TOKEN, "Origin": "http://localhost:8000"}
    r = c.post("/api/runs", json={"task": "x", "max_run_tokens": 60_000}, headers=h)
    assert r.status_code == 429 and "the run cap is 60,000 tokens" in r.json()["detail"]


def test_numbered_keys_keep_their_label_and_follow_the_key_order():
    """LLM_API_KEY1..3 with LLM_KEY_ORDER=2,3: key 2 first, then 3, then the rest. Labels come from the variable's
    number, not the position, so key 1's past usage in the ledger stays with key 1."""
    env = {"LLM_API_KEY1": "a", "LLM_API_KEY2": "b", "LLM_API_KEY3": "c", "LLM_MODEL": QWEN, "LLM_KEY_ORDER": "2,3"}
    s = Settings.from_env(env)
    assert (s.llm_key_label, s.llm_api_key) == ("groq#2", "b")
    assert [(f.key_label, f.api_key) for f in s.llm_fallbacks] == [("groq#3", "c"), ("groq#1", "a")]
    plain = Settings.from_env({k: v for k, v in env.items() if k != "LLM_KEY_ORDER"})
    assert [plain.llm_key_label, *(f.key_label for f in plain.llm_fallbacks)] == ["groq#1", "groq#2", "groq#3"]
    with pytest.raises(ValueError, match="LLM_KEY_ORDER"):
        Settings.from_env({**env, "LLM_KEY_ORDER": "4"})


def test_preflight_uses_the_key_order_and_skips_a_used_key(tmp_path):
    env = {
        "LLM_API_KEY1": "a",
        "LLM_API_KEY2": "b",
        "LLM_API_KEY3": "c",
        "LLM_MODEL": QWEN,
        "LLM_KEY_ORDER": "2,3",
        "AUTOWORK_MAX_RUN_TOKENS": "150000",
    }
    write_usage(tmp_path, "r1", [use(T0, 150_000, key="groq#1")])  # the old key's usage stays with key 1
    assert preflight(Settings.from_env(env), tmp_path, now=T0).llm_key_label == "groq#2"
    write_usage(tmp_path, "r2", [use(T0, 100_000, key="groq#2")])
    chosen = preflight(Settings.from_env(env), tmp_path, now=T0)
    assert chosen.llm_key_label == "groq#3" and [f.key_label for f in chosen.llm_fallbacks][:2] == ["groq#2", "groq#1"]
