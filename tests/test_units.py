"""Fast unit tests for the pure / mockable pieces. No browser, no network."""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from agent import policy
from agent.browser import Snapshot
from agent.config import Settings, SettingsError
from agent.context import Turn, build_messages, note
from agent.llm import LLMClient, LLMError, QuotaExhausted, ToolCall, _retry_after, parse_json
from agent.memory import Playbook, WorkingMemory
from agent.stuck import ErrorStreak, ProgressTracker, RepetitionDetector, Signal
from agent.tools import ToolResult
from agent.vault import REDACTED, Redactor


# ----------------------------------------------------------------- policy edge cases
def snap(*elements, has_password=False):
    return Snapshot("http://localhost:8001/erp/bills/1", "t", 200, list(elements), [], "", has_password)


def btn(i, text, **kw):
    return {"id": i, "tag": "button", "text": text, **kw}


@pytest.mark.parametrize(
    "label",
    [
        "Mark as paid",
        "MARK AS PAID",
        "pay now",
        "Delete vendor",
        "Approve payment",
        "Wire funds",
        "Cancel subscription",
    ],
)
def test_high_risk_labels_need_approval(label):
    assert policy.evaluate("browser_click", {"element_id": 1}, snap(btn(1, label))).verdict == "approve"


@pytest.mark.parametrize("label", ["", "Settle", "Prepay", "Go", "Search", "Sign in"])
def test_labels_the_rule_does_not_catch(label):
    """Documented limitation: the label rule is a keyword list. These are allowed by the LABEL rule;
    the network-level gate (netpolicy.is_high_risk) is what catches a payment behind such a button."""
    assert policy.evaluate("browser_click", {"element_id": 1}, snap(btn(1, label))).verdict == "allow"


def test_submit_input_and_role_button_are_controls():
    s = snap(
        {"id": 1, "tag": "input", "type": "submit", "text": "Pay"}, {"id": 2, "tag": "a", "text": "Pay", "href": "/x"}
    )
    assert policy.evaluate("browser_click", {"element_id": 1}, s).verdict == "approve"
    assert policy.evaluate("browser_click", {"element_id": 2}, s).verdict == "allow"  # links only navigate


def test_write_buttons_gated_only_in_supervised_and_never_on_login_forms():
    s = snap(btn(1, "Save bill"))
    assert policy.evaluate("browser_click", {"element_id": 1}, s, "balanced").verdict == "allow"
    assert policy.evaluate("browser_click", {"element_id": 1}, s, "supervised").verdict == "approve"
    login = snap(btn(1, "Submit"), has_password=True)
    assert policy.evaluate("browser_click", {"element_id": 1}, login, "supervised").verdict == "allow"


def test_bad_element_ids_are_left_to_the_browser_layer():
    for eid in ("abc", -3, 99):
        assert policy.evaluate("browser_click", {"element_id": eid}, snap(btn(1, "Pay"))).verdict == "allow"


# ----------------------------------------------------------------- snapshot rendering
def test_snapshot_render_truncates_and_flags_failures():
    els = [{"id": i, "tag": "a", "text": f"link {i}", "href": f"/{i}"} for i in range(1, 101)]
    s = Snapshot("http://x/p", "Title", 504, els, ["Something broke"], "x" * 5000, False)
    out = s.render(text_chars=100)
    assert "HTTP STATUS: 504" in out and "ALERTS ON PAGE: Something broke" in out
    assert "[70] link" in out and "[71] link" not in out and "30 more elements" in out
    assert "4900 more chars" in out
    assert out.startswith("URL: http://x/p\n<<<UNTRUSTED_WEB_PAGE") and out.endswith("END_UNTRUSTED_WEB_PAGE>>>")


def test_page_text_skips_lines_that_repeat_listed_elements_and_is_shorter_with_elements():
    els = [
        btn(1, "Save bill"),
        {"id": 2, "tag": "a", "text": "Bills", "href": "/b"},
        {"id": 3, "tag": "input", "type": "text", "label": "Due date (YYYY-MM-DD)", "value": ""},
    ]
    text = "Bills\nNew bill\nDue date (YYYY-MM-DD)\n  save   BILL \nInvoice IN-7002 due 2026-10-30"
    out = Snapshot("http://x/p", "T", 200, els, [], text, False).render()
    page = out.split("PAGE TEXT:\n", 1)[1]
    assert page.startswith("New bill\nInvoice IN-7002 due 2026-10-30")  # repeats of elements dropped
    assert "[1] button" in out and "[2] link" in out  # ...they are still listed as elements
    long = "y" * 3000
    assert "1800 more chars" in Snapshot("u", "T", 200, els, [], long, False).render()  # 1200 with elements
    assert "1200 more chars" in Snapshot("u", "T", 200, [], [], long, False).render()  # 1800 without


def test_snapshot_masks_nothing_it_was_not_given_and_fingerprint_tracks_content():
    a = Snapshot("http://x/p", "t", 200, [], [], "hello", False)
    b = Snapshot("http://x/p", "t", 200, [], [], "hello!", False)
    assert (
        a.fingerprint() != b.fingerprint()
        and a.fingerprint() == Snapshot("http://x/p", "other", 500, [], [], "hello", True).fingerprint()
    )


# ----------------------------------------------------------------- parse_json
@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here it is: {"a": 1} hope it helps'])
def test_parse_json_is_lenient(text):
    assert parse_json(text) == {"a": 1}


def test_parse_json_rejects_garbage():
    with pytest.raises(ValueError):
        parse_json("no json here")


# ----------------------------------------------------------------- LLM client retries (mocked SDK)
class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def create(self, **kw):
        self.calls.append(kw)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


class FakeClient:
    def __init__(self, outcomes):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(outcomes)


def ok_response(content="hi", tool=None):
    from types import SimpleNamespace as NS

    tcs = [NS(id="c1", function=NS(name=tool[0], arguments=tool[1]))] if tool else None
    return NS(
        choices=[NS(message=NS(content=content, tool_calls=tcs))],
        usage=NS(prompt_tokens=10, completion_tokens=5, model_dump=lambda: {}),
    )


def err(cls, status, body="", headers=None):
    req = httpx.Request("POST", "http://llm")
    return cls(body or "error", response=httpx.Response(status, request=req, headers=headers or {}), body=None)


SETTINGS = Settings.from_env({"LLM_API_KEY": "test"})


def make(outcomes):
    waits: list[float] = []
    c = LLMClient(SETTINGS, client=FakeClient(outcomes), sleep=waits.append)  # type: ignore[arg-type]
    return c, waits


def test_rate_limit_honours_retry_after():
    c, waits = make([err(openai.RateLimitError, 429, headers={"retry-after": "3"}), ok_response()])
    assert c.chat([{"role": "user", "content": "x"}]).content == "hi"
    assert waits == [3.5] and c.stats["retries"] == 1 and c.stats["prompt_tokens"] == 10


def test_malformed_tool_call_is_retried_with_a_correction():
    c, _ = make([err(openai.BadRequestError, 400, "tool_use_failed: bad"), ok_response(tool=("finish", "{}"))])
    r = c.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert r.tool_calls[0].name == "finish"
    second = c.client.chat.completions.calls[1]["messages"]
    assert "malformed" in second[-1]["content"]


def test_tool_choice_required_falls_back_to_auto():
    c, _ = make([err(openai.BadRequestError, 400, "tool_choice required is not supported"), ok_response()])
    c.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}], require_tool=True)
    calls = c.client.chat.completions.calls
    assert calls[0]["tool_choice"] == "required" and calls[1]["tool_choice"] == "auto"


def test_invalid_tool_arguments_are_passed_through_not_raised():
    c, _ = make([ok_response(tool=("browser_click", "{not json"))])
    r = c.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert r.tool_calls[0].arguments == {"__invalid_json__": "{not json"}


def test_gives_up_after_max_attempts_and_rejects_other_400s():
    c, waits = make([err(openai.InternalServerError, 500)] * 3)
    with pytest.raises(LLMError):
        c.chat([{"role": "user", "content": "x"}], max_attempts=3)
    assert len(waits) == 3
    c, _ = make([err(openai.BadRequestError, 400, "context length exceeded")])
    with pytest.raises(LLMError, match="rejected"):
        c.chat([{"role": "user", "content": "x"}])


def test_rate_limits_do_not_consume_failure_attempts():
    # Regression from the first live eval: free-tier 429s used up max_attempts and killed a healthy run.
    limited = err(openai.RateLimitError, 429, headers={"retry-after": "1"})
    c, waits = make([limited] * 8 + [ok_response()])
    assert c.chat([{"role": "user", "content": "x"}], max_attempts=3).content == "hi"
    assert len(waits) == 8


def test_rate_limit_wait_is_bounded_in_seconds():
    limited = err(openai.RateLimitError, 429, headers={"retry-after": "50"})
    c, waits = make([limited] * 10)
    assert not c.quota_exhausted
    with pytest.raises(QuotaExhausted, match="rate limited"):
        c.chat([{"role": "user", "content": "x"}], max_rate_limit_wait_s=120)
    assert sum(waits) <= 120
    assert c.quota_exhausted  # the eval harness discards such runs instead of grading them


def test_malformed_tool_call_corrections_do_not_stack():
    bad = err(openai.BadRequestError, 400, "tool_use_failed: bad")
    c, _ = make([bad, bad, ok_response(tool=("finish", "{}"))])
    c.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    third = c.client.chat.completions.calls[2]["messages"]
    assert len(third) == 2 and "malformed" in third[-1]["content"]


def test_reasoning_budget_exhaustion_retries_with_more_budget_and_less_effort():
    # Regression from the first live eval: gpt-oss spent max_tokens on hidden reasoning and never wrote the
    # tool call. Groq reports that as tool_use_failed with an empty failed_generation; a nudge can't fix it.
    exhausted = err(openai.BadRequestError, 400, "{'error': {'code': 'tool_use_failed', 'failed_generation': ''}}")
    s = Settings.from_env({"LLM_API_KEY": "t", "LLM_REASONING_EFFORT": "medium", "LLM_MAX_OUTPUT_TOKENS": "1000"})
    c = LLMClient(s, client=FakeClient([exhausted, ok_response(tool=("finish", "{}"))]), sleep=lambda w: None)  # type: ignore[arg-type]
    c.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}], require_tool=True)
    first, second = c.client.chat.completions.calls
    assert (first["max_tokens"], first["reasoning_effort"]) == (1000, "medium")
    assert (second["max_tokens"], second["reasoning_effort"]) == (2000, "low")
    assert len(second["messages"]) == 1  # no misleading "your call was malformed" correction


def test_reasoning_effort_is_not_sent_unless_configured():
    c, _ = make([ok_response()])
    c.chat([{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in c.client.chat.completions.calls[0]


def test_provider_prefix_selects_base_url_key_and_api_model():
    env = {"LLM_API_KEY": "groq-key", "GEMINI_API_KEY": "gem-key", "LLM_MODEL": "gemini:gemini-3.8-flash"}
    s = Settings.from_env(env)
    assert s.llm_api_key == "gem-key" and "generativelanguage.googleapis.com" in s.llm_base_url
    assert s.llm_model == "gemini:gemini-3.8-flash" and s.api_model == "gemini-3.8-flash"
    groq = s.for_model("qwen/qwen3.8-27b", env)  # slashes are not provider prefixes
    assert groq.llm_api_key == "groq-key" and groq.api_model == "qwen/qwen3.8-27b" and "groq" in groq.llm_base_url


def test_provider_prefix_without_its_key_fails_fast():
    with pytest.raises(SettingsError, match="GEMINI_API_KEY"):
        Settings.from_env({"LLM_API_KEY": "k", "LLM_MODEL": "gemini:gemini-3.8-flash"})


def test_request_uses_provider_model_id_and_drops_rejected_parallel_param():
    s = Settings.from_env({"GEMINI_API_KEY": "g", "LLM_MODEL": "gemini:gemini-3.8-flash"})
    rejected = err(openai.BadRequestError, 400, "Unknown name 'parallel_tool_calls': Cannot find field.")
    c = LLMClient(s, client=FakeClient([rejected, ok_response()]), sleep=lambda w: None)  # type: ignore[arg-type]
    c.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    first, second = c.client.chat.completions.calls
    assert first["model"] == "gemini-3.8-flash" and "parallel_tool_calls" in first
    assert "parallel_tool_calls" not in second and c.model == "gemini:gemini-3.8-flash"


def test_retry_after_parses_groq_message():
    e = err(openai.RateLimitError, 429, "Rate limit reached. Please try again in 350ms.")
    assert _retry_after(e) == pytest.approx(0.85)


def test_retry_after_reads_long_waits_in_full():
    # Regressions: Groq's "7h8m49.92s" was read as 49.92 s, and Gemini's retryDelay was not read at all,
    # so the client kept retrying models that were blocked for hours.
    groq = err(openai.RateLimitError, 429, "Limit 200000, Used 198501. Please try again in 7h8m49.92s. Need more")
    assert _retry_after(groq) == pytest.approx(7 * 3600 + 8 * 60 + 49.92 + 0.5)
    gemini = err(openai.RateLimitError, 429, "[{'error': {'details': [{'retryDelay': '46179s'}]}}]")
    assert _retry_after(gemini) == pytest.approx(46179.5)


def test_daily_quota_gives_up_immediately_instead_of_waiting():
    c, waits = make([err(openai.RateLimitError, 429, "Please try again in 6h4m51.1s.")])
    with pytest.raises(QuotaExhausted):
        c.chat([{"role": "user", "content": "x"}])
    assert waits == [] and c.quota_exhausted


FALLBACK_ENV = {
    "LLM_API_KEY": "groq",
    "GEMINI_API_KEY": "g",
    "NVIDIA_API_KEY": "n",
    "OPENROUTER_API_KEY": "o",
    "LLM_FALLBACK_MODELS": "gemini:gemini-3.5-flash, nvidia:nvidia/nemotron-3-super-120b-a12b,"
    "openrouter:qwen/qwen3.8-27b:free",
}


def test_fallback_models_resolve_their_own_provider_and_key():
    fbs = Settings.from_env(FALLBACK_ENV).llm_fallbacks
    assert [(f.model, f.api_key) for f in fbs] == [
        ("gemini:gemini-3.5-flash", "g"),
        ("nvidia:nvidia/nemotron-3-super-120b-a12b", "n"),
        ("openrouter:qwen/qwen3.8-27b:free", "o"),
    ]
    assert "openrouter.ai" in fbs[2].base_url
    with pytest.raises(SettingsError, match="NVIDIA_API_KEY"):
        Settings.from_env({**FALLBACK_ENV, "NVIDIA_API_KEY": ""})
    assert Settings.from_env(FALLBACK_ENV).for_model("qwen/qwen3.8-27b", FALLBACK_ENV).llm_fallbacks == ()


def test_out_of_quota_switches_to_next_fallback_and_stays_there():
    s = Settings.from_env(FALLBACK_ENV)
    daily = err(openai.RateLimitError, 429, "Please try again in 6h4m51.1s.")
    down = err(openai.InternalServerError, 503, "high demand")
    groq, gemini, nvidia, openrouter = (
        FakeClient([daily]),
        FakeClient([down] * 6),
        FakeClient([ok_response(), ok_response()]),
        FakeClient([]),
    )
    c = LLMClient(s, client=groq, sleep=lambda w: None, fallback_clients=[gemini, nvidia, openrouter])  # type: ignore[arg-type]
    assert c.chat([{"role": "user", "content": "x"}]).content == "hi"
    assert c.chat([{"role": "user", "content": "y"}]).content == "hi"
    assert c.calls_by_model == {
        "qwen/qwen3.8-27b": 0,
        "gemini:gemini-3.5-flash": 0,
        "nvidia:nvidia/nemotron-3-super-120b-a12b": 2,
    }
    assert nvidia.chat.completions.calls[0]["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert len(nvidia.chat.completions.calls) == 2 and not c.quota_exhausted


def test_quota_exhausted_only_when_every_fallback_is():
    s = Settings.from_env({**FALLBACK_ENV, "LLM_FALLBACK_MODELS": "gemini:gemini-3.5-flash"})
    daily = err(openai.RateLimitError, 429, "Please try again in 6h4m51.1s.")
    c = LLMClient(s, client=FakeClient([daily]), sleep=lambda w: None, fallback_clients=[FakeClient([daily])])  # type: ignore[arg-type]
    with pytest.raises(QuotaExhausted):
        c.chat([{"role": "user", "content": "x"}])
    assert c.quota_exhausted


def test_missing_api_key_fails_clearly():
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        LLMClient(Settings.from_env({}))


# ----------------------------------------------------------------- settings validation
def test_settings_collects_all_errors():
    with pytest.raises(SettingsError) as ei:
        Settings.from_env(
            {"AUTOWORK_MAX_STEPS": "4o", "LLM_BASE_URL": "ftp://x", "AUTOWORK_MODE": "yolo", "AUTOWORK_MAX_TOKENS": "5"}
        )
    msg = str(ei.value)
    for part in ("AUTOWORK_MAX_STEPS", "LLM_BASE_URL", "AUTOWORK_MODE", "AUTOWORK_MAX_TOKENS"):
        assert part in msg


def test_settings_defaults_and_key_fallback():
    s = Settings.from_env({"GROQ_API_KEY": "g"})
    assert s.llm_api_key == "g" and s.max_steps == 40 and s.mode == "balanced"
    assert Settings.from_env({"GROQ_API_KEY": "g", "LLM_API_KEY": "l"}).llm_api_key == "l"


# ----------------------------------------------------------------- stuck detection
def test_repetition_detector_stages_and_fingerprint():
    d = RepetitionDetector()
    seq = [d.observe("browser_click", {"element_id": 3}, "fp1")[0] for _ in range(4)]
    assert seq == [Signal.OK, Signal.OK, Signal.WARN, Signal.ESCALATE]
    assert d.observe("browser_click", {"element_id": 3}, "fp1")[0] is Signal.OK  # reset after escalation
    d2 = RepetitionDetector()
    for i in range(5):  # the page changes every time: not stuck
        assert d2.observe("browser_click", {"element_id": 3}, f"fp{i}")[0] is Signal.OK
    for _ in range(5):
        assert d2.observe("remember", {"key": "k"}, "fp")[0] is Signal.OK
    assert RepetitionDetector.signature("t", {"b": 1, "a": 2}, "f") == RepetitionDetector.signature(
        "t", {"a": 2, "b": 1}, "f"
    )


def test_progress_tracker_stops_busy_work():
    p = ProgressTracker(stop_after=5)  # warns at 2 (5 - 3)
    assert p.observe("/mail", "inbox page", {}) is Signal.OK  # first sight of a URL and its text
    # re-typing the search box (no observation) and reopening the same inbox: nothing new
    seq = [p.observe("/mail", obs, {}) for obs in (None, None, "inbox page", None)]
    assert seq == [Signal.OK, Signal.WARN, Signal.OK, Signal.OK]
    assert p.observe("/mail", None, {}) is Signal.STOP and p.stalled == 5
    assert "no new URL, no new memory fact" in ProgressTracker.why(5)


def test_progress_tracker_counts_each_kind_of_progress():
    p = ProgressTracker(stop_after=3)
    p.observe("/a", "page a", {})
    p.observe("/a", None, {})
    assert p.observe("/b", None, {}) is Signal.OK and p.stalled == 0  # new URL
    p.observe("/b", None, {})
    assert p.observe("/b", "page b, next chunk", {}) is Signal.OK and p.stalled == 0  # new information
    p.observe("/b", None, {})
    assert p.observe("/b", None, {"vendor": "Globex"}) is Signal.OK and p.stalled == 0  # new memory fact
    p.observe("/b", None, {"vendor": "Globex"})  # re-remembering the same fact is not new
    assert p.stalled == 1
    assert p.observe("/b", None, {"vendor": "Globex"}, state_changed=True) is Signal.OK and p.stalled == 0
    assert ProgressTracker(stop_after=0).observe("/a", None, {}) is Signal.OK  # disabled: never stops
    off = ProgressTracker(stop_after=0)
    assert all(off.observe("/a", None, {}) is Signal.OK for _ in range(50))


def test_error_streak():
    e = ErrorStreak()
    assert [e.observe(False) for _ in range(6)] == [
        Signal.OK,
        Signal.OK,
        Signal.WARN,
        Signal.OK,
        Signal.OK,
        Signal.ESCALATE,
    ]
    assert e.observe(True) is Signal.OK and e.count == 0


# ----------------------------------------------------------------- memory limits
def test_working_memory_limits():
    m = WorkingMemory()
    m.remember("k" * 100, "v" * 1000, 1)
    ((key, val),) = m.as_dict().items()
    assert len(key) == 60 and len(val) == 500
    assert "was" in m.remember("k" * 60, "new", 2)


def test_playbook_dedupes_caps_and_skips_long_notes(tmp_path):
    pb = Playbook(tmp_path / "pb.json", max_notes=3)
    pb.add(["A", "a", "x" * 400, "B"], "r1")
    assert [n["note"] for n in pb.load()] == ["A", "B"]
    pb.add(["C", "D"], "r2")
    assert [n["note"] for n in pb.load()] == ["B", "C", "D"]
    assert Playbook(tmp_path / "missing.json").render() == ""


# ----------------------------------------------------------------- context compression
def test_build_messages_compresses_old_observations():
    turns = [Turn("r", f"c{i}", "browser_goto", "{}", ToolResult("FULL" + "x" * 500, f"short{i}")) for i in range(4)]
    turns.insert(2, note("a system note"))
    turns[-1].extra_calls = [ToolCall("e1", "browser_click", {}, "{}")]
    msgs = build_messages("SYS", "BRIEF", turns, "STATUS", keep_full=2)
    tool_contents = [m["content"] for m in msgs if m["role"] == "tool"]
    assert tool_contents[:2] == ["short0", "short1"] and tool_contents[2].startswith("FULL")
    assert tool_contents[-1].startswith("Not executed")
    assert {"role": "user", "content": "a system note"} in msgs and msgs[-1]["content"] == "STATUS"


def _page(i: int) -> ToolResult:
    return ToolResult(f"URL: http://w/p{i}\n" + "x" * 500, f"[opened -> http://w/p{i} | P{i}] (old observation elided)")


def _digest_turns() -> list[Turn]:
    """10 steps: pages, a bad-arguments error, a human denial, a human note, a memory write."""
    t = [
        Turn(f"reason {i}", f"c{i}", "browser_goto", json.dumps({"url": f"http://w/p{i}"}), _page(i)) for i in range(10)
    ]
    t[1] = Turn(
        "r",
        "c1",
        "browser_back",
        '{"session_id": "x"}',
        ToolResult(
            "ERROR: bad arguments for browser_back: unexpected keyword argument 'session_id'", "bad arguments", ok=False
        ),
    )
    t[2] = Turn(
        "r",
        "c2",
        "browser_click",
        '{"element_id": 9}',
        ToolResult(
            "DENIED by human: controller must approve payments. Do not retry this action.", "denied by human", ok=False
        ),
    )
    t[4] = Turn(
        "r", "c4", "remember", '{"key": "amount", "value": "4250.00"}', ToolResult("Stored 'amount'", "Stored 'amount'")
    )
    t.insert(3, note("Human guidance after repeated failures: use the ERP search box"))
    return t


def test_digest_collapses_old_turns_into_one_message_and_keeps_recent_native():
    turns = _digest_turns()
    msgs = build_messages("SYS", "BRIEF", turns, "STATUS", keep_full=2, scheme="digest")
    digests = [m for m in msgs if m["role"] == "user" and m["content"].startswith("EARLIER STEPS")]
    assert len(digests) == 1 and msgs[2] is digests[0]  # right after the brief
    native = [m for m in msgs if m["role"] == "assistant"]
    assert len(native) == 3  # the last 3 turns (the 2 full observations are among them)
    for i, m in enumerate(msgs):  # every native tool call is answered, so the request is valid
        if m["role"] == "assistant":
            assert msgs[i + 1]["role"] == "tool" and msgs[i + 1]["tool_call_id"] == m["tool_calls"][0]["id"]
    tool_texts = [m["content"] for m in msgs if m["role"] == "tool"]
    assert sum(t.startswith("URL:") for t in tool_texts) == 2  # keep_full=2 still holds
    assert msgs[-1]["content"] == "STATUS"


def test_digest_never_drops_errors_denials_human_notes_or_memory_writes():
    text = build_messages("S", "B", _digest_turns(), "ST", keep_full=2, scheme="digest")[2]["content"]
    lines = text.splitlines()[1:]
    assert lines[0] == '1. browser_goto(url="http://w/p0") -> [opened -> http://w/p0 | P0]'  # pages: short form
    assert '2. browser_back(session_id="x") -> FAILED: ERROR: bad arguments for browser_back: unexpected' in lines[1]
    assert "DENIED by human: controller must approve payments. Do not retry this action." in lines[2]
    assert lines[3] == "   note: Human guidance after repeated failures: use the ERP search box"
    assert lines[4].startswith("4. browser_goto")  # notes don't shift step numbers
    assert lines[5] == '5. remember(key="amount", value="4250.00") -> Stored \'amount\''
    assert "reason" not in text  # old reasoning is dropped


def test_digest_keeps_both_full_observations_native_even_beyond_three_turns():
    turns = [Turn("r", f"c{i}", "browser_goto", "{}", _page(i)) for i in range(5)]
    turns += [Turn("r", f"m{i}", "remember", "{}", ToolResult("Stored", "Stored")) for i in range(3)]
    msgs = build_messages("S", "B", turns, "ST", keep_full=2, scheme="digest")
    tool_texts = [m["content"] for m in msgs if m["role"] == "tool"]
    assert [t[:15] for t in tool_texts if t.startswith("URL:")] == ["URL: http://w/p", "URL: http://w/p"]
    assert len([m for m in msgs if m["role"] == "assistant"]) == 5  # tail extended from 3 to 5 turns


def test_digest_is_identical_to_classic_for_short_runs_and_rejects_unknown_schemes():
    turns = [Turn("r", f"c{i}", "browser_goto", "{}", _page(i)) for i in range(3)]
    assert build_messages("S", "B", turns, "ST", scheme="digest") == build_messages("S", "B", turns, "ST")
    with pytest.raises(ValueError, match="unknown context scheme"):
        build_messages("S", "B", turns, "ST", scheme="zip")


def test_context_scheme_setting_is_validated_and_survives_for_model():
    assert Settings.from_env({"LLM_API_KEY": "k"}).context_scheme == "classic"
    env = {"LLM_API_KEY": "k", "AUTOWORK_CONTEXT": "Digest"}
    assert Settings.from_env(env).for_model("qwen/qwen3.8-27b", env).context_scheme == "digest"
    with pytest.raises(SettingsError, match="AUTOWORK_CONTEXT='zip'"):
        Settings.from_env({"LLM_API_KEY": "k", "AUTOWORK_CONTEXT": "zip"})


# ----------------------------------------------------------------- redaction
def test_redactor_handles_nested_structures():
    r = Redactor(["hunter22", "s3cret"])
    out = r.obj({"a": ["pw hunter22", {"b": "s3cret!"}], "n": 3})
    assert out == {"a": [f"pw {REDACTED}", {"b": f"{REDACTED}!"}], "n": 3}


@pytest.mark.parametrize(
    "cls,status,hint",
    [
        (openai.AuthenticationError, 401, "API key"),
        (openai.PermissionDeniedError, 403, "no access"),
        (openai.NotFoundError, 404, "model name"),
    ],
)
def test_config_errors_raise_clearly_and_never_fall_back(cls, status, hint):
    from agent.llm import ProviderConfigError

    s = Settings.from_env(FALLBACK_ENV)
    primary, fallback = FakeClient([err(cls, status, "invalid")]), FakeClient([ok_response()])
    c = LLMClient(s, client=primary, sleep=lambda w: None, fallback_clients=[fallback, FakeClient([]), FakeClient([])])  # type: ignore[arg-type]
    with pytest.raises(ProviderConfigError, match=f"qwen/qwen3.8-27b: provider returned {status}.*{hint}"):
        c.chat([{"role": "user", "content": "x"}])
    assert len(primary.chat.completions.calls) == 1 and fallback.chat.completions.calls == []  # no retry, no switch
    assert c.model == "qwen/qwen3.8-27b" and not c.quota_exhausted


# ----------------------------------------------------------------- world config (config/world.json)
ORIGINAL_RISK = {  # the vocabulary that was hardcoded in agent/policy.py and agent/netpolicy.py before the move
    "button_words": [
        "pay",
        "paid",
        "payment",
        "delete",
        "remove",
        "wire",
        "transfer",
        "refund",
        "approve",
        "cancel",
        "terminate",
    ],
    "path_segments": [
        "pay",
        "payment",
        "payments",
        "delete",
        "remove",
        "transfer",
        "wire",
        "refund",
        "approve",
        "cancel",
    ],
    "path_substrings": ["bank", "iban", "payout"],
    "field_substrings": ["bank", "iban", "swift", "routing", "account_number", "payout"],
}


def test_world_compiler_reproduces_the_previously_hardcoded_rules_exactly():
    """The gates' vocabulary moved from agent code to config/world.json. Compiled from the original word lists, the
    rules must be identical to the regex literals the code had before the move, so nothing changed for the ERP."""
    import re as re_

    from agent.world import parse_world

    w = parse_world(
        {
            "start_url": "http://localhost:8001/",
            "description": "d",
            "allowed_origins": ["http://localhost:8001"],
            "high_risk": ORIGINAL_RISK,
        }
    )
    old_label = r"\b(pay|paid|payment|delete|remove|wire|transfer|refund|approve|cancel|terminate)\b"
    old_path = r"/(pay|payment|payments|delete|remove|transfer|wire|refund|approve|cancel)(/|$)|bank|iban|payout"
    old_field = r"(^|&)[^=&]*(bank|iban|swift|routing|account_number|payout)[^=&]*="
    assert (w.risk.button_label.pattern, w.risk.path.pattern, w.risk.form_field.pattern) == (
        old_label,
        old_path,
        old_field,
    )
    assert all(p.flags & re_.I for p in (w.risk.button_label, w.risk.path, w.risk.form_field))


def test_checked_in_world_keeps_every_original_rule_and_origin():
    import json as json_

    from agent.world import DEFAULT_WORLD_FILE, default_world

    risk = json_.loads(DEFAULT_WORLD_FILE.read_text())["high_risk"]
    for key, words in ORIGINAL_RISK.items():
        assert set(words) <= set(risk[key]), key  # apps may add risky actions; none may be dropped silently
    w = default_world()
    assert w.allowed_origins == frozenset({("http", "localhost", 8001)}) and w.blocked_path_prefixes == ("/admin",)
    assert w.description.startswith("Company intranet start page: http://localhost:8001/  (links to webmail")


def _world_env(tmp_path, **override) -> dict:
    import json as json_

    from agent.world import DEFAULT_WORLD_FILE

    data = {**json_.loads(DEFAULT_WORLD_FILE.read_text()), **override}
    f = tmp_path / "world.json"
    f.write_text(json_.dumps(data))
    return {"LLM_API_KEY": "k", "AUTOWORK_WORLD_FILE": str(f)}


def test_world_file_is_validated_at_startup(tmp_path):
    with pytest.raises(SettingsError, match="start_url .* must be an http"):
        Settings.from_env(_world_env(tmp_path, start_url="http://elsewhere.example/"))
    with pytest.raises(SettingsError, match="button_words must be a non-empty list"):
        Settings.from_env(
            _world_env(
                tmp_path,
                high_risk={
                    "button_words": [],
                    "path_segments": ["x"],
                    "path_substrings": ["x"],
                    "field_substrings": ["x"],
                },
            )
        )
    with pytest.raises(SettingsError, match="cannot read"):
        Settings.from_env({"LLM_API_KEY": "k", "AUTOWORK_WORLD_FILE": str(tmp_path / "missing.json")})


def test_a_new_apps_risky_actions_are_declared_in_config_not_code(tmp_path):
    from agent.core import Agent

    risk = {
        "button_words": ["pay", "grant admin"],
        "path_segments": ["pay"],
        "path_substrings": ["iban"],
        "field_substrings": ["iban"],
    }
    w = Settings.from_env(_world_env(tmp_path, high_risk=risk, start_url="http://localhost:8001/helpdesk")).world
    finance_only = Settings.from_env(_world_env(tmp_path, high_risk=ORIGINAL_RISK)).world
    button = snap(btn(1, "Grant admin"))
    undeclared = policy.evaluate("browser_click", {"element_id": 1}, button, high_risk=finance_only.risk.button_label)
    assert undeclared.verdict == "allow"  # not declared: the gate doesn't know it is risky (a documented limitation)
    assert (
        policy.evaluate("browser_click", {"element_id": 1}, button).verdict == "approve"
    )  # checked-in world declares it
    assert (
        policy.evaluate("browser_click", {"element_id": 1}, button, high_risk=w.risk.button_label).verdict == "approve"
    )
    assert Agent._brief("t", {}, w.start_url).endswith("Start page: http://localhost:8001/helpdesk . Begin.")
