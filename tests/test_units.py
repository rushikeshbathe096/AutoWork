"""Fast unit tests for the pure / mockable pieces. No browser, no network."""

from __future__ import annotations

import httpx
import openai
import pytest

from agent import policy
from agent.browser import Snapshot
from agent.config import Settings, SettingsError
from agent.context import Turn, build_messages, note
from agent.llm import LLMClient, LLMError, QuotaExhausted, ToolCall, _retry_after, parse_json
from agent.memory import Playbook, WorkingMemory
from agent.stuck import ErrorStreak, RepetitionDetector, Signal
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


# ----------------------------------------------------------------- redaction
def test_redactor_handles_nested_structures():
    r = Redactor(["hunter22", "s3cret"])
    out = r.obj({"a": ["pw hunter22", {"b": "s3cret!"}], "n": 3})
    assert out == {"a": [f"pw {REDACTED}", {"b": f"{REDACTED}!"}], "n": 3}
