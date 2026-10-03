"""Unit tests for evals.usage_report and agent.usage."""

from __future__ import annotations

import json
from pathlib import Path

from agent.usage import UsageLogger, analyze_sections, count_tokens, detect_provider, extract_rate_limit_headers
from agent.vault import Redactor
from evals.usage_report import analyze_run, format_report, parse_usage_file


def test_token_counting_cl100k_base():
    # count_tokens returns positive integer for non-empty text
    assert count_tokens("") == 0
    assert count_tokens("Hello world") >= 2


def test_detect_provider():
    assert detect_provider("qwen/qwen3.8-27b", "https://api.groq.com/openai/v1") == "groq"
    assert detect_provider("gemini:gemini-3.8-flash", "https://generativelanguage.googleapis.com") == "gemini"
    assert detect_provider("nvidia:nvidia/nemotron", "https://integrate.api.nvidia.com/v1") == "nvidia"
    assert detect_provider("openrouter:qwen/qwen3.8-27b:free", "https://openrouter.ai/api/v1") == "openrouter"


def test_extract_rate_limit_headers():
    hdrs = {
        "Content-Type": "application/json",
        "X-RateLimit-Limit-Requests": "30",
        "X-RateLimit-Remaining-Tokens": "5000",
        "retry-after": "6",
    }
    extracted = extract_rate_limit_headers(hdrs)
    assert extracted == {
        "x-ratelimit-limit-requests": "30",
        "x-ratelimit-remaining-tokens": "5000",
        "retry-after": "6",
    }


def test_analyze_sections_worker():
    system = "You are AutoWork.\nNotes learned from previous successful runs:\n- note 1"
    brief = "TASK FROM USER:\nFind invoice.\n\nGOAL: enter invoice\n\nStart page: http://localhost:8001/ . Begin."
    status = "[status] step 1/40. WORKING MEMORY:\ninvoice=123\nContinue: one tool call."
    tools = [{"type": "function", "function": {"name": "browser_click", "parameters": {}}}]
    obs_text = "URL: http://site\nINTERACTIVE ELEMENTS:\n  [1] button 'Submit'\nPAGE TEXT:\nInvoice text"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": brief},
        {"role": "assistant", "content": "reasoning", "tool_calls": []},
        {"role": "tool", "content": obs_text},
        {"role": "user", "content": status},
    ]
    sections = analyze_sections("worker", messages, tools)
    assert sections["system_prompt"]["tokens"] > 0
    assert sections["playbook_notes"]["tokens"] > 0
    assert sections["tool_schemas"]["tokens"] > 0
    assert sections["task_text"]["tokens"] > 0
    assert sections["plan"]["tokens"] > 0
    assert sections["working_memory"]["tokens"] > 0
    assert sections["latest_observation_dom"]["tokens"] > 0
    assert sections["latest_observation_page_text"]["tokens"] > 0


def test_usage_logger_redacts_and_logs(tmp_path: Path):
    vault_secrets = ["super_secret_token_12345"]
    redactor = Redactor(vault_secrets)
    logger = UsageLogger(tmp_path, "run-test-1", redactor)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    logger.log_call(
        timestamp_wall=100.0,
        timestamp_mono=10.0,
        latency_ms=150.0,
        role="planner",
        step=0,
        model="qwen/qwen3.8-27b",
        base_url="https://api.groq.com/openai/v1",
        messages=messages,
        tools=None,
        success=True,
        resp_usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        headers={"x-ratelimit-limit-requests": "30", "x-secret": "super_secret_token_12345"},
    )

    records = parse_usage_file(tmp_path / "llm_usage.jsonl")
    assert len(records) == 1
    rec = records[0]
    assert rec["run_id"] == "run-test-1"
    assert rec["role"] == "planner"
    assert rec["prompt_tokens"] == 120
    assert rec["completion_tokens"] == 30
    # verify secret was redacted
    raw_text = (tmp_path / "llm_usage.jsonl").read_text()
    assert "super_secret_token_12345" not in raw_text


def test_usage_report_analysis_synthetic(tmp_path: Path):
    # Construct synthetic run with planner, 2 worker calls, a 429 error, and a verifier call
    synthetic_records = [
        {
            "timestamp_wall": 1000.0,
            "timestamp_mono": 10.0,
            "run_id": "test-run-synth",
            "step": 0,
            "role": "planner",
            "model": "qwen/qwen3.8-27b",
            "provider": "groq",
            "latency_ms": 250.0,
            "success": True,
            "prompt_tokens": 500,
            "completion_tokens": 100,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 600,
            "rate_limit_headers": {
                "x-ratelimit-limit-requests": "30",
                "x-ratelimit-limit-tokens": "6000",
                "x-ratelimit-limit-tokens-day": "200000",
            },
            "sections": {
                "system_prompt": {"chars": 400, "tokens": 100},
                "tool_schemas": {"chars": 0, "tokens": 0},
                "task_text": {"chars": 200, "tokens": 50},
                "plan": {"chars": 0, "tokens": 0},
                "working_memory": {"chars": 0, "tokens": 0},
                "playbook_notes": {"chars": 0, "tokens": 0},
                "conversation_history": {"chars": 0, "tokens": 0},
                "latest_observation_dom": {"chars": 0, "tokens": 0},
                "latest_observation_page_text": {"chars": 0, "tokens": 0},
                "other": {"chars": 0, "tokens": 0},
            },
        },
        {
            "timestamp_wall": 1010.0,
            "timestamp_mono": 20.0,
            "run_id": "test-run-synth",
            "step": 1,
            "role": "worker",
            "model": "qwen/qwen3.8-27b",
            "provider": "groq",
            "latency_ms": 400.0,
            "success": True,
            "prompt_tokens": 3000,
            "completion_tokens": 150,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 3150,
            "rate_limit_headers": {
                "x-ratelimit-limit-requests": "30",
                "x-ratelimit-limit-tokens": "6000",
                "x-ratelimit-remaining-tokens": "2850",
            },
            "sections": {
                "system_prompt": {"chars": 1000, "tokens": 250},
                "tool_schemas": {"chars": 4000, "tokens": 1000},
                "task_text": {"chars": 200, "tokens": 50},
                "plan": {"chars": 400, "tokens": 100},
                "working_memory": {"chars": 200, "tokens": 50},
                "playbook_notes": {"chars": 0, "tokens": 0},
                "conversation_history": {"chars": 500, "tokens": 125},
                "latest_observation_dom": {"chars": 3500, "tokens": 875},
                "latest_observation_page_text": {"chars": 2200, "tokens": 550},
                "other": {"chars": 0, "tokens": 0},
            },
        },
        {
            "timestamp_wall": 1015.0,
            "timestamp_mono": 25.0,
            "run_id": "test-run-synth",
            "step": 2,
            "role": "worker",
            "model": "qwen/qwen3.8-27b",
            "provider": "groq",
            "latency_ms": 80.0,
            "success": False,
            "status_code": 429,
            "error_type": "RateLimitError",
            "error_body": (
                "Rate limit reached for model `qwen/qwen3.8-27b` on tokens per minute (TPM): "
                "Limit 6000, Used 5950, Requested 3200. Please try again in 5.2s."
            ),
            "rate_limit_headers": {
                "retry-after": "6",
                "x-ratelimit-limit-requests": "30",
                "x-ratelimit-limit-tokens": "6000",
                "x-ratelimit-remaining-tokens": "50",
            },
            "sections": {
                "system_prompt": {"chars": 1000, "tokens": 250},
                "tool_schemas": {"chars": 4000, "tokens": 1000},
                "task_text": {"chars": 200, "tokens": 50},
                "plan": {"chars": 400, "tokens": 100},
                "working_memory": {"chars": 200, "tokens": 50},
                "playbook_notes": {"chars": 0, "tokens": 0},
                "conversation_history": {"chars": 1000, "tokens": 250},
                "latest_observation_dom": {"chars": 3600, "tokens": 900},
                "latest_observation_page_text": {"chars": 2400, "tokens": 600},
                "other": {"chars": 0, "tokens": 0},
            },
        },
        {
            "timestamp_wall": 1022.0,
            "timestamp_mono": 32.0,
            "run_id": "test-run-synth",
            "step": 2,
            "role": "worker",
            "model": "qwen/qwen3.8-27b",
            "provider": "groq",
            "latency_ms": 420.0,
            "success": True,
            "prompt_tokens": 3200,
            "completion_tokens": 120,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 3320,
            "rate_limit_headers": {
                "x-ratelimit-limit-requests": "30",
                "x-ratelimit-limit-tokens": "6000",
                "x-ratelimit-remaining-tokens": "2680",
            },
            "sections": {
                "system_prompt": {"chars": 1000, "tokens": 250},
                "tool_schemas": {"chars": 4000, "tokens": 1000},
                "task_text": {"chars": 200, "tokens": 50},
                "plan": {"chars": 400, "tokens": 100},
                "working_memory": {"chars": 200, "tokens": 50},
                "playbook_notes": {"chars": 0, "tokens": 0},
                "conversation_history": {"chars": 1000, "tokens": 250},
                "latest_observation_dom": {"chars": 3600, "tokens": 900},
                "latest_observation_page_text": {"chars": 2400, "tokens": 600},
                "other": {"chars": 0, "tokens": 0},
            },
        },
    ]

    log_file = tmp_path / "llm_usage.jsonl"
    with log_file.open("w") as f:
        for r in synthetic_records:
            f.write(json.dumps(r) + "\n")

    loaded = parse_usage_file(log_file)
    assert len(loaded) == 4

    analysis = analyze_run(loaded)
    assert analysis["total_calls"] == 4
    assert analysis["successful_calls"] == 3
    assert analysis["rate_limited_calls"] == 1
    assert analysis["total_prompt_tokens"] == 500 + 3000 + 3200
    assert analysis["header_tpm_limit"] == 6000
    assert analysis["header_rpm_limit"] == 30
    assert analysis["header_tpd_limit"] == 200000

    # Verify 429 named limit detection
    assert len(analysis["events_429"]) == 1
    ev429 = analysis["events_429"][0]
    assert "TPM" in ev429["named_limit"]
    assert "Limit=6000" in ev429["limit_details"]
    assert ev429["prior_60s_requests"] == 2  # planner and step 1
    assert ev429["prior_60s_tokens"] == 3500

    # Verify section breakdown is sorted descending by avg tokens
    worker_sections = analysis["section_breakdown"]["worker"]["sections"]
    for i in range(len(worker_sections) - 1):
        assert worker_sections[i]["avg_tokens"] >= worker_sections[i + 1]["avg_tokens"]
    assert worker_sections[0]["name"] == "tool_schemas"

    # Test report formatting
    report_text = format_report(analysis)
    assert "TOTALS PER RUN AND PER ROLE" in report_text
    assert "ROLLING 60-SECOND WINDOWS" in report_text
    assert "RATE LIMIT 429 ERRORS" in report_text
    assert "Tokens Per Minute (TPM)" in report_text
    assert "DIAGNOSTIC VERDICT" in report_text
