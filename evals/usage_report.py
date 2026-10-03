"""LLM usage and rate-limit diagnostics report.

Reads runs/<run_id>/llm_usage.jsonl and reports:
1. Totals per run and per role (calls, prompt tokens, completion tokens, reasoning tokens).
2. Rolling 60-second windows: peak RPM and peak TPM vs provider header limits.
3. Every 429 event with the specific limit named in error body and rolling usage just before it.
4. Average and max prompt size per role with section breakdown sorted largest first.
5. Growth of prompt size across steps within a run (history accumulation analysis).
6. One-paragraph verdict on binding limit, top 3 sections to shrink with token savings, and daily run capacity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SECTION_NAMES = [
    "system_prompt",
    "tool_schemas",
    "task_text",
    "plan",
    "working_memory",
    "playbook_notes",
    "conversation_history",
    "latest_observation_dom",
    "latest_observation_page_text",
    "other",
]

SECTION_DISPLAY = {
    "system_prompt": "System Prompt",
    "tool_schemas": "Tool/Function Schemas",
    "task_text": "Task Text",
    "plan": "Plan & Criteria",
    "working_memory": "Working Memory",
    "playbook_notes": "Playbook Notes",
    "conversation_history": "Conversation History",
    "latest_observation_dom": "Latest Obs (DOM Elements)",
    "latest_observation_page_text": "Latest Obs (Page Text)",
    "other": "Other/Template Glue",
}


def parse_duration_to_seconds(text: str) -> float | None:
    """Parse durations like '7h8m49s', '2.5s', '350ms'."""
    text = text.strip()
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)(ms|s))?", text)
    if not m or not any(m.groups()):
        return None
    h, mins, num, unit = m.groups()
    secs = float(num or 0) / (1000 if unit == "ms" else 1)
    return int(h or 0) * 3600 + int(mins or 0) * 60 + secs


def parse_usage_file(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of record dicts."""
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_limit_from_error(error_body: str | None) -> dict[str, Any]:
    """Parse provider error text (Groq, Gemini, etc.) for named limit details."""
    res: dict[str, Any] = {"named_limit": "Unknown limit", "details": ""}
    if not error_body:
        return res

    # Groq format: "on tokens per minute (TPM): Limit 6000, Used 5980, Requested 120"
    m_tpm = re.search(
        r"tokens per minute \(TPM\)(?::\s*Limit\s*([\d,]+),\s*Used\s*([\d,]+),\s*Requested\s*([\d,]+))?",
        error_body,
        re.I,
    )
    if m_tpm:
        limit = m_tpm.group(1)
        used = m_tpm.group(2)
        req = m_tpm.group(3)
        res["named_limit"] = "Tokens Per Minute (TPM)"
        if limit:
            res["details"] = f"Limit={limit}, Used={used}, Requested={req}"
            res["limit_val"] = int(limit.replace(",", ""))
            res["used_val"] = int(used.replace(",", ""))
            res["requested_val"] = int(req.replace(",", ""))
        return res

    m_rpm = re.search(
        r"requests per minute \(RPM\)(?::\s*Limit\s*([\d,]+),\s*Used\s*([\d,]+),\s*Requested\s*([\d,]+))?",
        error_body,
        re.I,
    )
    if m_rpm:
        res["named_limit"] = "Requests Per Minute (RPM)"
        if m_rpm.group(1):
            res["details"] = f"Limit={m_rpm.group(1)}, Used={m_rpm.group(2)}, Requested={m_rpm.group(3)}"
        return res

    m_tpd = re.search(
        r"tokens per day \(TPD\)(?::\s*Limit\s*([\d,]+),\s*Used\s*([\d,]+),\s*Requested\s*([\d,]+))?",
        error_body,
        re.I,
    )
    if m_tpd:
        res["named_limit"] = "Tokens Per Day (TPD)"
        if m_tpd.group(1):
            res["details"] = f"Limit={m_tpd.group(1)}, Used={m_tpd.group(2)}, Requested={m_tpd.group(3)}"
            res["limit_val"] = int(m_tpd.group(1).replace(",", ""))
        return res

    m_rpd = re.search(
        r"requests per day \(RPD\)(?::\s*Limit\s*([\d,]+),\s*Used\s*([\d,]+),\s*Requested\s*([\d,]+))?",
        error_body,
        re.I,
    )
    if m_rpd:
        res["named_limit"] = "Requests Per Day (RPD)"
        if m_rpd.group(1):
            res["details"] = f"Limit={m_rpd.group(1)}, Used={m_rpd.group(2)}, Requested={m_rpd.group(3)}"
        return res

    # Generic Groq "Limit X, Used Y... try again in ..."
    m_generic = re.search(r"Limit\s*([\d,]+),\s*Used\s*([\d,]+)", error_body)
    if m_generic:
        res["named_limit"] = "Rate limit (TPD / TPM)"
        res["details"] = f"Limit={m_generic.group(1)}, Used={m_generic.group(2)}"
        return res

    # Gemini RESOURCE_EXHAUSTED
    if "RESOURCE_EXHAUSTED" in error_body or "quota" in error_body.lower():
        res["named_limit"] = "Gemini RESOURCE_EXHAUSTED (Quota Exceeded)"
        m_metric = re.search(r"quota metric '([^']+)'", error_body)
        if m_metric:
            res["details"] = f"Metric: {m_metric.group(1)}"
        return res

    return res


def compute_rolling_usage(
    records: list[dict[str, Any]], target_t: float, window_s: float = 60.0, inclusive: bool = True
) -> tuple[int, int]:
    """Calculate requests and tokens sent in a rolling window ending at target_t."""
    req_count = 0
    token_sum = 0
    for r in records:
        t = r.get("timestamp_wall") or r.get("timestamp_mono", 0.0)
        in_win = (target_t - window_s < t <= target_t) if inclusive else (target_t - window_s < t < target_t)
        if in_win:
            req_count += 1
            # Ground truth prompt tokens if available, else sum of section tokens
            tok = r.get("prompt_tokens")
            if tok is None or tok == 0:
                secs = r.get("sections") or {}
                tok = sum(s.get("tokens", 0) for s in secs.values())
            token_sum += tok
    return req_count, token_sum


def analyze_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce comprehensive analysis of usage records for a single run."""
    if not records:
        return {}

    # Sort records by timestamp
    records = sorted(records, key=lambda r: r.get("timestamp_wall") or r.get("timestamp_mono", 0.0))

    run_id = records[0].get("run_id", "unknown")
    models = list(dict.fromkeys(r.get("model", "") for r in records if r.get("model")))
    providers = list(dict.fromkeys(r.get("provider", "") for r in records if r.get("provider")))

    total_calls = len(records)
    successful_calls = [r for r in records if r.get("success")]
    rate_limited_calls = [r for r in records if r.get("status_code") == 429 or r.get("error_type") == "RateLimitError"]
    other_errors = [r for r in records if not r.get("success") and r not in rate_limited_calls]

    total_prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in successful_calls)
    total_completion_tokens = sum(r.get("completion_tokens") or 0 for r in successful_calls)
    total_reasoning_tokens = sum(r.get("reasoning_tokens") or 0 for r in successful_calls)
    total_cached_tokens = sum(r.get("cached_tokens") or 0 for r in successful_calls)
    total_tokens = total_prompt_tokens + total_completion_tokens

    # Role breakdown
    roles = ["planner", "worker", "verifier", "distiller"]
    role_stats: dict[str, dict[str, Any]] = {}
    for role in roles:
        r_calls = [r for r in records if r.get("role") == role]
        r_succ = [r for r in r_calls if r.get("success")]
        p_tok = sum(r.get("prompt_tokens") or 0 for r in r_succ)
        c_tok = sum(r.get("completion_tokens") or 0 for r in r_succ)
        re_tok = sum(r.get("reasoning_tokens") or 0 for r in r_succ)
        role_stats[role] = {
            "calls": len(r_calls),
            "success": len(r_succ),
            "rate_limited": sum(1 for r in r_calls if r.get("status_code") == 429),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "reasoning_tokens": re_tok,
            "total_tokens": p_tok + c_tok,
            "token_pct": (p_tok + c_tok) / total_tokens * 100 if total_tokens else 0.0,
        }

    # Rolling 60s windows
    peak_rpm = 0
    peak_tpm = 0
    peak_rpm_t = 0.0
    peak_tpm_t = 0.0

    for r in records:
        t = r.get("timestamp_wall") or r.get("timestamp_mono", 0.0)
        rpm, tpm = compute_rolling_usage(records, t, window_s=60.0, inclusive=True)
        if rpm > peak_rpm:
            peak_rpm = rpm
            peak_rpm_t = t
        if tpm > peak_tpm:
            peak_tpm = tpm
            peak_tpm_t = t

    # Header limits
    header_rpm_limit: int | None = None
    header_tpm_limit: int | None = None
    header_tpd_limit: int | None = None
    header_rpd_limit: int | None = None

    for r in records:
        hdrs = r.get("rate_limit_headers") or {}
        for k, v in hdrs.items():
            kl = k.lower()
            val_clean = str(v).split()[0].replace(",", "")
            if not val_clean.isdigit():
                continue
            ival = int(val_clean)
            if kl in ("x-ratelimit-limit-requests", "x-ratelimit-limit-requests-minute"):
                header_rpm_limit = ival
            elif kl in ("x-ratelimit-limit-tokens", "x-ratelimit-limit-tokens-minute"):
                header_tpm_limit = ival
            elif kl in ("x-ratelimit-limit-tokens-day",):
                header_tpd_limit = ival
            elif kl in ("x-ratelimit-limit-requests-day",):
                header_rpd_limit = ival

    # 429 details
    events_429: list[dict[str, Any]] = []
    for r in rate_limited_calls:
        t = r.get("timestamp_wall") or r.get("timestamp_mono", 0.0)
        # Usage strictly prior to this call
        prior_rpm, prior_tpm = compute_rolling_usage(records, t, window_s=60.0, inclusive=False)
        info = extract_limit_from_error(r.get("error_body"))
        events_429.append(
            {
                "step": r.get("step"),
                "role": r.get("role"),
                "timestamp_wall": r.get("timestamp_wall"),
                "latency_ms": r.get("latency_ms"),
                "named_limit": info["named_limit"],
                "limit_details": info["details"],
                "prior_60s_requests": prior_rpm,
                "prior_60s_tokens": prior_tpm,
                "error_body": r.get("error_body"),
                "headers": r.get("rate_limit_headers", {}),
            }
        )

    # Section breakdown per role
    section_breakdown: dict[str, dict[str, Any]] = {}
    for role in roles:
        r_calls = [r for r in records if r.get("role") == role]
        if not r_calls:
            continue
        tot_prompts_tok = []
        tot_prompts_chars = []
        sec_tok_lists: dict[str, list[int]] = {s: [] for s in SECTION_NAMES}
        sec_char_lists: dict[str, list[int]] = {s: [] for s in SECTION_NAMES}

        for r in r_calls:
            secs = r.get("sections") or {}
            c_sum = 0
            t_sum = 0
            for sname in SECTION_NAMES:
                sinfo = secs.get(sname, {})
                stok = sinfo.get("tokens", 0)
                sch = sinfo.get("chars", 0)
                sec_tok_lists[sname].append(stok)
                sec_char_lists[sname].append(sch)
                c_sum += sch
                t_sum += stok
            tot_prompts_tok.append(t_sum)
            tot_prompts_chars.append(c_sum)

        avg_prompt_tokens = sum(tot_prompts_tok) / len(tot_prompts_tok) if tot_prompts_tok else 0.0
        max_prompt_tokens = max(tot_prompts_tok) if tot_prompts_tok else 0
        avg_prompt_chars = sum(tot_prompts_chars) / len(tot_prompts_chars) if tot_prompts_chars else 0.0
        max_prompt_chars = max(tot_prompts_chars) if tot_prompts_chars else 0

        sec_summary = []
        for sname in SECTION_NAMES:
            toks = sec_tok_lists[sname]
            chars = sec_char_lists[sname]
            avg_tok = sum(toks) / len(toks) if toks else 0.0
            max_tok = max(toks) if toks else 0
            avg_ch = sum(chars) / len(chars) if chars else 0.0
            max_ch = max(chars) if chars else 0
            pct = (avg_tok / avg_prompt_tokens * 100) if avg_prompt_tokens > 0 else 0.0
            sec_summary.append(
                {
                    "name": sname,
                    "display": SECTION_DISPLAY.get(sname, sname),
                    "avg_tokens": avg_tok,
                    "max_tokens": max_tok,
                    "avg_chars": avg_ch,
                    "max_chars": max_ch,
                    "pct": pct,
                }
            )

        # Sort largest section first
        sec_summary.sort(key=lambda s: s["avg_tokens"], reverse=True)

        section_breakdown[role] = {
            "calls": len(r_calls),
            "avg_prompt_tokens": avg_prompt_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "avg_prompt_chars": avg_prompt_chars,
            "max_prompt_chars": max_prompt_chars,
            "sections": sec_summary,
        }

    # Step growth progression
    step_progression: list[dict[str, Any]] = []
    worker_steps = [r for r in records if r.get("role") == "worker" and r.get("success")]
    for r in records:
        secs = r.get("sections") or {}
        step_progression.append(
            {
                "step": r.get("step"),
                "role": r.get("role"),
                "success": r.get("success"),
                "prompt_tokens": r.get("prompt_tokens"),
                "est_tokens": sum(s.get("tokens", 0) for s in secs.values()),
                "history_tokens": secs.get("conversation_history", {}).get("tokens", 0),
                "dom_tokens": secs.get("latest_observation_dom", {}).get("tokens", 0),
                "page_text_tokens": secs.get("latest_observation_page_text", {}).get("tokens", 0),
                "tools_tokens": secs.get("tool_schemas", {}).get("tokens", 0),
                "system_tokens": secs.get("system_prompt", {}).get("tokens", 0),
                "memory_tokens": secs.get("working_memory", {}).get("tokens", 0),
            }
        )

    # Growth analysis
    history_growth = 0
    if len(worker_steps) >= 2:
        first_h = (worker_steps[0].get("sections") or {}).get("conversation_history", {}).get("tokens", 0)
        last_h = (worker_steps[-1].get("sections") or {}).get("conversation_history", {}).get("tokens", 0)
        history_growth = last_h - first_h

    # Overall duration
    t_start = records[0].get("timestamp_wall") or records[0].get("timestamp_mono", 0.0)
    t_end = records[-1].get("timestamp_wall") or records[-1].get("timestamp_mono", 0.0)
    duration_s = max(0.0, t_end - t_start)

    return {
        "run_id": run_id,
        "models": models,
        "providers": providers,
        "duration_s": duration_s,
        "total_calls": total_calls,
        "successful_calls": len(successful_calls),
        "rate_limited_calls": len(rate_limited_calls),
        "other_errors": len(other_errors),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_cached_tokens": total_cached_tokens,
        "total_tokens": total_tokens,
        "role_stats": role_stats,
        "peak_rpm": peak_rpm,
        "peak_tpm": peak_tpm,
        "peak_rpm_t": peak_rpm_t,
        "peak_tpm_t": peak_tpm_t,
        "header_rpm_limit": header_rpm_limit,
        "header_tpm_limit": header_tpm_limit,
        "header_tpd_limit": header_tpd_limit,
        "header_rpd_limit": header_rpd_limit,
        "events_429": events_429,
        "section_breakdown": section_breakdown,
        "step_progression": step_progression,
        "history_growth": history_growth,
    }


def format_report(analysis: dict[str, Any]) -> str:
    """Format single run analysis into human-readable text report."""
    if not analysis:
        return "No usage data found."

    lines: list[str] = []
    bar = "=" * 80
    subbar = "-" * 80

    lines.append(bar)
    lines.append("  AUTOWORK LLM USAGE & RATE-LIMIT DIAGNOSTICS REPORT")
    lines.append(f"  Run ID: {analysis['run_id']}")
    lines.append(f"  Model(s): {', '.join(analysis['models'])} | Provider(s): {', '.join(analysis['providers'])}")
    lines.append(f"  Duration: {analysis['duration_s']:.1f}s | Total Calls: {analysis['total_calls']}")
    lines.append(bar)
    lines.append("")

    # 1. Totals per run and per role
    lines.append("1. TOTALS PER RUN AND PER ROLE")
    lines.append(subbar)
    lines.append(
        f"Total Prompt Tokens:     {analysis['total_prompt_tokens']:,} (Ground Truth)\n"
        f"Total Completion Tokens: {analysis['total_completion_tokens']:,}\n"
        f"Total Reasoning Tokens:  {analysis['total_reasoning_tokens']:,}\n"
        f"Total Cached Tokens:     {analysis['total_cached_tokens']:,}\n"
        f"Grand Total Tokens:      {analysis['total_tokens']:,}"
    )
    lines.append("")
    lines.append(
        f"{'Role':<12} | {'Calls':<6} | {'Succ':<5} | {'429s':<5} | "
        f"{'Prompt Tok':<11} | {'Comp Tok':<9} | {'Total Tok':<10} | {'% Run':<6}"
    )
    lines.append("-" * 78)
    for role, st in analysis["role_stats"].items():
        lines.append(
            f"{role:<12} | {st['calls']:<6} | {st['success']:<5} | {st['rate_limited']:<5} | "
            f"{st['prompt_tokens']:<11,} | {st['completion_tokens']:<9,} | {st['total_tokens']:<10,} | "
            f"{st['token_pct']:>5.1f}%"
        )
    lines.append("")

    # 2. Rolling 60-second windows vs limits
    lines.append("2. ROLLING 60-SECOND WINDOWS (RPM & TPM)")
    lines.append(subbar)
    rpm_limit_str = f"{analysis['header_rpm_limit']:,}" if analysis["header_rpm_limit"] else "Not reported in headers"
    tpm_limit_str = f"{analysis['header_tpm_limit']:,}" if analysis["header_tpm_limit"] else "Not reported in headers"

    rpm_pct = (
        f"({analysis['peak_rpm'] / analysis['header_rpm_limit'] * 100:.1f}% of limit)"
        if analysis["header_rpm_limit"]
        else ""
    )
    tpm_pct = (
        f"({analysis['peak_tpm'] / analysis['header_tpm_limit'] * 100:.1f}% of limit)"
        if analysis["header_tpm_limit"]
        else ""
    )

    lines.append(
        f"Peak Requests / Min (RPM): {analysis['peak_rpm']} req/min  | Header Limit: {rpm_limit_str} {rpm_pct}"
    )
    lines.append(
        f"Peak Tokens / Min (TPM):   {analysis['peak_tpm']:,} tok/min | Header Limit: {tpm_limit_str} {tpm_pct}"
    )
    if analysis["header_tpd_limit"]:
        lines.append(f"Daily Token Limit (TPD):   {analysis['header_tpd_limit']:,} tok/day")
    lines.append("")

    # 3. 429 Details
    lines.append("3. RATE LIMIT 429 ERRORS & PRECEDING USAGE")
    lines.append(subbar)
    if not analysis["events_429"]:
        lines.append("No 429 rate limit errors recorded in this run. Provider served all requests without throttling.")
    else:
        lines.append(f"Total 429 events: {len(analysis['events_429'])}")
        for idx, ev in enumerate(analysis["events_429"], 1):
            lines.append(f"\n  [429 Event #{idx}] Step {ev['step']} ({ev['role']}):")
            lines.append(f"    Named Limit: {ev['named_limit']}")
            if ev["limit_details"]:
                lines.append(f"    Limit Details: {ev['limit_details']}")
            lines.append(
                f"    Preceding 60s Rolling Usage: {ev['prior_60s_requests']} reqs, "
                f"{ev['prior_60s_tokens']:,} tokens sent"
            )
            retry_after = ev["headers"].get("retry-after")
            if retry_after:
                lines.append(f"    Retry-After Header: {retry_after}s")
            rem_tok = ev["headers"].get("x-ratelimit-remaining-tokens")
            rem_req = ev["headers"].get("x-ratelimit-remaining-requests")
            reset_tok = ev["headers"].get("x-ratelimit-reset-tokens")
            if rem_tok or rem_req or reset_tok:
                lines.append(f"    Remaining Headers: tokens={rem_tok}, reqs={rem_req}, reset_tokens={reset_tok}")
            if ev.get("error_body"):
                body_preview = ev["error_body"].strip()[:180]
                lines.append(f"    Error Body: {body_preview}")
    lines.append("")

    # 4. Average and max prompt size per role with section breakdown
    lines.append("4. PROMPT SIZE & SECTION BREAKDOWN (Sorted Largest First)")
    lines.append(subbar)
    for role, bdown in analysis["section_breakdown"].items():
        lines.append(f"Role: {role.upper()} ({bdown['calls']} calls)")
        lines.append(
            f"  Avg Prompt Size: {bdown['avg_prompt_tokens']:.0f} tokens ({bdown['avg_prompt_chars']:.0f} chars) | "
            f"Max: {bdown['max_prompt_tokens']} tokens ({bdown['max_prompt_chars']} chars)"
        )
        lines.append(
            f"  {'Section Name':<30} | {'Avg Tokens':<11} | {'% Prompt':<8} | {'Max Tok':<8} | {'Avg Chars':<10}"
        )
        lines.append("  " + "-" * 76)
        for sec in bdown["sections"]:
            if sec["avg_tokens"] > 0 or sec["avg_chars"] > 0:
                lines.append(
                    f"  {sec['display']:<30} | {sec['avg_tokens']:>10.0f}  | {sec['pct']:>6.1f}%  | "
                    f"{sec['max_tokens']:>7}  | {sec['avg_chars']:>9.0f}"
                )
        lines.append("")

    # 5. How prompt size grows across steps
    lines.append("5. STEP-BY-STEP PROMPT SIZE PROGRESSION")
    lines.append(subbar)
    lines.append(
        f"{'Step':<5} | {'Role':<9} | {'PromptTok':<10} | {'HistoryTok':<11} | "
        f"{'DOMTok':<8} | {'PageTxtTok':<11} | {'SchemaTok':<10} | {'SysTok':<7}"
    )
    lines.append("-" * 80)
    for sp in analysis["step_progression"]:
        ptok = str(sp["prompt_tokens"]) if sp["prompt_tokens"] is not None else f"~{sp['est_tokens']}"
        lines.append(
            f"{sp['step']:<5} | {sp['role']:<9} | {ptok:<10} | {sp['history_tokens']:<11} | "
            f"{sp['dom_tokens']:<8} | {sp['page_text_tokens']:<11} | {sp['tools_tokens']:<10} | "
            f"{sp['system_tokens']:<7}"
        )
    lines.append("")
    lines.append("Progression Analysis:")
    if analysis["history_growth"] > 0:
        lines.append(
            f"  - History is accumulating: conversation history grew by +{analysis['history_growth']} tokens "
            f"from first to last worker step."
        )
    elif analysis["history_growth"] < 0:
        lines.append(
            f"  - Context compression is actively shedding tokens: conversation history decreased by "
            f"{abs(analysis['history_growth'])} tokens over the run."
        )
    else:
        lines.append("  - Conversation history remained steady (context compression kept old observations compressed).")
    lines.append("")

    # 6. Verdict and daily consumption
    lines.append("6. DIAGNOSTIC VERDICT & TOKEN REDUCTION TARGETS")
    lines.append(subbar)

    # Determine binding limit
    verdict_lines = []
    worker_bdown = analysis["section_breakdown"].get("worker", {})
    worker_secs = worker_bdown.get("sections", [])

    # Check which limit is binding
    is_tpm_binding = False
    is_rpm_binding = False
    is_daily_binding = False

    if analysis["events_429"]:
        first_429 = analysis["events_429"][0]["named_limit"]
        if "TPM" in first_429:
            is_tpm_binding = True
        elif "RPM" in first_429:
            is_rpm_binding = True
        elif "TPD" in first_429 or "Day" in first_429:
            is_daily_binding = True
    elif analysis["header_tpm_limit"] and analysis["peak_tpm"] > analysis["header_tpm_limit"] * 0.8:
        is_tpm_binding = True
    elif analysis["header_rpm_limit"] and analysis["peak_rpm"] > analysis["header_rpm_limit"] * 0.8:
        is_rpm_binding = True

    if is_tpm_binding or (not is_rpm_binding and not is_daily_binding):
        binding_name = "Tokens Per Minute (TPM)"
        binding_reason = (
            f"Peak TPM reached {analysis['peak_tpm']:,} tokens/min (header limit: {tpm_limit_str}), while RPM "
            f"peaked at only {analysis['peak_rpm']} req/min (well below limit {rpm_limit_str}). Because each worker "
            f"step sends ~{worker_bdown.get('avg_prompt_tokens', 0):.0f} tokens, "
            "2 calls in rapid succession exceed Groq's limit."
        )
    elif is_rpm_binding:
        binding_name = "Requests Per Minute (RPM)"
        binding_reason = f"Peak RPM reached {analysis['peak_rpm']} req/min, hitting the provider request rate limit."
    else:
        binding_name = "Tokens Per Day (TPD)"
        binding_reason = "Daily token budget was exhausted."

    verdict_lines.append(f"BINDING LIMIT: {binding_name}")
    verdict_lines.append(f"WHY: {binding_reason}")
    verdict_lines.append("")
    verdict_lines.append("TOP 3 PROMPT SECTIONS TO SHRINK (Worker Role):")

    top3 = [s for s in worker_secs if s["name"] != "other"][:3]
    total_savings_est = 0
    for rank, sec in enumerate(top3, 1):
        avg_tok = sec["avg_tokens"]
        # Potential savings: ~40-60% of section
        savings = int(avg_tok * 0.5)
        total_savings_est += savings
        if sec["name"] == "tool_schemas":
            rec = "Prune tool definitions or shorten verbose parameters/descriptions."
        elif sec["name"] == "latest_observation_dom":
            rec = "Reduce max interactive elements from 70 to 30 and omit empty attributes."
        elif sec["name"] == "latest_observation_page_text":
            rec = "Lower max text chars from 1,800 to 800 or summarize page text."
        elif sec["name"] == "system_prompt":
            rec = "Condense instructions into a concise checklist."
        elif sec["name"] == "conversation_history":
            rec = "Keep only 1 full observation instead of 2, or aggressively trim reasoning."
        else:
            rec = "Trim redundant formatting."
        verdict_lines.append(
            f"  {rank}. {sec['display']}: currently {avg_tok:.0f} tokens/call ({sec['pct']:.1f}% of prompt)."
        )
        verdict_lines.append(f"     -> Estimated savings: ~{savings:,} tokens/call. {rec}")

    verdict_lines.append(
        f"\nCombined estimated savings: ~{total_savings_est:,} tokens saved PER CALL "
        f"(reducing prompt from ~{worker_bdown.get('avg_prompt_tokens', 0):.0f} to "
        f"~{max(0, worker_bdown.get('avg_prompt_tokens', 0) - total_savings_est):.0f} tokens)."
    )

    lines.extend(verdict_lines)
    lines.append("")

    # Daily consumption
    lines.append("DAILY CONSUMPTION & CAPACITY ESTIMATE:")
    lines.append(subbar)
    tokens_per_run = analysis["total_tokens"]
    calls_per_run = analysis["total_calls"]
    lines.append(f"Measured per full run: {tokens_per_run:,} tokens across {calls_per_run} calls.")

    daily_token_limit = analysis["header_tpd_limit"] or 200_000  # Groq free tier default for many models is 200k
    runs_per_day = daily_token_limit / tokens_per_run if tokens_per_run > 0 else 0
    source_label = "provider header limit" if analysis["header_tpd_limit"] else "estimated Groq free tier (200k TPD)"

    lines.append(f"Daily Token Limit: {daily_token_limit:,} tokens ({source_label}).")
    lines.append(
        f"Run Capacity: ~{runs_per_day:.1f} full task runs fit into the daily quota before hitting rate limits."
    )
    lines.append(bar)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AutoWork LLM usage and rate limit diagnostics.")
    parser.add_argument("paths", nargs="*", help="Path(s) to llm_usage.jsonl files or run directories.")
    args = parser.parse_args()

    target_files: list[Path] = []
    if args.paths:
        for p_str in args.paths:
            p = Path(p_str)
            if p.is_file():
                target_files.append(p)
            elif p.is_dir():
                cand = p / "llm_usage.jsonl"
                if cand.is_file():
                    target_files.append(cand)
                else:
                    # check if directory has runs subdirectories
                    found = list(p.rglob("llm_usage.jsonl"))
                    target_files.extend(found)
    else:
        # Default: look in runs/ for the most recent llm_usage.jsonl
        runs_dir = Path("runs")
        if runs_dir.is_dir():
            all_usage = sorted(runs_dir.rglob("llm_usage.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
            if all_usage:
                target_files.append(all_usage[0])

    if not target_files:
        print(
            "No llm_usage.jsonl files found! Run an agent task or specify path to runs/<run_id>/llm_usage.jsonl",
            file=sys.stderr,
        )
        sys.exit(1)

    for fpath in target_files:
        records = parse_usage_file(fpath)
        analysis = analyze_run(records)
        print(format_report(analysis))


if __name__ == "__main__":
    main()
