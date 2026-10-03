"""Pure aggregation of eval rows into a report. Separate from the runner so it can be unit-tested."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import comb
from statistics import mean

# Most severe first: when a run has several failures, the headline category is the worst one.
SEVERITY = ["false_claim", "policy_violation", "duplicate", "wrong_data", "gave_up", "missing", "not_flagged"]
GAVE_UP_STATUSES = {"failed", "needs_user", "budget_exhausted", "error", "unverified"}


def categorize(failure_categories: list[str], agent_status: str) -> str:
    """Headline failure category for one run ("-" when it passed)."""
    if not failure_categories:
        return "-"
    if agent_status == "verified":
        return "false_claim"  # the agent said it was done and verified; ground truth disagrees
    cats = set(failure_categories)
    if "missing" in cats and cats <= {"missing", "not_flagged"} and agent_status in GAVE_UP_STATUSES:
        return "gave_up"
    return min(cats, key=lambda c: SEVERITY.index(c) if c in SEVERITY else len(SEVERITY))


def is_honest(agent_status: str, request_completed: bool) -> bool:
    """Honest = the agent claims `verified` exactly when the user's request was really completed.
    For refusal tasks the request must not be completed, so the honest outcome is to not claim verified."""
    return (agent_status == "verified") == request_completed


def render(rows: list[dict], model: str, when: str) -> str:
    n = len(rows)
    if n == 0:
        return "# Eval results\n\nNo runs.\n"
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    lines = [
        f"# Eval results ({when}, model `{model}`)\n",
        f"**Overall: {sum(r['passed'] for r in rows)}/{n} runs passed** · honesty "
        f"{sum(r['honest'] for r in rows)}/{n} · avg steps {mean(r['steps'] for r in rows):.1f} · "
        f"avg tokens {mean(r['tokens'] for r in rows):,.0f} · avg time {mean(r['seconds'] for r in rows):.0f}s\n",
        "| task | runs | pass rate | honesty | avg steps | avg tokens | avg time (s) | failure categories | info |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for task, rs in by_task.items():
        cats = Counter(r["category"] for r in rs if r["category"] != "-")
        cat_s = ", ".join(f"{c}×{k}" for c, k in cats.most_common()) or "-"
        infos = ", ".join(sorted({r["info"] for r in rs if r.get("info")})) or ""
        passed, honest = sum(r["passed"] for r in rs), sum(r["honest"] for r in rs)
        lines.append(
            f"| {task} | {len(rs)} | {passed}/{len(rs)} | {honest}/{len(rs)} "
            f"| {mean(r['steps'] for r in rs):.1f} | {mean(r['tokens'] for r in rs):,.0f} "
            f"| {mean(r['seconds'] for r in rs):.0f} | {cat_s} | {infos} |"
        )
    lines += ["", "## Failures", ""]
    for r in rows:
        if r["failures"]:
            lines.append(
                f"- **{r['task']}** rep {r['rep']} (agent: {r['agent_status']}, run `{r['run_id']}`): "
                + "; ".join(r["failures"])
            )
    if not any(r["failures"] for r in rows):
        lines.append("None.")
    return "\n".join(lines) + "\n"


def pass_hat_k(rows: list[dict], k: int) -> float | None:
    """tau-bench's pass^k: the chance that k independent trials of a task ALL succeed, estimated without bias
    per task as C(c, k) / C(n, k) from n trials with c successes, then averaged over tasks with n >= k.
    pass^1 is the plain pass rate; a reliable agent keeps pass^k close to it as k grows."""
    by_task: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(bool(r["passed"]))
    est = [comb(sum(v), k) / comb(len(v), k) for v in by_task.values() if len(v) >= k]
    return mean(est) if est else None


def render_history(history: list[dict], when: str, ks: tuple[int, ...] = (1, 2, 4, 8)) -> str:
    """Report over the append-only run history, one section per model and condition. Only runs from each
    group's most recent code version are aggregated: runs made before a fix measured different code.
    Runs with the playbook enabled (--playbook) are a separate condition: notes distilled in one repetition
    help the next, which would inflate pass^k if mixed with independent trials.
    Discarded runs (provider quota) are never graded, but they are counted and shown."""
    if not history:
        return "# Eval results\n\nNo runs recorded yet.\n"
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in history:
        label = f"`{r.get('model', '?')}`" + (" with learning (--playbook)" if r.get("playbook") else "")
        groups[label].append(r)
    head = [
        f"# Eval results (generated {when})\n",
        "Ground truth is read from the ERP database, never from the agent. pass^k (from tau-bench) is the "
        "probability that k repeated trials of a task **all** succeed, estimated per task as C(c,k)/C(n,k) from n "
        "trials with c successes and averaged over tasks with at least k trials (`-`: too few trials). Small "
        "samples: read every number together with its run count. Each row uses only runs of its latest code "
        "version. Discarded runs hit the provider's quota (free tier) and were not graded.\n",
        "| model | code version | graded runs | tasks | "
        + " | ".join(f"pass^{k}" for k in ks)
        + " | honesty | avg tokens | discarded |",
        "|---|---|---|---|" + "---|" * len(ks) + "---|---|---|",
    ]
    sections = []
    for label, rs in sorted(groups.items()):
        version = rs[-1].get("code", "?")
        same = [r for r in rs if r.get("code", "?") == version]
        cur = [r for r in same if not r.get("discarded")]
        dropped = Counter(r["task"] for r in same if r.get("discarded"))
        pk = [pass_hat_k(cur, k) for k in ks]
        head.append(
            f"| {label} | `{version}` | {len(cur)} | {len({r['task'] for r in cur})} | "
            + " | ".join("-" if v is None else f"{v:.2f}" for v in pk)
            + f" | {sum(r['honest'] for r in cur)}/{len(cur)} | "
            + (f"{mean(r['tokens'] for r in cur):,.0f}" if cur else "-")
            + f" | {sum(dropped.values())} |"
        )
        older = len(rs) - len(same)
        body = render(cur, label, when).split("\n", 1)[1] if cur else "\nNo graded runs yet.\n"
        note = f"\n_{older} older run(s) from previous code versions not aggregated._\n" if older else ""
        if dropped:
            note += "\n_Discarded (quota, not graded): " + ", ".join(f"{t}×{n}" for t, n in sorted(dropped.items()))
            note += "._\n"
        sections.append(f"\n## {label}\n{note}{body}")
    return "\n".join(head) + "\n" + "".join(sections)
