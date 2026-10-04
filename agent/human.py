"""Ways for the agent to reach a human. The agent only sees the `ask(kind, question, options)` interface."""

from __future__ import annotations

import re
import secrets
import threading


class CLIHuman:
    def ask(self, kind, question, options=None):
        print("\n" + "=" * 70 + f"\n[{kind.upper()} NEEDED]\n{question}")
        if options:
            print("Options: " + " | ".join(options))
        if kind == "approval":
            a = input("Approve? [y/N] (optionally add a comment after a space): ").strip()
            return {"approved": a.lower().startswith("y"), "comment": a[1:].strip()}
        return {"answer": input("Your answer: ").strip()}


class ScriptedHuman:
    """For evals: answers from a script. `approvals` is a bool or a list of regex->bool rules;
    `answers` maps a regex on the question to an answer."""

    def __init__(
        self,
        approve: bool | list = False,
        answers: dict[str, str] | None = None,
        default_answer: str = "I don't have more information. Use your best judgment; if unsafe, stop.",
    ):
        self.approve, self.answers, self.default = approve, answers or {}, default_answer
        self.log: list[dict] = []

    def ask(self, kind, question, options=None):
        if kind == "approval":
            ok = self.approve
            if isinstance(self.approve, list):
                ok = next((v for pat, v in self.approve if re.search(pat, question, re.I)), False)
            resp = {"approved": bool(ok), "comment": "" if ok else "Not approved for this test."}
        else:
            ans = next((a for pat, a in self.answers.items() if re.search(pat, question, re.I)), self.default)
            resp = {"answer": ans}
        self.log.append({"kind": kind, "question": question, **resp})
        return resp


class WebHuman:
    """Blocks the agent thread until the web UI posts an answer (or timeout)."""

    def __init__(self, emit, timeout_s: int = 900):
        self.emit, self.timeout = emit, timeout_s
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()

    def ask(self, kind, question, options=None):
        qid = secrets.token_urlsafe(16)  # unguessable: 128 bits
        ev = threading.Event()
        with self._lock:
            self._pending[qid] = {"event": ev, "answer": None}
        self.emit("waiting_for_human", {"qid": qid, "kind": kind, "question": question, "options": options or []})
        got = ev.wait(self.timeout)
        with self._lock:
            ans = self._pending.pop(qid)["answer"]
        if not got or ans is None:
            return (
                {"approved": False, "comment": "No response (timed out)"}
                if kind == "approval"
                else {"answer": "No response from user (timed out). Do not take risky actions; finish with needs_user."}
            )
        return ans

    def respond(self, qid: str, answer: dict) -> bool:
        """Single use: the first answer for a qid wins; later answers (replays, double clicks) are rejected."""
        with self._lock:
            p = self._pending.get(qid)
            if not p or p["answer"] is not None:
                return False
            p["answer"] = answer
            p["event"].set()
            return True

    def pending(self) -> list[str]:
        with self._lock:
            return list(self._pending)

    def cancel_all(self, reason: str) -> None:
        """Answer every open question at once (the user pressed Stop): approvals are declined."""
        with self._lock:
            open_ = [(qid, p) for qid, p in self._pending.items() if p["answer"] is None]
            for _, p in open_:
                p["answer"] = {"approved": False, "comment": reason, "answer": reason}
                p["event"].set()
