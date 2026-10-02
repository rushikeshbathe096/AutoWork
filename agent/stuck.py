"""Detecting an agent that is not making progress.

WHY in code: models in a loop often repeat the same failing action with full confidence. Two
independent signals are tracked:
  * repetition: the same tool + same args on the same page state (page fingerprint). A page that
    changed means the "same" action may legitimately do something new, so it doesn't count.
  * error streak: consecutive failed actions, even if they differ.
Each escalates in two stages: first a note to the model, then a question to the human.
"""

from __future__ import annotations

import json
from enum import Enum

IGNORED_TOOLS = frozenset({"remember"})  # re-saving a fact is harmless


class Signal(Enum):
    OK = "ok"
    WARN = "warn"  # tell the model it is repeating itself / failing
    ESCALATE = "escalate"  # stop and ask the human


class RepetitionDetector:
    def __init__(self, warn_at: int = 3, escalate_at: int = 4):
        self.warn_at, self.escalate_at = warn_at, escalate_at
        self._counts: dict[str, int] = {}

    @staticmethod
    def signature(tool: str, args: dict, page_fingerprint: str) -> str:
        return f"{tool}|{json.dumps(args, sort_keys=True)}|{page_fingerprint}"

    def observe(self, tool: str, args: dict, page_fingerprint: str) -> tuple[Signal, int]:
        """Record an action about to be taken. Returns the signal and how often it has been seen."""
        if tool in IGNORED_TOOLS:
            return Signal.OK, 0
        sig = self.signature(tool, args, page_fingerprint)
        n = self._counts[sig] = self._counts.get(sig, 0) + 1
        if n >= self.escalate_at:
            self._counts[sig] = 0  # after a human weighs in, give the action a fresh chance
            return Signal.ESCALATE, n
        return (Signal.WARN if n == self.warn_at else Signal.OK), n


class ErrorStreak:
    def __init__(self, warn_at: int = 3, escalate_at: int = 6):
        self.warn_at, self.escalate_at = warn_at, escalate_at
        self.count = 0

    def observe(self, ok: bool) -> Signal:
        self.count = 0 if ok else self.count + 1
        if self.count >= self.escalate_at:
            return Signal.ESCALATE
        return Signal.WARN if self.count == self.warn_at else Signal.OK

    def reset(self) -> None:
        self.count = 0
