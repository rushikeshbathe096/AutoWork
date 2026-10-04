"""Detecting an agent that is not making progress.

WHY in code: models in a loop often repeat the same failing action with full confidence. Two
independent signals are tracked:
  * repetition: the same tool + same args on the same page state (page fingerprint). A page that
    changed means the "same" action may legitimately do something new, so it doesn't count.
  * error streak: consecutive failed actions, even if they differ.
  * no progress: steps that succeed and differ but get nowhere (re-typing a search box with new
    words, revisiting pages already seen). The first two signals miss this; measured on live runs
    it was the main way the agent burned its whole step budget (~190k tokens per stuck run).
The first two escalate in two stages: a note to the model, then a question to the human. No
progress stops the run (status no_progress) instead, because asking again rarely helps a model
that is going in circles.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum

IGNORED_TOOLS = frozenset({"remember"})  # re-saving a fact is harmless


class Signal(Enum):
    OK = "ok"
    WARN = "warn"  # tell the model it is repeating itself / failing
    ESCALATE = "escalate"  # stop and ask the human
    STOP = "stop"  # end the run honestly instead of spending the rest of the budget


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


class ProgressTracker:
    """Counts consecutive steps that made no progress. A step makes progress when it reaches a URL not seen before
    in this run, adds or changes a working-memory fact, successfully changes state (a form submission that landed,
    a file written), or observes text from the environment it has not observed before (a new page state, the next
    part of a long page, a file). The agent's own echoes (fill read-backs, warnings) are not information.
    `stop_after=0` disables the stop."""

    def __init__(self, stop_after: int = 8):
        self.stop_after = stop_after
        self.warn_after = stop_after - 3 if stop_after > 3 else 0
        self.stalled = 0
        self._urls: set[str] = set()
        self._observations: set[str] = set()
        self._memory: str | None = None

    def observe(self, url: str, observation: str | None, memory: dict, state_changed: bool = False) -> Signal:
        """`observation`: the text an environment-reading tool returned this step, or None for other tools."""
        mem = json.dumps(memory, sort_keys=True)
        new_url = bool(url) and url not in self._urls
        self._urls.add(url)
        new_info = False
        if observation is not None:
            h = hashlib.sha256(observation.encode()).hexdigest()
            new_info = h not in self._observations
            self._observations.add(h)
        new_fact = self._memory is not None and mem != self._memory
        self._memory = mem
        self.stalled = 0 if (new_url or new_info or new_fact or state_changed) else self.stalled + 1
        if self.stop_after and self.stalled >= self.stop_after:
            return Signal.STOP
        return Signal.WARN if self.warn_after and self.stalled == self.warn_after else Signal.OK

    @staticmethod
    def why(n: int) -> str:
        return (
            f"no progress in the last {n} steps: no new URL, no new memory fact, no successful state-changing "
            "action, and no new information in observations"
        )
