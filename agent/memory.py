"""Two kinds of memory.

WorkingMemory - facts discovered during THIS run (amounts, ids, what has already been done).
    Rendered into every prompt, so facts survive even after old observations are elided from
    the context window. This is what lets the agent run long tasks on a small context budget.

Playbook - lessons distilled from previous *verified-successful* runs (where things live, login
    flows, input format quirks, pitfalls). Injected into future runs as hints. This is the
    "turn a successful experiment into a reusable capability" loop.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path


class WorkingMemory:
    def __init__(self):
        self.facts: dict[str, dict] = {}

    def remember(self, key: str, value: str, step: int) -> str:
        key = key.strip()[:60]
        prev = self.facts.get(key)
        self.facts[key] = {"value": str(value)[:500], "step": step}
        return f"Stored {key!r}" + (f" (was {prev['value']!r})" if prev else "")

    def render(self) -> str:
        if not self.facts:
            return "(nothing stored yet)"
        return "\n".join(f"- {k}: {v['value']}" for k, v in self.facts.items())

    def as_dict(self) -> dict:
        return {k: v["value"] for k, v in self.facts.items()}


class Playbook:
    _lock = threading.Lock()

    def __init__(self, path: Path, max_notes: int = 40):
        self.path = path
        self.max_notes = max_notes

    def load(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []

    def render(self, limit: int = 15) -> str:
        notes = self.load()[-limit:]
        return "\n".join(f"- {n['note']}" for n in notes)

    def add(self, notes: list[str], run_id: str):
        with self._lock:
            existing = self.load()
            seen = {n["note"].lower() for n in existing}
            for note in notes:
                note = note.strip()
                if note and note.lower() not in seen and len(note) < 300:
                    existing.append({"note": note, "run": run_id, "at": datetime.now().isoformat(timespec="seconds")})
                    seen.add(note.lower())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(existing[-self.max_notes:], indent=2))
