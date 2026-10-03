"""Structural interfaces between the agent loop and its collaborators.

WHY Protocols: the loop only needs a handful of methods from each collaborator. Typing against
these (instead of the concrete classes) lets tests use a scripted LLM / scripted human, and lets
a different provider or browser driver drop in without touching agent/core.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page

    from .browser import Snapshot
    from .llm import LLMResponse
    from .vault import SiteCredential

Message = dict[str, Any]  # one OpenAI-format chat message
Emit = Callable[[str, dict], None]


class LLM(Protocol):
    model: str
    stats: dict[str, int]
    on_retry: Callable[[str], None] | None

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        require_tool: bool = False,
        json_mode: bool = False,
        role: str = "worker",
        step: int = 0,
        **kw: Any,
    ) -> LLMResponse: ...


class Human(Protocol):
    def ask(self, kind: str, question: str, options: list[str] | None = None) -> dict:
        """kind 'approval' -> {'approved': bool, 'comment': str}; 'clarification' -> {'answer': str}"""
        ...


class Browser(Protocol):
    """What ToolBox and the agent loop need from a browser session."""

    last: Snapshot | None
    page: Page
    context: BrowserContext

    def goto(self, url: str) -> Snapshot: ...
    def click(self, eid: int) -> Snapshot: ...
    def fill(self, fields: list[dict]) -> str: ...
    def back(self) -> Snapshot: ...
    def read(self, offset: int = 0, chars: int = 6000) -> str: ...
    def login(self, cred: SiteCredential) -> Snapshot: ...
    def snapshot(self, label: str = "") -> Snapshot: ...
    def preapprove(self, key: str) -> None: ...
    def close(self) -> None: ...
