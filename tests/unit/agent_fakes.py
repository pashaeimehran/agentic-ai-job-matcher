"""Deterministic stand-ins for a tool-calling model (no network)."""
from __future__ import annotations

import json
from typing import Callable

from src.llm.tool_calling import AssistantTurn, ToolCall, ToolResultMessage


def call(name: str, arguments: dict | str, call_id: str | None = None) -> ToolCall:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(call_id=call_id or f"call-{name}-{abs(hash(raw)) % 10_000}", name=name, arguments=raw)


def tools(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(tool_calls=list(calls))


def answer(text: str) -> AssistantTurn:
    return AssistantTurn(text=text)


Step = AssistantTurn | Callable[[list], AssistantTurn]


class ScriptedProvider:
    """Replays scripted turns; a callable step can inspect prior observations."""

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.calls: list[list] = []
        self.tool_names: list[list[str]] = []

    def complete(self, messages, tools):
        self.calls.append(list(messages))
        self.tool_names.append([t.name for t in tools])
        if not self.steps:
            raise AssertionError("ScriptedProvider ran out of steps")
        step = self.steps.pop(0)
        return step(list(messages)) if callable(step) else step


def observations(messages: list) -> list[dict]:
    return [json.loads(m.content) for m in messages if isinstance(m, ToolResultMessage)]
