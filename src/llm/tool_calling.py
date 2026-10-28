"""Provider-neutral contracts for LLM tool calling.

The agent loop and the tool layer only see these types. Provider adapters
(see openai_provider.py) translate them to and from SDK-specific payloads.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolSpec(_Frozen):
    """What the model is told about a tool: name, purpose and JSON-schema arguments."""
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str = Field(min_length=1)
    parameters: dict[str, Any]


class ChatMessage(_Frozen):
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(_Frozen):
    """A model-requested call. arguments stays raw JSON text until validated."""
    call_id: str = Field(min_length=1)
    name: str
    arguments: str


class AssistantTurn(_Frozen):
    """One model response: optional text and zero or more tool calls."""
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolResultMessage(_Frozen):
    """Observation returned to the model for one tool call."""
    call_id: str = Field(min_length=1)
    name: str
    content: str


ConversationItem = ChatMessage | AssistantTurn | ToolResultMessage


class ToolCallingLLMProvider(Protocol):
    """Provider-neutral interface for one tool-aware model step."""

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolSpec],
    ) -> AssistantTurn:
        ...
