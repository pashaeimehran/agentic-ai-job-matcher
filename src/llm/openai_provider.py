from __future__ import annotations

from typing import Any, Sequence, TypeVar, cast

from openai import OpenAI
from pydantic import BaseModel

from src.llm.provider import LLMMessage, LLMProviderError
from src.llm.tool_calling import (AssistantTurn, ChatMessage, ConversationItem, ToolCall,
                                  ToolResultMessage, ToolSpec)


T = TypeVar("T", bound=BaseModel)


class OpenAIProvider:
    """OpenAI implementation of the provider-neutral structured LLM interface."""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model

    def generate(
        self,
        messages: Sequence[LLMMessage],
        schema: type[T],
    ) -> T:
        response = self._client.responses.parse(
            model=self._model,
            input=list(messages),
            text_format=schema,
        )

        parsed = response.output_parsed
        if parsed is None:
            raise LLMProviderError("OpenAI returned no parsed structured output.")

        return cast(T, parsed)

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolSpec],
    ) -> AssistantTurn:
        """One tool-aware Responses API step; arguments are returned unvalidated."""
        # strict=False: Pydantic validation in the tool layer is the authority.
        response = self._client.responses.create(
            model=self._model,
            input=_to_openai_input(messages),
            tools=[{"type": "function", "name": t.name, "description": t.description,
                    "parameters": t.parameters, "strict": False} for t in tools],
        )
        calls = [ToolCall(call_id=item.call_id, name=item.name, arguments=item.arguments or "")
                 for item in response.output if getattr(item, "type", None) == "function_call"]
        text = response.output_text or None
        if text is None and not calls:
            raise LLMProviderError("OpenAI returned neither text nor tool calls.")
        return AssistantTurn(text=text, tool_calls=calls)


def _to_openai_input(messages: Sequence[ConversationItem]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            items.append({"role": message.role, "content": message.content})
        elif isinstance(message, AssistantTurn):
            if message.text:
                items.append({"role": "assistant", "content": message.text})
            items.extend({"type": "function_call", "call_id": call.call_id,
                          "name": call.name, "arguments": call.arguments}
                         for call in message.tool_calls)
        elif isinstance(message, ToolResultMessage):
            items.append({"type": "function_call_output", "call_id": message.call_id,
                          "output": message.content})
        else:
            raise TypeError(f"Unsupported conversation item: {type(message).__name__}")
    return items
