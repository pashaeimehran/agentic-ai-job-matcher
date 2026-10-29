from __future__ import annotations

from typing import Any, Sequence, TypeVar, cast

from openai import OpenAI
from pydantic import BaseModel

from src.llm.provider import LLMMessage, LLMProviderError, LLMUsage
from src.llm.tool_calling import (AssistantTurn, ChatMessage, ConversationItem, ToolCall,
                                  ToolResultMessage, ToolSpec)


T = TypeVar("T", bound=BaseModel)


def usage_from_response(response: Any) -> LLMUsage | None:
    """Token counts from a Responses API result; None when the provider reports no usage."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    details = getattr(usage, "input_tokens_details", None)
    return LLMUsage(input_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                    cached_input_tokens=getattr(details, "cached_tokens", 0) or 0)


class OpenAIProvider:
    """OpenAI implementation of the provider-neutral structured LLM interface.

    The *_with_usage variants also return token usage for observability and cost
    accounting; generate() and complete() keep the provider-neutral contracts.
    """

    provider_name = "openai"

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini",
                 max_output_tokens: int | None = None) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_parameters(self) -> dict[str, Any]:
        """Request parameters that change the output; part of the response-cache key."""
        return {"max_output_tokens": self._max_output_tokens}

    def _limits(self) -> dict[str, Any]:
        return {"max_output_tokens": self._max_output_tokens} if self._max_output_tokens else {}

    def generate_with_usage(self, messages: Sequence[LLMMessage], schema: type[T]) -> tuple[T, LLMUsage | None]:
        response = self._client.responses.parse(
            model=self._model,
            input=list(messages),
            text_format=schema,
            **self._limits(),
        )

        parsed = response.output_parsed
        if parsed is None:
            raise LLMProviderError("OpenAI returned no parsed structured output.")

        return cast(T, parsed), usage_from_response(response)

    def generate(
        self,
        messages: Sequence[LLMMessage],
        schema: type[T],
    ) -> T:
        return self.generate_with_usage(messages, schema)[0]

    def complete_with_usage(self, messages: Sequence[ConversationItem],
                            tools: Sequence[ToolSpec]) -> tuple[AssistantTurn, LLMUsage | None]:
        """One tool-aware Responses API step; arguments are returned unvalidated."""
        # strict=False: Pydantic validation in the tool layer is the authority.
        response = self._client.responses.create(
            model=self._model,
            input=_to_openai_input(messages),
            tools=[{"type": "function", "name": t.name, "description": t.description,
                    "parameters": t.parameters, "strict": False} for t in tools],
            **self._limits(),
        )
        calls = [ToolCall(call_id=item.call_id, name=item.name, arguments=item.arguments or "")
                 for item in response.output if getattr(item, "type", None) == "function_call"]
        text = response.output_text or None
        if text is None and not calls:
            raise LLMProviderError("OpenAI returned neither text nor tool calls.")
        return AssistantTurn(text=text, tool_calls=calls), usage_from_response(response)

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolSpec],
    ) -> AssistantTurn:
        return self.complete_with_usage(messages, tools)[0]


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
