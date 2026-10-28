from types import SimpleNamespace

import pytest

from src.llm.openai_provider import OpenAIProvider
from src.llm.provider import LLMProviderError
from src.llm.tool_calling import AssistantTurn, ChatMessage, ToolCall, ToolResultMessage, ToolSpec


class FakeResponses:
    def __init__(self, response) -> None:
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def provider_with(output, output_text=""):
    responses = FakeResponses(SimpleNamespace(output=output, output_text=output_text))
    return OpenAIProvider(SimpleNamespace(responses=responses), model="test-model"), responses


SPEC = ToolSpec(name="search_jobs", description="Search", parameters={"type": "object", "properties": {}})


def test_complete_translates_tools_history_and_function_calls():
    fc = SimpleNamespace(type="function_call", call_id="c2", name="search_jobs", arguments='{"query":"AI"}')
    provider, responses = provider_with([fc])
    history = [ChatMessage(role="user", content="find"),
               AssistantTurn(text=None, tool_calls=[ToolCall(call_id="c1", name="search_cv", arguments="{}")]),
               ToolResultMessage(call_id="c1", name="search_cv", content='{"status":"ok"}')]
    turn = provider.complete(history, [SPEC])
    assert turn == AssistantTurn(text=None, tool_calls=[ToolCall(call_id="c2", name="search_jobs",
                                                                 arguments='{"query":"AI"}')])
    assert responses.kwargs["model"] == "test-model"
    assert responses.kwargs["tools"] == [{"type": "function", "name": "search_jobs", "description": "Search",
                                          "parameters": {"type": "object", "properties": {}}, "strict": False}]
    assert responses.kwargs["input"] == [
        {"role": "user", "content": "find"},
        {"type": "function_call", "call_id": "c1", "name": "search_cv", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": '{"status":"ok"}'}]


def test_complete_returns_text_answer():
    provider, _ = provider_with([SimpleNamespace(type="message")], output_text="Hello")
    assert provider.complete([ChatMessage(role="user", content="hi")], []) == AssistantTurn(text="Hello")


def test_complete_rejects_empty_response():
    provider, _ = provider_with([])
    with pytest.raises(LLMProviderError):
        provider.complete([ChatMessage(role="user", content="hi")], [])
