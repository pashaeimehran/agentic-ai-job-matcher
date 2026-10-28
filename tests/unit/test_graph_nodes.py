"""LangGraph routing functions and nodes tested in isolation (no compiled graph, no live API)."""
import json

import pytest

from src.graph.nodes import (GraphNodes, close_dangling_calls, failure_status, route_after_agent,
                             route_after_tools)
from src.graph.state import GraphConfig
from src.llm.tool_calling import AssistantTurn, ChatMessage, ToolResultMessage
from src.tools import default_registry
from src.tools.context import build_offline_context
from tests.unit.agent_fakes import ScriptedProvider, answer, call, tools


@pytest.fixture
def ctx():
    return build_offline_context()


def nodes(ctx, steps=(), **config):
    return GraphNodes(ScriptedProvider(list(steps)), default_registry(), ctx, GraphConfig(**config))


def prepared(n, request="Find AI jobs"):
    return {"request": request, "run_id": "r1", **n.prepare({"request": request, "run_id": "r1"})}


# --- Routing -------------------------------------------------------------------

@pytest.mark.parametrize("state, expected", [
    ({"pending_calls": [call("search_jobs", {"query": "AI"})]}, "execute_tools"),
    ({"pending_calls": []}, "finalize"),
    ({}, "finalize"),
    ({"status": "provider_error", "pending_calls": []}, "fail"),
])
def test_route_after_agent(state, expected):
    assert route_after_agent(state) == expected


@pytest.mark.parametrize("state, expected", [
    ({"model_turns": 1}, "agent"),
    ({"model_turns": 3}, "fail"),                                   # turn budget exhausted
    ({"model_turns": 1, "status": "max_tool_calls_exceeded"}, "fail"),
    ({"model_turns": 1, "status": "invalid_tool_requests"}, "fail"),
])
def test_route_after_tools(state, expected):
    assert route_after_tools(state, GraphConfig(max_model_turns=3)) == expected


def test_failure_status_defaults_to_turn_limit():
    assert failure_status({}) == "max_model_turns_exceeded"
    assert failure_status({"status": "provider_error"}) == "provider_error"


def test_recursion_limit_exceeds_routed_bound():
    cfg = GraphConfig(max_model_turns=4)
    # prepare + 4 * (agent + execute_tools) + fail = 10 supersteps
    assert cfg.recursion_limit() > 1 + 2 * cfg.max_model_turns + 1


# --- Nodes ---------------------------------------------------------------------

def test_prepare_starts_new_thread_with_system_and_user_messages(ctx):
    update = nodes(ctx).prepare({"request": "hi", "run_id": "r1"})
    assert [(m.role, m.content) for m in update["messages"]][1:] == [("user", "hi")]
    assert update["messages"][0].role == "system"
    assert update["model_turns"] == 0 and update["tool_calls"] == [] and update["status"] is None


def test_prepare_continues_thread_and_resets_run_counters(ctx):
    history = [ChatMessage(role="system", content="s"), ChatMessage(role="user", content="first"),
               AssistantTurn(text="answer")]
    update = nodes(ctx).prepare({"request": "second", "run_id": "r2", "messages": history,
                                 "model_turns": 5, "status": "completed", "final_answer": "answer"})
    assert update["messages"][:3] == history and update["messages"][-1].content == "second"
    assert update["model_turns"] == 0 and update["status"] is None and update["final_answer"] is None


def test_close_dangling_calls_answers_unexecuted_calls():
    turn = AssistantTurn(tool_calls=[call("search_jobs", {"query": "a"}, "c1"),
                                     call("search_cv", {"query": "b"}, "c2")])
    messages = [ChatMessage(role="user", content="x"), turn,
                ToolResultMessage(call_id="c1", name="search_jobs", content="{}")]
    closed = close_dangling_calls(messages)
    assert closed[:3] == messages and len(closed) == 4
    assert closed[3].call_id == "c2" and json.loads(closed[3].content)["status"] == "not_executed"


def test_agent_node_records_tool_request(ctx):
    n = nodes(ctx, [tools(call("search_cv", {"query": "Python"}, "c1"))])
    update = n.agent(prepared(n))
    assert update["model_turns"] == 1 and [c.name for c in update["pending_calls"]] == ["search_cv"]
    assert update["path"][-1] == "agent" and "status" not in update


def test_agent_node_converts_provider_exception(ctx):
    class Broken:
        def complete(self, messages, tools):
            raise ConnectionError("api key sk-secret")

    n = GraphNodes(Broken(), default_registry(), ctx)
    update = n.agent(prepared(n))
    assert update["status"] == "provider_error" and "sk-secret" not in update["error"]
    assert route_after_agent(update) == "fail"


def test_execute_tools_node_uses_shared_registry(ctx):
    n = nodes(ctx)
    state = {**prepared(n), "model_turns": 1,
             "pending_calls": [call("get_job_details", {"job_id": "fixture-01"}, "c1")]}
    update = n.execute_tools(state)
    direct = default_registry().execute("get_job_details", {"job_id": "fixture-01"}, ctx)
    record = update["tool_calls"][0]
    assert record.outcome.data == direct.data and record.step == 1
    assert isinstance(update["messages"][-1], ToolResultMessage) and update["pending_calls"] == []


def test_execute_tools_node_enforces_tool_budget(ctx):
    n = nodes(ctx, max_tool_calls=1)
    state = {**prepared(n), "model_turns": 1,
             "pending_calls": [call("get_job_details", {"job_id": "fixture-01"}, f"c{i}") for i in range(3)]}
    update = n.execute_tools(state)
    assert len(update["tool_calls"]) == 1 and update["status"] == "max_tool_calls_exceeded"


def test_execute_tools_node_counts_consecutive_invalid_turns(ctx):
    n = nodes(ctx, max_invalid_tool_turns=2)
    bad = {**prepared(n), "model_turns": 1, "pending_calls": [call("find_jobs", {}, "c1")]}
    first = n.execute_tools(bad)
    assert first["invalid_tool_turns"] == 1 and "status" not in first
    second = n.execute_tools({**bad, **first, "pending_calls": [call("search_jobs", {"q": 1}, "c2")]})
    assert second["invalid_tool_turns"] == 2 and second["status"] == "invalid_tool_requests"
    good = n.execute_tools({**bad, **first, "pending_calls": [call("search_jobs", {"query": "AI"}, "c3")]})
    assert good["invalid_tool_turns"] == 0


def test_execution_failure_is_not_an_invalid_request(ctx):
    n = nodes(ctx, max_invalid_tool_turns=1)
    state = {**prepared(n), "model_turns": 1,
             "pending_calls": [call("get_job_details", {"job_id": "no-such-job"}, "c1")]}
    update = n.execute_tools(state)
    assert update["tool_calls"][0].error_code == "not_found" and update["tool_calls"][0].arguments_valid
    assert update["invalid_tool_turns"] == 0 and "status" not in update


def test_finalize_node_flags_ungrounded_scores(ctx):
    n = nodes(ctx)
    state = {**prepared(n), "messages": [AssistantTurn(text="fixture-01 is a 97/100 fit.")]}
    update = n.finalize(state)
    assert update["status"] == "completed" and update["ungrounded_scores"] == ["97"]


def test_fail_node_reports_turn_limit_when_no_status(ctx):
    n = nodes(ctx, max_model_turns=2)
    update = n.fail({**prepared(n), "model_turns": 2})
    assert update["status"] == "max_model_turns_exceeded" and "2" in update["error"]
    assert update["final_answer"] is None
