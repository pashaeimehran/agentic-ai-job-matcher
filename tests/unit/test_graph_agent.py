"""End-to-end LangGraph agent behaviour with scripted model responses (no live API calls)."""
import json

import pytest

from src.graph import GraphConfig, LangGraphAgent, memory_checkpointer
from src.llm.tool_calling import ToolResultMessage
from src.matching.service import analyze_jobs
from src.tools import default_registry
from src.tools.errors import ToolPreconditionError, TransientToolError
from src.tools.registry import RetryPolicy, Tool
from src.tools.schemas import SearchJobsArgs, SearchJobsResult
from tests.unit.agent_fakes import ScriptedProvider, answer, call, observations, tools
from src.tools.context import build_offline_context


@pytest.fixture
def ctx():
    return build_offline_context()


def graph_agent(steps, ctx, registry=None, checkpointer=None, **config):
    provider = ScriptedProvider(steps)
    return LangGraphAgent(provider, registry or default_registry(), ctx, GraphConfig(**config),
                          checkpointer=checkpointer), provider


def failing_registry(handler):
    registry = default_registry()
    registry._tools["search_jobs"] = Tool("search_jobs", "injected failure", SearchJobsArgs, handler)
    return registry


# 1
def test_direct_response_without_tools(ctx):
    a, provider = graph_agent([answer("A CV summary lists your key skills first.")], ctx)
    result = a.run("What is a CV summary?")
    assert result.status == "completed" and result.tool_calls == [] and result.model_turns == 1
    assert result.path == ["prepare", "agent", "finalize"]
    assert provider.tool_names[0] == default_registry().names()


# 2
def test_one_tool_call(ctx):
    a, provider = graph_agent([tools(call("search_cv", {"query": "ROS2"}, "c1")),
                               answer("Your CV mentions ROS2.")], ctx)
    result = a.run("Search my CV for ROS2")
    assert result.status == "completed" and result.tool_sequence == ["search_cv"]
    assert result.path == ["prepare", "agent", "execute_tools", "agent", "finalize"]
    obs = provider.calls[1][-1]
    assert isinstance(obs, ToolResultMessage) and obs.call_id == "c1"


# 3
def test_multiple_tool_calls(ctx):
    def pick_first_job(messages):
        return tools(call("get_job_details", {"job_id": observations(messages)[-1]["data"]["jobs"][0]["job_id"]}))

    a, _ = graph_agent([tools(call("search_jobs", {"query": "Robotics"})), pick_first_job,
                        tools(call("save_job", {"job_id": "match-03-robotics"})), answer("Saved.")], ctx)
    result = a.run("Find a robotics job, show details and save it")
    assert result.tool_sequence == ["search_jobs", "get_job_details", "save_job"]
    assert [r.step for r in result.tool_calls] == [1, 2, 3]
    assert [j.job_id for j in ctx.saved_jobs.list()] == ["match-03-robotics"]


def test_save_request_takes_short_path(ctx):
    a, _ = graph_agent([tools(call("save_job", {"job_id": "fixture-13"})), answer("Saved fixture-13.")], ctx)
    result = a.run("Save job fixture-13")
    assert result.tool_sequence == ["save_job"] and "calculate_match" not in result.tool_sequence


# 4
def test_invalid_arguments_are_not_executed_and_model_corrects(ctx):
    a, provider = graph_agent([tools(call("calculate_match", {"job_id": "fixture-01", "score": 100})),
                               tools(call("calculate_match", {"job_ids": ["fixture-01"]})),
                               answer("Done.")], ctx)
    result = a.run("Match fixture-01")
    bad, good = result.tool_calls
    assert bad.error_code == "invalid_arguments" and bad.attempts == 0 and not bad.arguments_valid
    fields = {tuple(d["loc"]) for d in observations(provider.calls[1])[0]["error"]["details"]}
    assert fields == {("job_ids",), ("job_id",), ("score",)}
    assert good.status == "ok" and result.status == "completed"


def test_repeated_invalid_requests_terminate(ctx):
    a, provider = graph_agent([tools(call("find_jobs", {"query": "AI"})),
                               tools(call("search_jobs", {"q": "AI"})), answer("unreachable")],
                              ctx, max_invalid_tool_turns=2)
    result = a.run("Find AI jobs")
    assert result.status == "invalid_tool_requests" and result.final_answer is None
    assert [r.error_code for r in result.tool_calls] == ["unknown_tool", "invalid_arguments"]
    assert len(provider.calls) == 2 and result.path[-1] == "fail"


# 5
def test_tool_failure_is_observed_without_crashing(ctx):
    def explode(args, ctx):
        raise RuntimeError("database password=hunter2")

    a, provider = graph_agent([tools(call("search_jobs", {"query": "AI"})),
                               answer("The job search is unavailable right now.")], ctx, failing_registry(explode))
    result = a.run("Find AI jobs")
    assert result.status == "completed" and result.tool_calls[0].error_code == "internal"
    assert "hunter2" not in provider.calls[1][-1].content


# 6
def test_retryable_failure_recovers_within_bounded_retries(ctx):
    attempts = []

    def flaky(args, ctx):
        attempts.append(1)
        if len(attempts) == 1:
            raise TransientToolError("Catalog temporarily unavailable.")
        return SearchJobsResult(jobs=[], total_matches=0, message="none")

    a, _ = graph_agent([tools(call("search_jobs", {"query": "AI"})), answer("No jobs.")], ctx,
                       failing_registry(flaky), tool_retry=RetryPolicy(max_retries=2))
    result = a.run("Find AI jobs")
    assert len(attempts) == 2 and result.tool_calls[0].attempts == 2 and result.tool_calls[0].status == "empty"


def test_retryable_failure_exhausts_retries(ctx):
    attempts = []

    def down(args, ctx):
        attempts.append(1)
        raise TimeoutError()

    a, _ = graph_agent([tools(call("search_jobs", {"query": "AI"})), answer("Unavailable.")], ctx,
                       failing_registry(down), tool_retry=RetryPolicy(max_retries=1))
    record = a.run("Find AI jobs").tool_calls[0]
    assert len(attempts) == 2 and record.error_code == "transient" and record.outcome.error.retryable


# 7
def test_non_retryable_failure_is_not_retried(ctx):
    attempts = []

    def precondition(args, ctx):
        attempts.append(1)
        raise ToolPreconditionError("The user's CV has not been ingested.")

    a, _ = graph_agent([tools(call("search_jobs", {"query": "AI"})), answer("Please upload a CV.")], ctx,
                       failing_registry(precondition), tool_retry=RetryPolicy(max_retries=3))
    result = a.run("Find AI jobs")
    assert len(attempts) == 1 and result.tool_calls[0].error_code == "precondition_failed"


def test_provider_failure_routes_to_fail(ctx):
    class Broken:
        def complete(self, messages, tools):
            raise ConnectionError("api key sk-secret")

    result = LangGraphAgent(Broken(), default_registry(), ctx).run("hi")
    assert result.status == "provider_error" and "sk-secret" not in result.error
    assert result.path == ["prepare", "agent", "fail"]


# 8
def test_step_limit(ctx):
    looping = [tools(call("search_jobs", {"query": f"AI {i}"})) for i in range(10)]
    a, provider = graph_agent(looping, ctx, max_model_turns=3, max_tool_calls=50)
    result = a.run("Keep searching")
    assert result.status == "max_model_turns_exceeded" and result.final_answer is None
    assert result.model_turns == 3 and len(provider.calls) == 3 and len(result.tool_calls) == 3


# 9
def test_tool_call_limit(ctx):
    burst = tools(*[call("get_job_details", {"job_id": "fixture-01"}, f"c{i}") for i in range(5)])
    a, _ = graph_agent([burst, answer("unreachable")], ctx, max_model_turns=5, max_tool_calls=3)
    result = a.run("Show details many times")
    assert result.status == "max_tool_calls_exceeded" and len(result.tool_calls) == 3


# 10
@pytest.mark.parametrize("steps, expected_path", [
    ([answer("hi")], ["prepare", "agent", "finalize"]),
    ([tools(call("search_cv", {"query": "Python"})), answer("ok")],
     ["prepare", "agent", "execute_tools", "agent", "finalize"]),
    ([tools(call("search_cv", {"query": "Python"}))] * 2,
     ["prepare", "agent", "execute_tools", "agent", "execute_tools", "fail"]),
])
def test_conditional_routing_paths(ctx, steps, expected_path):
    a, _ = graph_agent(steps, ctx, max_model_turns=2)
    assert a.run("request").path == expected_path


# 11
def test_deterministic_match_score_is_preserved(ctx):
    def misquote(messages):
        return answer("fixture-01 is a 99/100 fit.")

    a, _ = graph_agent([tools(call("calculate_match", {"job_ids": ["fixture-01", "fixture-21"]})), misquote], ctx)
    result = a.run("How well do I fit fixture-01 and fixture-21?")
    expected = analyze_jobs([ctx.catalog.get(i).requirements() for i in ("fixture-01", "fixture-21")],
                            ctx.cv_id, ctx.cv_service, profile=ctx.profile())
    matches = result.tool_calls[0].outcome.data["matches"]
    assert [(m["job_id"], m["score"]["value"]) for m in matches] == [(x.job_id, x.score.value) for x in expected]
    assert result.ungrounded_scores == ["99"]  # the model's invented score is flagged, never adopted


# 12
def test_reference_scenario_germany_ai_jobs(ctx):
    def match_found_jobs(messages):
        ids = [j["job_id"] for j in observations(messages)[-1]["data"]["jobs"]]
        return tools(call("calculate_match", {"job_ids": ids}))

    def explain(messages):
        matches = observations(messages)[-1]["data"]["matches"]
        return answer("Best matches:\n" + "\n".join(
            f"{m['rank']}. {m['role']} ({m['job_id']}): {m['score']['value']}/100" for m in matches[:3]))

    a, _ = graph_agent([tools(call("search_jobs", {"query": "AI Engineer", "location": "Germany", "limit": 5})),
                        match_found_jobs, explain], ctx)
    result = a.run("Find AI Engineer jobs in Germany that fit my CV and explain the best matches.")
    assert result.status == "completed" and result.tool_sequence == ["search_jobs", "calculate_match"]
    ids = [j["job_id"] for j in result.tool_calls[0].outcome.data["jobs"]]
    expected = analyze_jobs([ctx.catalog.get(i).requirements() for i in ids], ctx.cv_id, ctx.cv_service,
                            profile=ctx.profile())
    matches = result.tool_calls[1].outcome.data["matches"]
    assert [(m["job_id"], m["score"]["value"]) for m in matches] == [(x.job_id, x.score.value) for x in expected]
    assert result.ungrounded_scores == [] and f"{expected[0].score.value}/100" in result.final_answer


# --- Checkpointing --------------------------------------------------------------

def test_checkpointed_thread_continues_conversation(ctx):
    from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener
    events = []
    unregister = register_serde_event_listener(events.append)
    a, provider = graph_agent([tools(call("search_jobs", {"query": "Robotics"})), answer("Found match-03."),
                               tools(call("save_job", {"job_id": "match-03-robotics"})), answer("Saved.")],
                              ctx, checkpointer=memory_checkpointer())
    first = a.run("Find robotics jobs", thread_id="t1")
    second = a.run("Save the first one", thread_id="t1")
    unregister()
    # Every restored application type is on the explicit allowlist (CHECKPOINT_TYPES).
    assert not [e for e in events if "unregistered" in e.get("kind", "")]
    assert first.thread_id == second.thread_id == "t1"
    assert second.tool_sequence == ["save_job"] and second.model_turns == 2  # counters are per run
    roles = [getattr(m, "role", type(m).__name__) for m in provider.calls[2]]
    assert roles.count("system") == 1 and roles.count("user") == 2


def test_threads_are_isolated(ctx):
    a, provider = graph_agent([answer("one"), answer("two")], ctx, checkpointer=memory_checkpointer())
    a.run("first", thread_id="a")
    a.run("second", thread_id="b")
    assert len(provider.calls[1]) == 2  # system + user only


def test_continued_thread_closes_calls_left_by_a_limit(ctx):
    burst = tools(*[call("get_job_details", {"job_id": "fixture-01"}, f"c{i}") for i in range(3)])
    a, provider = graph_agent([burst, answer("Recovered.")], ctx, checkpointer=memory_checkpointer(),
                              max_tool_calls=1)
    assert a.run("details", thread_id="t").status == "max_tool_calls_exceeded"
    assert a.run("try again", thread_id="t").status == "completed"
    results = [m for m in provider.calls[1] if isinstance(m, ToolResultMessage)]
    assert [m.call_id for m in results] == ["c0", "c1", "c2"]
    assert json.loads(results[2].content)["status"] == "not_executed"


def test_thread_id_requires_checkpointer(ctx):
    a, _ = graph_agent([answer("x")], ctx)
    with pytest.raises(ValueError):
        a.run("x", thread_id="t")


def test_graph_logs_metadata_without_request_or_cv_text(ctx, caplog):
    caplog.set_level("INFO")
    a, _ = graph_agent([tools(call("search_cv", {"query": "Python"})), answer("ok")], ctx)
    a.run("My private phone number is 0151-123", run_id="run-42")
    events = [json.loads(r.getMessage()) for r in caplog.records if r.name in ("src.graph", "src.tools")]
    assert [e["event"] for e in events] == ["agent_run_started", "model_turn", "tool_succeeded",
                                            "model_turn", "agent_run_finished"]
    assert all(e["run_id"] == "run-42" for e in events)
    assert "0151" not in caplog.text and "data processing" not in caplog.text
