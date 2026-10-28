"""Manual agent loop tests with scripted model responses (no live API calls)."""
import json

import pytest

from src.agent import AgentConfig, ManualAgent
from src.agent.loop import ungrounded_scores
from src.llm.tool_calling import ChatMessage, ToolResultMessage
from src.tools import default_registry
from src.tools.context import build_offline_context
from src.tools.errors import ToolPreconditionError, TransientToolError
from src.tools.registry import RetryPolicy, Tool, ToolRegistry
from src.tools.schemas import SearchJobsArgs, SearchJobsResult
from tests.unit.agent_fakes import ScriptedProvider, answer, call, observations, tools


@pytest.fixture
def ctx():
    return build_offline_context()


def agent(steps, ctx, registry=None, **config):
    provider = ScriptedProvider(steps)
    return ManualAgent(provider, registry or default_registry(), ctx, AgentConfig(**config)), provider


def test_direct_final_answer_without_tools(ctx):
    a, provider = agent([answer("A CV summary lists your key skills first.")], ctx)
    result = a.run("What is a CV summary?")
    assert result.status == "completed" and result.tool_calls == [] and result.model_turns == 1
    assert result.final_answer.startswith("A CV summary")
    first = provider.calls[0]
    assert isinstance(first[0], ChatMessage) and first[0].role == "system"
    assert provider.tool_names[0] == ["search_jobs", "get_job_details", "search_cv",
                                      "calculate_match", "save_job"]


def test_one_tool_call_then_final_answer(ctx):
    a, provider = agent([tools(call("search_cv", {"query": "Python"}, "c1")),
                         answer("Your CV mentions Python.")], ctx)
    result = a.run("Do I have Python experience?")
    assert result.status == "completed" and result.tool_sequence == ["search_cv"]
    obs = provider.calls[1][-1]
    assert isinstance(obs, ToolResultMessage) and obs.call_id == "c1"
    assert json.loads(obs.content)["status"] == "ok"


def test_multiple_sequential_tool_calls(ctx):
    def pick_first_job(messages):
        job_id = observations(messages)[-1]["data"]["jobs"][0]["job_id"]
        return tools(call("get_job_details", {"job_id": job_id}))

    a, provider = agent([tools(call("search_jobs", {"query": "Robotics"})), pick_first_job,
                         tools(call("save_job", {"job_id": "match-03-robotics"})), answer("Saved.")], ctx)
    result = a.run("Find a robotics job, show details and save it")
    assert result.tool_sequence == ["search_jobs", "get_job_details", "save_job"]
    assert [r.step for r in result.tool_calls] == [1, 2, 3]
    assert result.tool_calls[1].outcome.data["job_id"] == "match-03-robotics"
    assert [j.job_id for j in ctx.saved_jobs.list()] == ["match-03-robotics"]


def test_parallel_tool_calls_in_one_turn(ctx):
    a, provider = agent([tools(call("get_job_details", {"job_id": "fixture-01"}, "a"),
                               call("get_job_details", {"job_id": "fixture-05"}, "b")),
                         answer("Two jobs.")], ctx)
    result = a.run("Compare fixture-01 and fixture-05")
    assert [r.call_id for r in result.tool_calls] == ["a", "b"]
    assert [m.call_id for m in provider.calls[1] if isinstance(m, ToolResultMessage)] == ["a", "b"]


def test_invalid_tool_name_is_reported_and_model_recovers(ctx):
    a, provider = agent([tools(call("find_jobs", {"query": "AI"})),
                         tools(call("search_jobs", {"query": "AI"})), answer("Found jobs.")], ctx)
    result = a.run("Find AI jobs")
    assert result.status == "completed"
    assert [(r.tool_name, r.error_code, r.arguments_valid) for r in result.tool_calls] == [
        ("find_jobs", "unknown_tool", False), ("search_jobs", None, True)]
    assert observations(provider.calls[1])[0]["error"]["code"] == "unknown_tool"


def test_invalid_arguments_are_not_executed_and_model_corrects(ctx):
    a, provider = agent([tools(call("calculate_match", {"job_id": "fixture-01", "score": 100})),
                         tools(call("calculate_match", {"job_ids": ["fixture-01"]})), answer("Done.")], ctx)
    result = a.run("Match fixture-01")
    bad, good = result.tool_calls
    assert bad.error_code == "invalid_arguments" and bad.attempts == 0 and not bad.arguments_valid
    fields = {tuple(d["loc"]) for d in observations(provider.calls[1])[0]["error"]["details"]}
    assert fields == {("job_ids",), ("job_id",), ("score",)}
    assert good.status == "ok"


def failing_registry(handler):
    registry = default_registry()
    registry._tools["search_jobs"] = Tool("search_jobs", "injected failure", SearchJobsArgs, handler)
    return registry


def test_injected_tool_exception_does_not_crash_the_loop(ctx):
    def explode(args, ctx):
        raise RuntimeError("database password=hunter2")

    a, provider = agent([tools(call("search_jobs", {"query": "AI"})),
                         answer("The job search is unavailable right now.")],
                        ctx, failing_registry(explode))
    result = a.run("Find AI jobs")
    assert result.status == "completed"
    record = result.tool_calls[0]
    assert record.error_code == "internal" and record.attempts == 1 and record.arguments_valid
    assert "hunter2" not in provider.calls[1][-1].content


def test_retryable_failure_then_recovery(ctx):
    attempts = []

    def flaky(args, ctx):
        attempts.append(1)
        if len(attempts) == 1:
            raise TransientToolError("Catalog temporarily unavailable.")
        return SearchJobsResult(jobs=[], total_matches=0, message="none")

    a, _ = agent([tools(call("search_jobs", {"query": "AI"})), answer("No jobs.")], ctx,
                 failing_registry(flaky), tool_retry=RetryPolicy(max_retries=2))
    result = a.run("Find AI jobs")
    assert len(attempts) == 2 and result.tool_calls[0].attempts == 2
    assert result.tool_calls[0].status == "empty" and result.status == "completed"


def test_retryable_failure_exhausts_bounded_retries(ctx):
    attempts = []

    def down(args, ctx):
        attempts.append(1)
        raise TimeoutError()

    a, _ = agent([tools(call("search_jobs", {"query": "AI"})), answer("Unavailable.")], ctx,
                 failing_registry(down), tool_retry=RetryPolicy(max_retries=1))
    record = a.run("Find AI jobs").tool_calls[0]
    assert len(attempts) == 2 and record.error_code == "transient" and record.outcome.error.retryable


def test_non_retryable_failure_is_not_retried(ctx):
    attempts = []

    def precondition(args, ctx):
        attempts.append(1)
        raise ToolPreconditionError("The user's CV has not been ingested.")

    a, provider = agent([tools(call("search_jobs", {"query": "AI"})), answer("Please upload a CV.")], ctx,
                        failing_registry(precondition), tool_retry=RetryPolicy(max_retries=3))
    result = a.run("Find AI jobs")
    assert len(attempts) == 1 and result.tool_calls[0].error_code == "precondition_failed"
    assert observations(provider.calls[1])[0]["error"]["message"] == "The user's CV has not been ingested."


def test_max_model_turns_protection(ctx):
    looping = [tools(call("search_jobs", {"query": f"AI {i}"})) for i in range(10)]
    a, provider = agent(looping, ctx, max_model_turns=3, max_tool_calls=50)
    result = a.run("Keep searching")
    assert result.status == "max_model_turns_exceeded" and result.final_answer is None
    assert result.model_turns == 3 and len(provider.calls) == 3 and len(result.tool_calls) == 3


def test_max_tool_calls_protection(ctx):
    burst = tools(*[call("get_job_details", {"job_id": "fixture-01"}, f"c{i}") for i in range(5)])
    a, _ = agent([burst, answer("unreachable")], ctx, max_model_turns=5, max_tool_calls=3)
    result = a.run("Show details many times")
    assert result.status == "max_tool_calls_exceeded" and len(result.tool_calls) == 3
    assert "3" in result.error


def test_zero_tool_budget_allows_direct_answers_only(ctx):
    a, _ = agent([tools(call("search_jobs", {"query": "AI"}))], ctx, max_tool_calls=0)
    assert a.run("Find AI jobs").status == "max_tool_calls_exceeded"


def test_provider_failure_is_structured(ctx):
    class Broken:
        def complete(self, messages, tools):
            raise ConnectionError("api key sk-secret")

    result = ManualAgent(Broken(), default_registry(), ctx).run("hi")
    assert result.status == "provider_error" and "sk-secret" not in result.error


def test_agent_logs_metadata_without_request_or_cv_text(ctx, caplog):
    caplog.set_level("INFO")
    a, _ = agent([tools(call("search_cv", {"query": "Python"})), answer("ok")], ctx)
    a.run("My private phone number is 0151-123", run_id="run-42")
    events = [json.loads(r.getMessage()) for r in caplog.records if r.name in ("src.agent", "src.tools")]
    assert [e["event"] for e in events] == ["agent_run_started", "model_turn", "tool_succeeded",
                                            "model_turn", "agent_run_finished"]
    assert all(e["run_id"] == "run-42" for e in events)
    assert "0151" not in caplog.text and "data processing" not in caplog.text


def test_ungrounded_score_detection_ignores_identifiers():
    from src.agent.schemas import ToolCallRecord
    from src.tools.schemas import ToolOutcome
    outcome = ToolOutcome(tool_name="calculate_match", status="ok", attempts=1, duration_ms=0,
                          data={"matches": [{"score": {"value": 95.11}}]})
    records = [ToolCallRecord(step=1, call_id="x", tool_name="calculate_match", arguments_valid=True,
                              status="ok", attempts=1, duration_ms=0, outcome=outcome)]
    assert ungrounded_scores("match-01 scored 95.11/100", records) == []
    assert ungrounded_scores("score for match-01: 95/100, and 88/100", records) == ["88", "95"]


def test_reference_scenario_germany_ai_jobs_ranked_by_deterministic_scores(ctx):
    """Find AI Engineer jobs in Germany that fit my CV and explain the best matches."""

    def match_found_jobs(messages):
        ids = [j["job_id"] for j in observations(messages)[-1]["data"]["jobs"]]
        return tools(call("calculate_match", {"job_ids": ids}))

    def explain(messages):
        matches = observations(messages)[-1]["data"]["matches"]
        lines = [f"{m['rank']}. {m['role']} ({m['job_id']}): {m['score']['value']}/100; "
                 f"supported {', '.join(m['supported']) or 'none'}; unknown {', '.join(m['unknown']) or 'none'}"
                 for m in matches[:3]]
        return answer("Best matches:\n" + "\n".join(lines))

    a, _ = agent([tools(call("search_jobs", {"query": "AI Engineer", "location": "Germany", "limit": 5})),
                  match_found_jobs, explain], ctx)
    result = a.run("Find AI Engineer jobs in Germany that fit my CV and explain the best matches.")
    assert result.status == "completed"
    assert result.tool_sequence == ["search_jobs", "calculate_match"]
    matches = result.tool_calls[1].outcome.data["matches"]
    assert len(matches) == 5
    # Ranking is exactly the deterministic matching layer's ordering.
    from src.matching.service import analyze_jobs
    ids = [j["job_id"] for j in result.tool_calls[0].outcome.data["jobs"]]
    expected = analyze_jobs([ctx.catalog.get(i).requirements() for i in ids], ctx.cv_id,
                            ctx.cv_service, profile=ctx.profile())
    assert [(m["job_id"], m["score"]["value"]) for m in matches] == [(x.job_id, x.score.value) for x in expected]
    assert result.ungrounded_scores == []
    assert f"{expected[0].score.value}/100" in result.final_answer


def test_reference_scenario_flags_model_invented_scores(ctx):
    a, _ = agent([tools(call("search_jobs", {"query": "AI Engineer", "location": "Germany"})),
                  answer("fixture-01 looks like a 97/100 fit.")], ctx)
    result = a.run("Find AI Engineer jobs in Germany that fit my CV")
    assert result.ungrounded_scores == ["97"]
