"""Manual loop vs LangGraph: same scenarios, same functional behaviour (offline replay)."""
from scripts.compare_agents import compare
from scripts.evaluate_tool_selection import AGENTS, ReferenceReplayProvider, check_reference_statuses, evaluate


def test_engines_are_equivalent_except_documented_differences():
    report = compare()
    assert report["unexpected_differences"] == []
    assert report["expected_differences"] == ["repeated-invalid-requests"]
    assert report["expected_differences_not_observed"] == []
    assert report["score_consistency"] is True and report["score_scenarios"] >= 5


def test_documented_difference_is_the_invalid_request_guard():
    row = next(r for r in compare()["results"] if r["id"] == "repeated-invalid-requests")
    assert row["manual"]["status"] == "completed" and row["manual"]["tool_calls"] == 3
    assert row["langgraph"]["status"] == "invalid_tool_requests" and row["langgraph"]["tool_calls"] == 2


def test_tool_selection_replay_is_identical_for_both_agents():
    reports = {name: evaluate(lambda case: ReferenceReplayProvider(case["reference_calls"]), "replay",
                              agent_factory=factory, agent_name=name) for name, factory in AGENTS.items()}
    for report in reports.values():
        assert check_reference_statuses(report) == [] and report["task_completion_rate"] == 1.0
    assert reports["manual"]["results"] == reports["langgraph"]["results"]
