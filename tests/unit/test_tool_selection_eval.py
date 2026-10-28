"""Deterministic checks for the tool-selection evaluation harness (no live model)."""
from scripts.evaluate_tool_selection import (ReferenceReplayProvider, check_reference_statuses,
                                             evaluate, load_cases)
from tests.unit.agent_fakes import ScriptedProvider, answer, call, tools


def test_dataset_covers_required_request_types():
    cases = load_cases()
    categories = {c["category"] for c in cases}
    assert {"search_jobs", "get_job_details", "search_cv", "calculate_match", "save_job",
            "no_tool", "ambiguous", "malformed", "multi_step", "reference_scenario"} <= categories
    reference = next(c for c in cases if c["category"] == "reference_scenario")
    assert reference["expected_sequence"] == ["search_jobs", "calculate_match"]


def test_reference_replay_validates_dataset_and_harness():
    report = evaluate(lambda case: ReferenceReplayProvider(case["reference_calls"]), "replay")
    assert check_reference_statuses(report) == []
    assert report["first_tool_accuracy"] == 1.0 and report["task_completion_rate"] == 1.0
    assert report["argument_validity_rate"] == 1.0
    assert report["total_tool_calls"] == sum(len(c["reference_calls"]) for c in load_cases())


def test_metrics_are_computed_from_actual_runs_not_assumed():
    # A policy that always searches and invents a score must be penalised.
    def bad_policy(case):
        return ScriptedProvider([tools(call("search_jobs", {"query": "AI", "limit": "many"})),
                                 answer("Your fit is 99/100.")])

    report = evaluate(bad_policy, "scripted")
    rows = {r["id"]: r for r in report["results"]}
    assert rows["search-ai-engineer"]["first_tool_correct"] is True
    assert rows["no-tool-greeting"]["first_tool_correct"] is False
    assert report["argument_validity_rate"] == 0.0
    assert report["task_completion_rate"] == 0.0  # every answer has an ungrounded score
    assert rows["reference-scenario-germany"]["sequence_correct"] is False
    assert 0 < report["first_tool_accuracy"] < 1
