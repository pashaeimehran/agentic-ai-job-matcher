"""Run identical scenarios through the manual loop and the LangGraph agent and compare them.

Deterministic and offline: every scenario replays a fixed sequence of model turns
against the real tool registry and synthetic fixtures, so differences come from
orchestration only, never from model sampling.

Scenarios:
  - the 18 tool-selection cases (reference calls replayed)
  - failure and limit scenarios: invalid/unknown tool recovery, repeated invalid
    requests, tool exceptions, retryable and non-retryable failures, provider
    error, model-turn limit, tool-call limit, invented-score detection

Example:
    python -m scripts.compare_agents
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from scripts.evaluate_tool_selection import load_cases
from src.agent import AgentConfig, ManualAgent
from src.graph import GraphConfig, LangGraphAgent
from src.llm.tool_calling import AssistantTurn, ToolCall
from src.tools import default_registry
from src.tools.context import build_offline_context
from src.tools.errors import ToolPreconditionError, TransientToolError
from src.tools.registry import RetryPolicy, Tool, ToolRegistry
from src.tools.schemas import SearchJobsArgs, SearchJobsResult

DEFAULT_OUTPUT = "evaluation/results/agent_comparison_offline.json"


class TurnReplayProvider:
    """Returns pre-built assistant turns in order; raises if a turn is configured as an exception."""

    def __init__(self, turns: list[AssistantTurn | Exception]) -> None:
        self._turns = list(turns)

    def complete(self, messages, tools):
        if not self._turns:
            raise RuntimeError("Scenario script exhausted")
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def _calls(*specs: tuple[str, dict | str]) -> AssistantTurn:
    return AssistantTurn(tool_calls=[
        ToolCall(call_id=f"c{i}-{name}", name=name, arguments=args if isinstance(args, str) else json.dumps(args))
        for i, (name, args) in enumerate(specs)])


def _answer(text: str) -> AssistantTurn:
    return AssistantTurn(text=text)


def _with_search_handler(handler: Callable) -> Callable[[], ToolRegistry]:
    def build() -> ToolRegistry:
        registry = default_registry()
        registry._tools["search_jobs"] = Tool("search_jobs", "injected failure", SearchJobsArgs, handler)
        return registry
    return build


def _flaky_search() -> Callable:
    attempts = []

    def flaky(args, ctx):
        attempts.append(1)
        if len(attempts) == 1:
            raise TransientToolError("Catalog temporarily unavailable.")
        return SearchJobsResult(jobs=[], total_matches=0, message="none")
    return flaky


def _raise(exc: Exception) -> Callable:
    def handler(args, ctx):
        raise exc
    return handler


@dataclass
class Scenario:
    id: str
    category: str
    request: str
    turns: Callable[[], list[AssistantTurn | Exception]]
    registry: Callable[[], ToolRegistry] = default_registry
    limits: dict = field(default_factory=dict)
    # Set when the two engines are designed to behave differently; documents why.
    expected_difference: str | None = None


def tool_selection_scenarios() -> list[Scenario]:
    def replay(case: dict) -> Callable[[], list[AssistantTurn]]:
        return lambda: [*(_calls((c["tool"], c["arguments"])) for c in case["reference_calls"]),
                        _answer("Reference replay complete.")]
    return [Scenario(c["id"], f"tool_selection/{c['category']}", c["request"], replay(c)) for c in load_cases()]


def failure_scenarios() -> list[Scenario]:
    return [
        Scenario("invalid-arguments-recovery", "failure", "Match fixture-01",
                 lambda: [_calls(("calculate_match", {"job_id": "fixture-01", "score": 100})),
                          _calls(("calculate_match", {"job_ids": ["fixture-01"]})), _answer("Done.")]),
        Scenario("unknown-tool-recovery", "failure", "Find AI jobs",
                 lambda: [_calls(("find_jobs", {"query": "AI"})), _calls(("search_jobs", {"query": "AI"})),
                          _answer("Found jobs.")]),
        Scenario("repeated-invalid-requests", "failure", "Find AI jobs",
                 lambda: [_calls(("find_jobs", {"query": "AI"})), _calls(("search_jobs", {"q": "AI"})),
                          _calls(("search_jobs", "not json")), _answer("Gave up.")],
                 expected_difference=("LangGraph stops after max_invalid_tool_turns=2 consecutive "
                                      "all-invalid turns; the manual loop has no such guard and keeps "
                                      "spending model turns until the model answers.")),
        Scenario("tool-internal-failure", "failure", "Find AI jobs",
                 lambda: [_calls(("search_jobs", {"query": "AI"})), _answer("Search is unavailable.")],
                 registry=_with_search_handler(_raise(RuntimeError("secret detail")))),
        Scenario("retryable-failure-recovers", "failure", "Find AI jobs",
                 lambda: [_calls(("search_jobs", {"query": "AI"})), _answer("No jobs.")],
                 registry=lambda: _with_search_handler(_flaky_search())(),
                 limits={"tool_retry": RetryPolicy(max_retries=2)}),
        Scenario("retryable-failure-exhausted", "failure", "Find AI jobs",
                 lambda: [_calls(("search_jobs", {"query": "AI"})), _answer("Unavailable.")],
                 registry=_with_search_handler(_raise(TimeoutError()))),
        Scenario("non-retryable-failure", "failure", "Find AI jobs",
                 lambda: [_calls(("search_jobs", {"query": "AI"})), _answer("Please upload a CV.")],
                 registry=_with_search_handler(_raise(ToolPreconditionError("CV not ingested."))),
                 limits={"tool_retry": RetryPolicy(max_retries=3)}),
        Scenario("provider-error", "failure", "hi", lambda: [ConnectionError("provider down")]),
        Scenario("model-turn-limit", "limit", "Keep searching",
                 lambda: [_calls(("search_jobs", {"query": f"AI {i}"})) for i in range(10)],
                 limits={"max_model_turns": 3, "max_tool_calls": 50}),
        Scenario("tool-call-limit", "limit", "Show details many times",
                 lambda: [_calls(*[("get_job_details", {"job_id": "fixture-01"})] * 5), _answer("unreachable")],
                 limits={"max_model_turns": 5, "max_tool_calls": 3}),
        Scenario("invented-score-detected", "grounding", "How well do I fit fixture-01?",
                 lambda: [_calls(("calculate_match", {"job_ids": ["fixture-01", "fixture-21"]})),
                          _answer("fixture-01 is a 99/100 fit.")]),
    ]


def _scores(result) -> list[list]:
    return [[m["job_id"], m["score"]["value"]] for r in result.tool_calls
            if r.tool_name == "calculate_match" and r.outcome.data for m in r.outcome.data["matches"]]


def _summary(result) -> dict:
    return {"status": result.status, "task_completed": result.status == "completed" and bool(result.final_answer),
            "tool_sequence": result.tool_sequence, "tool_calls": len(result.tool_calls),
            "model_turns": result.model_turns,
            "tool_outcomes": [[r.status, r.error_code, r.attempts] for r in result.tool_calls],
            "match_scores": _scores(result), "ungrounded_scores": result.ungrounded_scores,
            "error": result.error}


def run_scenario(scenario: Scenario) -> dict:
    results = {}
    for engine in ("manual", "langgraph"):
        provider, registry, context = TurnReplayProvider(scenario.turns()), scenario.registry(), build_offline_context()
        agent = (ManualAgent(provider, registry, context, AgentConfig(**scenario.limits)) if engine == "manual"
                 else LangGraphAgent(provider, registry, context, GraphConfig(**scenario.limits)))
        result = agent.run(scenario.request, run_id=f"cmp-{scenario.id}-{engine}")
        results[engine] = {**_summary(result), **({"path": result.path} if engine == "langgraph" else {})}
    manual, graph = results["manual"], results["langgraph"]
    compared = ("status", "tool_sequence", "tool_calls", "model_turns", "tool_outcomes", "match_scores",
                "ungrounded_scores")
    differences = [key for key in compared if manual[key] != graph[key]]
    return {"id": scenario.id, "category": scenario.category, "equivalent": not differences,
            "differences": differences, "expected_difference": scenario.expected_difference,
            "manual": manual, "langgraph": graph}


def compare() -> dict:
    rows = [run_scenario(s) for s in [*tool_selection_scenarios(), *failure_scenarios()]]
    unexpected = [r["id"] for r in rows if not r["equivalent"] and not r["expected_difference"]]
    expected = [r["id"] for r in rows if not r["equivalent"] and r["expected_difference"]]
    unexplained_match = [r["id"] for r in rows if r["equivalent"] and r["expected_difference"]]

    def rate(engine: str) -> float:
        return round(sum(r[engine]["task_completed"] for r in rows) / len(rows), 6)

    scored = [r for r in rows if r["manual"]["match_scores"] or r["langgraph"]["match_scores"]]
    return {
        "mode": "offline_replay",
        "limitations": ("Replays fixed model turns through both orchestrators. It measures orchestration "
                        "equivalence, not model quality."),
        "scenarios": len(rows),
        "equivalent": sum(r["equivalent"] for r in rows),
        "expected_differences": expected,
        "unexpected_differences": unexpected,
        "expected_differences_not_observed": unexplained_match,
        "task_completion_rate": {"manual": rate("manual"), "langgraph": rate("langgraph")},
        "total_tool_calls": {e: sum(r[e]["tool_calls"] for r in rows) for e in ("manual", "langgraph")},
        "score_scenarios": len(scored),
        "score_consistency": all(r["manual"]["match_scores"] == r["langgraph"]["match_scores"] for r in scored),
        "termination_reasons": {e: sorted({r[e]["status"] for r in rows}) for e in ("manual", "langgraph")},
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the manual loop and the LangGraph agent")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = compare()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Scenarios: {report['scenarios']}, equivalent: {report['equivalent']}")
    print(f"Task completion: {report['task_completion_rate']}")
    print(f"Total tool calls: {report['total_tool_calls']}")
    print(f"Score consistency ({report['score_scenarios']} scenarios): {report['score_consistency']}")
    for row in report["results"]:
        mark = "same" if row["equivalent"] else ("EXPECTED-DIFF" if row["expected_difference"] else "DIFF")
        print(f"  [{mark:>13}] {row['id']}: manual={row['manual']['status']} {row['manual']['tool_sequence']} | "
              f"langgraph={row['langgraph']['status']} {row['langgraph']['tool_sequence']}")
    print("Report:", path)
    problems = report["unexpected_differences"] + report["expected_differences_not_observed"]
    if problems:
        print("Comparison problems:", problems)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
