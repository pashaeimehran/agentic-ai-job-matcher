"""Tool-selection evaluation for the manual agent loop and the LangGraph agent.

Modes:
  replay  Deterministic. Replays each case's reference_calls through the real
          loop and tool layer. Validates the dataset and harness; it does NOT
          measure model behaviour.
  live    Paid OpenAI calls. Measures the configured model's tool selection on
          the synthetic fixture CV and catalog (no private data).

--agent selects the orchestrator (default: manual). Both run the same cases,
registry and fixtures, so their reports are directly comparable.

Examples:
    python -m scripts.evaluate_tool_selection --mode replay
    python -m scripts.evaluate_tool_selection --mode replay --agent langgraph
    python -m scripts.evaluate_tool_selection --mode live
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

from src.agent import AgentConfig, AgentRunResult, ManualAgent
from src.graph import GraphConfig, LangGraphAgent
from src.llm.tool_calling import AssistantTurn, ToolCall, ToolCallingLLMProvider
from src.tools import default_registry
from src.tools.context import build_offline_context

CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "tool_selection_cases.json"

AGENTS = {
    "manual": lambda provider, registry, context: ManualAgent(provider, registry, context, AgentConfig()),
    "langgraph": lambda provider, registry, context: LangGraphAgent(provider, registry, context, GraphConfig()),
}


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Tool-selection case IDs must be unique")
    tools = set(default_registry().names())
    for case in cases:
        for name in [*case["acceptable_first_tools"], *case.get("expected_sequence", []),
                     *[c["tool"] for c in case["reference_calls"]]]:
            if name is not None and name not in tools:
                raise ValueError(f"Case {case['id']} references unknown tool {name}")
    return cases


class ReferenceReplayProvider:
    """Emits a case's reference calls one per turn, then a fixed final answer."""

    def __init__(self, reference_calls: list[dict]) -> None:
        self._turns = [AssistantTurn(tool_calls=[ToolCall(call_id=f"ref-{i}", name=c["tool"],
                                                          arguments=json.dumps(c["arguments"]))])
                       for i, c in enumerate(reference_calls)]
        self._turns.append(AssistantTurn(text="Reference replay complete."))

    def complete(self, messages, tools):
        return self._turns.pop(0)


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    remaining = iter(actual)
    return all(name in remaining for name in expected)


def score_case(case: dict, result: AgentRunResult) -> dict:
    sequence = result.tool_sequence
    first = sequence[0] if sequence else None
    expected_sequence = case.get("expected_sequence")
    sequence_ok = _is_subsequence(expected_sequence, sequence) if expected_sequence else None
    no_tool_case = case["acceptable_first_tools"] == [None]
    completed = (result.status == "completed" and bool(result.final_answer)
                 and not result.ungrounded_scores
                 and (not no_tool_case or not sequence)
                 and sequence_ok is not False)
    return {
        "id": case["id"], "category": case["category"],
        "first_tool": first, "first_tool_correct": first in case["acceptable_first_tools"],
        "tool_sequence": sequence, "sequence_correct": sequence_ok,
        "tool_calls": len(sequence),
        "valid_argument_calls": sum(r.arguments_valid for r in result.tool_calls),
        "tool_statuses": [r.status for r in result.tool_calls],
        "run_status": result.status, "model_turns": result.model_turns,
        "ungrounded_scores": result.ungrounded_scores, "task_completed": completed,
        "final_answer": result.final_answer,
    }


def evaluate(provider_for: Callable[[dict], ToolCallingLLMProvider], mode: str,
             config: AgentConfig = AgentConfig(), agent_factory: Callable | None = None,
             agent_name: str = "manual") -> dict:
    cases = load_cases()
    registry = default_registry()
    factory = agent_factory or (lambda p, r, c: ManualAgent(p, r, c, config))
    rows = []
    for case in cases:
        # Fresh context per case so saved jobs and cached state never leak between cases.
        agent = factory(provider_for(case), registry, build_offline_context())
        rows.append(score_case(case, agent.run(case["request"], run_id=f"eval-{case['id']}")))
    calls = sum(r["tool_calls"] for r in rows)
    sequence_rows = [r for r in rows if r["sequence_correct"] is not None]
    n = len(rows)
    return {
        "mode": mode,
        "agent": agent_name,
        "limitations": ("replay validates dataset and harness only; it is not a model measurement"
                        if mode == "replay" else
                        "single run per case on synthetic fixtures; LLM outputs are non-deterministic"),
        "cases": n,
        "first_tool_accuracy": round(sum(r["first_tool_correct"] for r in rows) / n, 6),
        "sequence_accuracy": (round(sum(r["sequence_correct"] for r in sequence_rows) / len(sequence_rows), 6)
                              if sequence_rows else None),
        "sequence_cases": len(sequence_rows),
        "argument_validity_rate": round(sum(r["valid_argument_calls"] for r in rows) / calls, 6) if calls else None,
        "task_completion_rate": round(sum(r["task_completed"] for r in rows) / n, 6),
        "total_tool_calls": calls,
        "mean_tool_calls": round(calls / n, 6),
        "results": rows,
    }


def check_reference_statuses(report: dict) -> list[str]:
    """Replay-only: every reference call must end with its expected status."""
    expected = {c["id"]: c.get("reference_status") for c in load_cases()}
    problems = []
    for row in report["results"]:
        wanted = expected[row["id"]]
        statuses = row["tool_statuses"]
        if wanted is not None and statuses[-1:] != wanted[-1:]:
            problems.append(f"{row['id']}: expected final status {wanted[-1]}, got {statuses}")
        if wanted is None and any(s != "ok" for s in statuses):
            problems.append(f"{row['id']}: reference call statuses {statuses}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate agent tool selection")
    parser.add_argument("--mode", choices=["replay", "live"], required=True)
    parser.add_argument("--agent", choices=sorted(AGENTS), default="manual")
    parser.add_argument("--output")
    args = parser.parse_args()
    agent = {"agent_factory": AGENTS[args.agent], "agent_name": args.agent}
    suffix = "" if args.agent == "manual" else f"_{args.agent}"
    if args.mode == "replay":
        report = evaluate(lambda case: ReferenceReplayProvider(case["reference_calls"]), "replay", **agent)
        report["reference_status_problems"] = check_reference_statuses(report)
        default_output = f"evaluation/results/tool_selection_replay{suffix}.json"
    else:
        from dotenv import load_dotenv
        from src.config import Settings
        from src.llm.factory import build_provider
        from src.observability.metrics import MemoryUsageSink, summarize
        load_dotenv()
        # Cache off: every case is an independent model sample.
        settings = Settings.from_env({**os.environ, "APP_MODE": "live", "LLM_CACHE_ENABLED": "false"})
        sink = MemoryUsageSink()
        provider = build_provider(settings, sink, None, prompt_version="v1")
        report = evaluate(lambda case: provider, "live", **agent)
        report["model"] = settings.openai_model
        report["usage"] = summarize(sink.records)
        default_output = f"evaluation/results/tool_selection_live{suffix}.json"
    path = Path(args.output or default_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Mode: {report['mode']}, agent: {report['agent']} ({report['cases']} cases)")
    for key in ("first_tool_accuracy", "sequence_accuracy", "argument_validity_rate",
                "task_completion_rate", "mean_tool_calls"):
        print(f"{key}: {report[key]}")
    for row in report["results"]:
        mark = "ok " if row["first_tool_correct"] and row["task_completed"] else "MISS"
        print(f"  [{mark}] {row['id']}: {row['tool_sequence']} -> {row['run_status']}")
    if report.get("reference_status_problems"):
        print("Reference problems:", *report["reference_status_problems"], sep="\n  ")
        return 1
    print("Report:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
