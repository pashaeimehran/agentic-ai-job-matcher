"""System-level evaluation: one entry point over the dedicated evaluators.

Offline (default, deterministic, no network):
  - retrieval: lexical Hit@3 / MRR (default 900/150 chunking) and vector-vs-hybrid precision/recall@3
  - matching: requirement-status accuracy and deterministic score reproducibility
  - tool selection: reference replay through both agents (harness check, not a model measurement)
  - orchestration: manual loop vs LangGraph equivalence, termination behaviour, score consistency
  - version comparison harness (prompt v1 vs v2) with the fixture provider
  - fixture-mode end to end: task success, tool calls and p50/p95 latency of LangGraph + real tools

--live adds (paid OpenAI calls, synthetic data only):
  - live tool selection with per-query tokens, estimated cost and p50/p95 latency
  - a response-cache experiment: the same cases twice, cold then warm cache

Recorded results of separately run evaluators (live version comparison, Ragas in
its isolated environment) are included with their source file, never re-labelled.

Examples:
    python -m scripts.evaluate_system
    python -m scripts.evaluate_system --live
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.observability.metrics import MemoryUsageSink, percentile, summarize

RESULTS = Path("evaluation/results")

FIXTURE_DEMO = [
    # (request, tools that must appear in order)
    ("Find AI Engineer jobs in Germany that fit my CV", ["search_jobs", "calculate_match"]),
    ("Search for Machine Learning Engineer jobs in Hamburg", ["search_jobs"]),
    ("Show details for match-03-robotics", ["get_job_details"]),
    ("How well do I fit fixture-21 and match-04-cloud-platform?", ["calculate_match"]),
    ("What is my German level in my CV?", ["search_cv"]),
    ("Save fixture-13", ["save_job"]),
    ("Find Underwater Welder jobs in Munich", ["search_jobs"]),
    ("Hello, what can you do?", []),
]


def _p(values: list[float], q: float) -> float | None:
    value = percentile(values, q)
    return None if value is None else round(value, 1)


def retrieval_offline() -> dict:
    from scripts.evaluate_cv_retrieval import evaluate
    from src.rag.embeddings import HashEmbeddings
    from src.rag.service import CVService
    from src.rag.store import MemoryVectorStore
    result = evaluate(CVService(HashEmbeddings(), MemoryVectorStore()), "system-eval", 900, 150, 3)
    return {"questions": result["questions"], "hit_at_3": round(result["hit_rate"], 4),
            "mrr": round(result["mrr"], 4), "embeddings": "lexical-hash-v1 (not semantic)"}


def matching_offline() -> dict:
    from scripts.evaluate_job_matching import evaluate
    result = evaluate()
    return {"job_cases": result["job_cases"], "labeled_requirement_checks": result["labeled_requirement_checks"],
            "requirement_status_accuracy": result["requirement_status_accuracy"],
            "score_reproducibility": result["score_reproducibility"], "retrieval_at_3": result["retrieval"]}


def tool_selection_replay() -> dict:
    from scripts.evaluate_tool_selection import AGENTS, ReferenceReplayProvider, check_reference_statuses, evaluate
    out = {}
    for name, factory in AGENTS.items():
        report = evaluate(lambda case: ReferenceReplayProvider(case["reference_calls"]), "replay",
                          agent_factory=factory, agent_name=name)
        out[name] = {"cases": report["cases"], "first_tool_accuracy": report["first_tool_accuracy"],
                     "task_completion_rate": report["task_completion_rate"],
                     "reference_status_problems": check_reference_statuses(report)}
    return out


def orchestration() -> dict:
    from scripts.compare_agents import compare
    report = compare()
    return {key: report[key] for key in ("scenarios", "equivalent", "expected_differences", "unexpected_differences",
                                         "task_completion_rate", "total_tool_calls", "score_consistency",
                                         "termination_reasons")}


def version_harness() -> dict:
    from scripts.compare_versions import run
    from src.llm.fixture_provider import FixtureToolCallingProvider
    from src.observability.instrumented import InstrumentedProvider
    sink = MemoryUsageSink()
    report = run(lambda v: InstrumentedProvider(FixtureToolCallingProvider(), sink=sink, prompt_version=v),
                 sink, 1, "offline")
    return {"label_problems": report["label_problems"],
            "deterministic_score_consistency": report["deterministic_score_consistency"],
            "fixture_task_success": {v: report["summary"][v]["task_success_rate"] for v in ("v1", "v2")}}


def fixture_end_to_end(repeats: int = 5) -> dict:
    """LangGraph + real tools + fixture provider: orchestration latency and scripted task success."""
    from src.graph import GraphConfig, LangGraphAgent
    from src.llm.fixture_provider import FixtureToolCallingProvider
    from src.observability.instrumented import InstrumentedProvider
    from src.tools import default_registry
    from src.tools.context import build_offline_context
    sink, rows = MemoryUsageSink(), []
    registry = default_registry()
    for repeat in range(repeats):
        for index, (request, expected) in enumerate(FIXTURE_DEMO):
            agent = LangGraphAgent(InstrumentedProvider(FixtureToolCallingProvider(), sink=sink), registry,
                                   build_offline_context(), GraphConfig())
            started = time.perf_counter()
            result = agent.run(request, run_id=f"fx-{repeat}-{index}")
            rows.append({"latency_ms": (time.perf_counter() - started) * 1000, "tool_calls": len(result.tool_calls),
                         "success": result.status == "completed" and result.tool_sequence == expected})
    latencies = [r["latency_ms"] for r in rows]
    return {"requests": len(FIXTURE_DEMO), "repeats": repeats,
            "task_success_rate": round(sum(r["success"] for r in rows) / len(rows), 4),
            "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / len(rows), 3),
            "latency_ms_p50": _p(latencies, 50), "latency_ms_p95": _p(latencies, 95),
            "note": "Scripted provider: measures orchestration + tool latency and wiring, not model quality."}


def live_tool_selection_and_cache() -> dict:
    """Two passes over the 18 tool-selection cases with one warm-able cache: cold, then warm."""
    import os
    from dotenv import load_dotenv
    from scripts.evaluate_tool_selection import AGENTS, evaluate
    from src.config import Settings
    from src.llm.factory import build_provider
    from src.observability.setup import build_cache
    load_dotenv()
    settings = Settings.from_env({**os.environ, "APP_MODE": "live", "LLM_CACHE_ENABLED": "true"})
    sink, cache = MemoryUsageSink(), build_cache(settings)
    provider = build_provider(settings, sink, cache)
    passes = {}
    for name in ("cold", "warm"):
        start = len(sink.records)
        timings: dict[str, float] = {}

        def timed_factory(p, r, c, _factory=AGENTS["langgraph"]):
            agent = _factory(p, r, c)
            original = agent.run

            def run(request, run_id=None):
                begun = time.perf_counter()
                try:
                    return original(request, run_id=run_id)
                finally:
                    timings[run_id] = (time.perf_counter() - begun) * 1000
            agent.run = run
            return agent

        report = evaluate(lambda case: provider, "live", agent_factory=timed_factory, agent_name="langgraph")
        records = sink.records[start:]
        usage = summarize(records)
        latencies = list(timings.values())
        passes[name] = {
            "cases": report["cases"], "first_tool_accuracy": report["first_tool_accuracy"],
            "task_completion_rate": report["task_completion_rate"], "mean_tool_calls": report["mean_tool_calls"],
            **usage,
            "mean_tokens_per_query": round(usage["total_tokens"] / report["cases"], 1),
            "mean_cost_usd_per_query": None if usage["estimated_cost_usd"] is None
            else round(usage["estimated_cost_usd"] / report["cases"], 6),
            "end_to_end_latency_ms_p50": _p(latencies, 50), "end_to_end_latency_ms_p95": _p(latencies, 95),
        }
    return {"model": settings.openai_model, "agent": "langgraph", "passes": passes,
            "note": "Single live run per case and pass. The warm pass reuses cached model turns when the full "
                    "conversation (including tool observations) is identical; time-dependent tool output "
                    "(save_job timestamps) makes later turns miss by design."}


def recorded(name: str, keys: tuple[str, ...]) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"source": str(path).replace("\\", "/"), **{k: data.get(k) for k in keys}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified system evaluation")
    parser.add_argument("--live", action="store_true", help="Add paid live tool-selection and cache measurements")
    parser.add_argument("--output")
    args = parser.parse_args()
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric_classes": {
            "deterministic_offline": ["retrieval", "matching", "tool_selection_replay", "orchestration",
                                      "version_harness", "fixture_end_to_end"],
            "single_live_run": ["live_tool_selection_and_cache"],
            "repeated_live_samples": ["recorded.version_comparison"],
            "llm_judge": ["recorded.ragas"],
        },
        "retrieval": retrieval_offline(),
        "matching": matching_offline(),
        "tool_selection_replay": tool_selection_replay(),
        "orchestration": orchestration(),
        "version_harness": version_harness(),
        "fixture_end_to_end": fixture_end_to_end(),
    }
    if args.live:
        report["live_tool_selection_and_cache"] = live_tool_selection_and_cache()
    report["recorded"] = {
        "version_comparison": recorded("version_comparison_live.json",
                                       ("generated_at", "model", "repeats", "scenarios", "summary",
                                        "deterministic_score_consistency", "mischaracterization_fisher_p_two_sided")),
        "ragas": recorded("matching_ragas_final.json", ("scope", "candidate_profile_source", "jobs", "metrics"))
        or recorded("matching_ragas_baseline.json", ("scope", "candidate_profile_source", "jobs", "metrics")),
        "tool_selection_live_manual": recorded("tool_selection_live.json",
                                               ("model", "cases", "first_tool_accuracy", "task_completion_rate")),
    }
    gates = {
        "matching_status_accuracy_is_1": report["matching"]["requirement_status_accuracy"] == 1.0,
        "scores_reproducible": report["matching"]["score_reproducibility"],
        "retrieval_hit_at_3_at_least_0.8": report["retrieval"]["hit_at_3"] >= 0.8,
        "replay_reference_statuses_ok": all(not v["reference_status_problems"]
                                            for v in report["tool_selection_replay"].values()),
        "no_unexpected_orchestration_differences": not report["orchestration"]["unexpected_differences"],
        "orchestration_score_consistency": report["orchestration"]["score_consistency"],
        "version_labels_match_calculate_match": not report["version_harness"]["label_problems"],
    }
    report["gates"] = gates
    path = Path(args.output or RESULTS / ("system_evaluation_live.json" if args.live else "system_evaluation_offline.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("retrieval", "fixture_end_to_end")}, indent=1))
    print("Matching status accuracy:", report["matching"]["requirement_status_accuracy"])
    print("Orchestration:", report["orchestration"]["equivalent"], "/", report["orchestration"]["scenarios"], "equivalent")
    if args.live:
        for name, values in report["live_tool_selection_and_cache"]["passes"].items():
            print(name, {k: values[k] for k in ("task_completion_rate", "cache_hit_rate", "mean_tokens_per_query",
                                                "mean_cost_usd_per_query", "end_to_end_latency_ms_p50",
                                                "end_to_end_latency_ms_p95")})
    print("Gates:", gates)
    print("Report:", path)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
