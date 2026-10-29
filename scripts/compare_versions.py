"""Controlled version comparison: agent prompt v1 (baseline) vs v2 (requirement-status grounding).

Only the system prompt differs. Model, tools, catalog, synthetic CV, labeled
profile, deterministic matching, limits and scenarios are identical, and the
response cache is disabled so every run is an independent model sample.

Modes:
  offline  Deterministic harness check. Runs every scenario through the LangGraph
           agent with the fixture provider under both prompt versions and verifies
           labels and deterministic scores. It does NOT measure model behaviour.
  live     Paid OpenAI calls. --repeats runs per scenario and version, interleaved
           (case by case, alternating versions) to limit time-of-day drift.

Metrics per version: task success, runs with unknown requirements described as
deficits (src.agent.grounding), unknown-requirement coverage, ungrounded scores,
tool calls, model turns, wall latency p50/p95, tokens and estimated cost per query.

Examples:
    python -m scripts.compare_versions --mode offline
    python -m scripts.compare_versions --mode live --repeats 5
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.agent.grounding import mentioned, mischaracterization_evidence, unknown_requirements
from src.agent.prompts import system_prompt
from src.graph import GraphConfig, LangGraphAgent
from src.observability.metrics import MemoryUsageSink, percentile, summarize
from src.tools import default_registry
from src.tools.context import build_offline_context

CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "grounding_cases.json"
VERSIONS = ("v1", "v2")
CHANGE = ("Prompt-only change. v2 adds explicit definitions of supported / unmet / unknown and forbids "
          "describing an unknown requirement as lacking, missing, a gap, a weakness or unmet. Tools, "
          "deterministic matching and scoring, model, limits and scenarios are unchanged.")


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Grounding case IDs must be unique")
    catalog = build_offline_context().catalog
    for case in cases:
        unknown_ids = [j for j in case["match_job_ids"] if catalog.get(j) is None]
        if unknown_ids:
            raise ValueError(f"Case {case['id']} references unknown jobs {unknown_ids}")
    return cases


def check_labels(cases: list[dict]) -> list[str]:
    """Labeled unmet/unknown statuses must equal what deterministic calculate_match returns."""
    registry, context, problems = default_registry(), build_offline_context(), []
    for case in cases:
        if not case["expected"]:
            continue
        outcome = registry.execute("calculate_match", {"job_ids": list(case["expected"])}, context)
        for match in outcome.data["matches"]:
            wanted = case["expected"][match["job_id"]]
            if sorted(match["unknown"]) != sorted(wanted["unknown"]) or sorted(match["unmet"]) != sorted(wanted["unmet"]):
                problems.append(f"{case['id']}/{match['job_id']}: labels differ from calculate_match")
    return problems


def score_run(case: dict, result, records: list, wall_ms: float) -> dict:
    by_job = unknown_requirements(result.tool_calls)
    unknown = set().union(*by_job.values()) if by_job else set()
    answer = result.final_answer or ""
    expected_jobs = set(case["match_job_ids"])
    matched_ok = expected_jobs <= set(by_job) if expected_jobs else bool(by_job)
    usage = summarize(records)
    scores = sorted({(m["job_id"], m["score"]["value"]) for r in result.tool_calls
                     if r.tool_name == "calculate_match" and r.outcome.data for m in r.outcome.data["matches"]})
    return judge({
        "case": case["id"], "status": result.status, "tool_sequence": result.tool_sequence,
        "tool_calls": len(result.tool_calls), "model_turns": result.model_turns,
        "calculate_match_covered": matched_ok, "unknown_requirements": sorted(unknown),
        "ungrounded_scores": result.ungrounded_scores, "scores": scores,
        # Synthetic CV and catalog only: answers are kept so flagged runs can be audited and rescored.
        "final_answer": answer,
        "wall_latency_ms": round(wall_ms, 3), "llm_calls": usage["llm_calls"],
        "total_tokens": usage["total_tokens"], "estimated_cost_usd": usage["estimated_cost_usd"],
    })


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]]."""
    row1, row2, col1, n = a + b, c + d, a + c, a + b + c + d

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = prob(a)
    support = range(max(0, col1 - row2), min(row1, col1) + 1)
    return min(1.0, sum(p for p in map(prob, support) if p <= observed * (1 + 1e-9)))


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    latencies = [r["wall_latency_ms"] for r in rows]
    unknown_total = sum(len(r["unknown_requirements"]) for r in rows)
    costs = [r["estimated_cost_usd"] for r in rows if r["estimated_cost_usd"] is not None]
    return {
        "runs": n,
        "task_success_rate": round(sum(r["task_success"] for r in rows) / n, 4),
        "completed_rate": round(sum(r["status"] == "completed" for r in rows) / n, 4),
        "runs_with_mischaracterized_unknown": sum(bool(r["mischaracterized_unknowns"]) for r in rows),
        "mischaracterized_unknown_rate": round(sum(bool(r["mischaracterized_unknowns"]) for r in rows) / n, 4),
        "mischaracterized_requirement_mentions": sum(len(r["mischaracterized_unknowns"]) for r in rows),
        "unknown_coverage": round(sum(len(r["unknown_mentioned"]) for r in rows) / unknown_total, 4)
        if unknown_total else None,
        "runs_with_ungrounded_scores": sum(bool(r["ungrounded_scores"]) for r in rows),
        "calculate_match_coverage": round(sum(r["calculate_match_covered"] for r in rows) / n, 4),
        "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 3),
        "mean_model_turns": round(sum(r["model_turns"] for r in rows) / n, 3),
        "wall_latency_ms_p50": round(percentile(latencies, 50), 1),
        "wall_latency_ms_p95": round(percentile(latencies, 95), 1),
        "mean_tokens_per_query": round(sum(r["total_tokens"] for r in rows) / n, 1),
        "mean_cost_usd_per_query": round(sum(costs) / len(costs), 6) if costs else None,
        "total_cost_usd": round(sum(costs), 6) if costs else None,
    }


def run(provider_for: Callable[[str], object], sink: MemoryUsageSink, repeats: int, mode: str) -> dict:
    cases = load_cases()
    label_problems = check_labels(cases)
    registry = default_registry()
    rows: dict[str, list[dict]] = {v: [] for v in VERSIONS}
    for repeat in range(repeats):
        for case in cases:
            for version in VERSIONS:
                agent = LangGraphAgent(provider_for(version), registry, build_offline_context(), GraphConfig(),
                                       system_prompt=system_prompt(version))
                run_id = f"vc-{version}-{case['id']}-{repeat}"
                started = time.perf_counter()
                result = agent.run(case["request"], run_id=run_id)
                wall = (time.perf_counter() - started) * 1000
                rows[version].append({"repeat": repeat, **score_run(case, result, sink.for_run(run_id), wall)})
    return build_report(rows, cases, repeats, mode, label_problems)


def judge(row: dict) -> dict:
    """(Re)apply the deterministic grounding check and task-success rule to a stored run."""
    evidence = mischaracterization_evidence(row["final_answer"], row["unknown_requirements"])
    flagged = sorted({name for name, _ in evidence})
    return {**row, "mischaracterized_unknowns": flagged,
            "flagged_sentences": [f"{name}: {sentence}" for name, sentence in evidence],
            "unknown_mentioned": mentioned(row["final_answer"], row["unknown_requirements"]),
            "task_success": (row["status"] == "completed" and bool(row["final_answer"])
                             and row["calculate_match_covered"] and not row["ungrounded_scores"] and not flagged)}


def build_report(rows: dict[str, list[dict]], cases: list[dict], repeats: int, mode: str,
                 label_problems: list[str]) -> dict:
    summary = {v: aggregate(rows[v]) for v in VERSIONS}
    observed_scores: dict[str, set] = {}
    for version in VERSIONS:
        for row in rows[version]:
            for job_id, value in row["scores"]:
                observed_scores.setdefault(job_id, set()).add(value)
    a = summary["v1"]["runs_with_mischaracterized_unknown"]
    c = summary["v2"]["runs_with_mischaracterized_unknown"]
    return {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline": "v1", "candidate": "v2", "change": CHANGE,
        "agent": "langgraph", "cache": "disabled", "scenarios": len(cases), "repeats": repeats,
        "label_problems": label_problems,
        "deterministic_score_consistency": all(len(values) == 1 for values in observed_scores.values()),
        "scored_jobs": len(observed_scores),
        "summary": summary,
        "mischaracterization_fisher_p_two_sided": round(fisher_exact_two_sided(
            a, summary["v1"]["runs"] - a, c, summary["v2"]["runs"] - c), 4),
        "per_case": {v: {case["id"]: aggregate([r for r in rows[v] if r["case"] == case["id"]])
                         for case in cases} for v in VERSIONS},
        "limitations": ("offline mode replays a deterministic fixture provider: it validates labels, harness and "
                        "score determinism, not model behaviour" if mode == "offline" else
                        f"{repeats} live sample(s) per scenario and version on synthetic data with one model; "
                        "the lexical deficit detector is conservative and can miss paraphrases. Treat differences "
                        "as indicative, not statistically conclusive."),
        "runs": rows,
    }


def markdown(report: dict) -> str:
    s1, s2 = report["summary"]["v1"], report["summary"]["v2"]
    rows = [("Task success rate", "task_success_rate"), ("Completed runs", "completed_rate"),
            ("Runs with an unknown requirement described as a deficit", "runs_with_mischaracterized_unknown"),
            ("Deficit-described unknown requirements (total)", "mischaracterized_requirement_mentions"),
            ("Unknown requirements mentioned in the answer", "unknown_coverage"),
            ("Runs with ungrounded scores", "runs_with_ungrounded_scores"),
            ("calculate_match covered the requested jobs", "calculate_match_coverage"),
            ("Mean tool calls", "mean_tool_calls"), ("Mean model turns", "mean_model_turns"),
            ("Wall latency p50 (ms)", "wall_latency_ms_p50"), ("Wall latency p95 (ms)", "wall_latency_ms_p95"),
            ("Mean tokens per query", "mean_tokens_per_query"),
            ("Mean estimated cost per query (USD)", "mean_cost_usd_per_query"),
            ("Total estimated cost (USD)", "total_cost_usd")]
    lines = [f"# Version comparison: prompt v1 vs v2 ({report['mode']})", "",
             f"Generated {report['generated_at']}. Agent: LangGraph. Cache: disabled. "
             f"{report['scenarios']} labeled scenarios x {report['repeats']} repeat(s) per version"
             + (f", model `{report['model']}`." if report.get("model") else "."), "",
             f"**What changed:** {report['change']}", "",
             "| Metric | v1 (baseline) | v2 (candidate) |", "|---|---:|---:|"]
    lines += [f"| {label} | {s1[key]} | {s2[key]} |" for label, key in rows]
    lines += ["", f"Deterministic score consistency across all runs and both versions: "
                  f"**{report['deterministic_score_consistency']}** ({report['scored_jobs']} jobs scored).",
              f"Fisher exact two-sided p-value for runs with a deficit-described unknown: "
              f"{report['mischaracterization_fisher_p_two_sided']}.", "",
              "## Per scenario (runs with a deficit-described unknown / task success rate)", "",
              "| Scenario | v1 | v2 |", "|---|---:|---:|"]
    for case, v1 in report["per_case"]["v1"].items():
        v2 = report["per_case"]["v2"][case]
        lines.append(f"| {case} | {v1['runs_with_mischaracterized_unknown']}/{v1['runs']} · {v1['task_success_rate']} "
                     f"| {v2['runs_with_mischaracterized_unknown']}/{v2['runs']} · {v2['task_success_rate']} |")
    if report.get("rescored_at"):
        lines += ["", f"Runs were re-judged on {report['rescored_at']} with the current detector revision "
                      "(same stored answers, same rules for both versions). Every flagged sentence is listed in "
                      "the JSON report under runs.<version>[].flagged_sentences for audit."]
    lines += ["", "## Limitations", "", report["limitations"], "",
              "Task success = run completed with an answer, calculate_match covered the requested jobs, no "
              "ungrounded score, and no unknown requirement described as a deficit. The deficit check is "
              "`src/agent/grounding.py` (lexical, deterministic).", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare agent prompt versions on labeled grounding scenarios")
    parser.add_argument("--mode", choices=["offline", "live"])
    parser.add_argument("--rescore", metavar="REPORT",
                        help="Re-apply the current grounding check to a stored live report (no model calls)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be in 1..20")
    if args.rescore:
        stored = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        rows = {v: [judge({k: val for k, val in row.items() if k != "repeat"}) | {"repeat": row["repeat"]}
                    for row in stored["runs"][v]] for v in VERSIONS}
        report = {**build_report(rows, load_cases(), stored["repeats"], stored["mode"], stored["label_problems"]),
                  "generated_at": stored["generated_at"], "rescored_at":
                  datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if stored.get("model"):
            report["model"] = stored["model"]
        Path(args.rescore).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        Path(args.rescore).with_name("version_comparison.md").write_text(markdown(report), encoding="utf-8")
        for version in VERSIONS:
            print(version, json.dumps(report["summary"][version]))
        return 0
    if args.mode is None:
        parser.error("--mode is required unless --rescore is given")
    sink = MemoryUsageSink()
    if args.mode == "offline":
        from src.llm.fixture_provider import FixtureToolCallingProvider
        from src.observability.instrumented import InstrumentedProvider
        report = run(lambda version: InstrumentedProvider(FixtureToolCallingProvider(), sink=sink,
                                                          prompt_version=version), sink, args.repeats, "offline")
        default = "evaluation/results/version_comparison_offline.json"
    else:
        from dotenv import load_dotenv
        from src.config import Settings
        from src.llm.factory import build_provider
        load_dotenv()
        settings = Settings.from_env({**__import__("os").environ, "APP_MODE": "live", "LLM_CACHE_ENABLED": "false"})
        report = run(lambda version: build_provider(settings, sink, None, prompt_version=version),
                     sink, args.repeats, "live")
        report["model"] = settings.openai_model
        default = "evaluation/results/version_comparison_live.json"
    path = Path(args.output or default)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.mode == "live":
        path.with_name("version_comparison.md").write_text(markdown(report), encoding="utf-8")
    for version in VERSIONS:
        print(version, json.dumps(report["summary"][version]))
    print("Deterministic score consistency:", report["deterministic_score_consistency"])
    print("Report:", path)
    if report["label_problems"]:
        print("Label problems:", report["label_problems"])
        return 1
    return 0 if report["deterministic_score_consistency"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
