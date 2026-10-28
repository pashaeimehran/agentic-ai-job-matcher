"""Run the tool-calling agent for one request (paid OpenAI calls).

Examples:
    python -m scripts.run_agent "Find AI Engineer jobs in Germany that fit my CV and explain the best matches."
    python -m scripts.run_agent --cv-id demo "Search my CV for Python experience"
    python -m scripts.run_agent --agent langgraph "Find AI Engineer jobs in Germany that fit my CV"

--agent selects the manual loop (default, baseline) or the LangGraph agent; both
use the same tools, limits and provider.

Without --cv-id the synthetic fixture CV is used (no database). With --cv-id the
CV must already be ingested in Postgres; its profile is extracted on first use.
"""
from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from src.agent import AgentConfig, ManualAgent
from src.graph import GraphConfig, LangGraphAgent
from src.tools import default_registry
from src.tools.context import ToolContext, build_offline_context
from src.tools.jobs import FixtureJobCatalog
from src.tools.saved_jobs import JsonFileSavedJobStore


def live_cv_context(cv_id: str, provider) -> ToolContext:
    from scripts.cv import live_service
    from src.matching.facts import extract_candidate_profile
    service, _ = live_service()
    return ToolContext(catalog=FixtureJobCatalog(), cv_service=service, cv_id=cv_id,
                       profile_loader=lambda: extract_candidate_profile(
                           provider, cv_id, service.store.list_chunks(cv_id)),
                       saved_jobs=JsonFileSavedJobStore())


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool-calling agent: manual loop or LangGraph")
    parser.add_argument("request")
    parser.add_argument("--agent", choices=["manual", "langgraph"], default="manual")
    parser.add_argument("--cv-id", help="Use an ingested Postgres CV instead of the synthetic fixture")
    parser.add_argument("--max-model-turns", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--verbose", action="store_true", help="Print structured agent/tool logs")
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(name)s %(message)s")
    try:
        from openai import OpenAI
        from src.llm.openai_provider import OpenAIProvider
        provider = OpenAIProvider(OpenAI(), os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        context = live_cv_context(args.cv_id, provider) if args.cv_id else build_offline_context()
        limits = {"max_model_turns": args.max_model_turns, "max_tool_calls": args.max_tool_calls}
        agent = (LangGraphAgent(provider, default_registry(), context, GraphConfig(**limits))
                 if args.agent == "langgraph" else
                 ManualAgent(provider, default_registry(), context, AgentConfig(**limits)))
        result = agent.run(args.request)
    except Exception as exc:
        # Do not echo provider payloads, CV content or credentials.
        print(f"Agent run failed ({type(exc).__name__}). Check API, database and configuration.")
        return 1
    print(f"Run {result.run_id} ({args.agent}): {result.status} after {result.model_turns} model turn(s)")
    if args.agent == "langgraph":
        print("  path:", " -> ".join(result.path))
    for record in result.tool_calls:
        detail = f" [{record.error_code}]" if record.error_code else ""
        print(f"  step {record.step}: {record.tool_name} -> {record.status}{detail} (attempts {record.attempts})")
    if result.ungrounded_scores:
        print("WARNING: answer mentions scores not produced by calculate_match:", result.ungrounded_scores)
    print()
    print(result.final_answer or result.error)
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
