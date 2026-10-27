"""Run a public offline demo, or match stored CV against synthetic sample jobs.

Examples:
    python -m scripts.match_jobs --offline
    python -m scripts.match_jobs --live --cv-id demo
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from src.matching.fixtures import ROOT, fixture_jobs, fixture_profile
from src.matching.service import analyze_jobs, render_match_explanation
from src.rag.embeddings import HashEmbeddings
from src.rag.service import CVService
from src.rag.store import MemoryVectorStore


def run_offline():
    cv_id = "synthetic-demo"
    service = CVService(HashEmbeddings(), MemoryVectorStore())
    chunks = service.ingest(ROOT / "sample_cv.md", cv_id)
    profile = fixture_profile(cv_id, chunks)
    return analyze_jobs(fixture_jobs(), cv_id, service, profile=profile)


def run_live(cv_id: str):
    from src.llm.openai_provider import OpenAIProvider
    from scripts.cv import live_service
    service, client = live_service()
    provider = OpenAIProvider(client, os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    # Intentional paid API usage: extract sourced profile once, then explain each job.
    return analyze_jobs(fixture_jobs(), cv_id, service,
                        facts_provider=provider, explanation_provider=provider)


def main():
    parser = argparse.ArgumentParser(description="Grounded, deterministic Job-CV ranking")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="Synthetic local fixture; no external services")
    mode.add_argument("--live", action="store_true", help="Existing Postgres CV, paid OpenAI calls")
    parser.add_argument("--cv-id", help="Required for --live; CV must already be ingested")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--explain", action="store_true",
                        help="Print a readable job-specific explanation to the terminal")
    parser.add_argument("--output", help="Optional JSON export; use only synthetic data for shared files")
    args = parser.parse_args()
    if args.live and not args.cv_id:
        parser.error("--live requires --cv-id")
    if not 1 <= args.top <= 50:
        parser.error("--top must be in 1..50")
    load_dotenv()
    try:
        matches = run_offline() if args.offline else run_live(args.cv_id)
        print("Mode:", "offline fixture (lexical hash; not semantic)" if args.offline else "live")
        for index, item in enumerate(matches[:args.top], 1):
            print(f"{index}. {item.role} [{item.job_id}] | score {item.score.value:.2f}/100 "
                  f"| evidenced {item.score.evidenced_requirements}/{item.score.total_requirements}")
            for comp in item.comparisons:
                print(f"   {comp.kind}: {comp.requirement} -> {comp.status}")
            if args.explain:
                print("   Explanation:", render_match_explanation(item))
        if args.output:
            if args.live:
                raise ValueError("Live mode does not export potentially private CV evidence")
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([item.model_dump() for item in matches], indent=2) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        # Do not echo raw CV, LLM exception payload, or DB credentials.
        print(f"Matching failed ({type(exc).__name__}). Check fixture integrity, DB and API configuration.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
