"""Optional, paid Ragas evaluation of LIVE grounded explanations.

Evaluates the top-N deterministically ranked synthetic jobs. Only those jobs
get paid LLM narratives. Reports scores without logging CV or LLM text.
Run only with an ingested synthetic CV; never use private personal data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from dotenv import load_dotenv


def evaluation_question(analysis) -> str:
    """Fixed between versions for an interpretable evaluation question."""
    return f"Explain the candidate's evidenced fit, gaps and unknowns for {analysis.role}."


def select_ranked_matches(jobs, cv_id, service, profile, provider, limit):
    """Rank all jobs with Python first; generate narratives for at most limit.

    Retrieval still runs across jobs to score each one, but LLM explanation
    calls are limited to the chosen top-N. Do not silently omit failed jobs.
    """
    from src.matching.service import analyze_jobs, analyze_match

    ranked = analyze_jobs(jobs, cv_id, service, profile=profile)
    jobs_by_id = {job.job_id: job for job in jobs}
    if len(jobs_by_id) != len(jobs):
        raise ValueError("Duplicate evaluation job IDs")
    return [analyze_match(jobs_by_id[item.job_id], profile, service,
                          provider=provider) for item in ranked[:limit]]


def evidence_contexts(analysis, chunks):
    """Include retrieved chunks AND source chunks used for scored claims.

    The narrative is grounded in comparison evidence, which may originate from
    CV chunks outside the retrieved top-k. Contexts therefore need to include
    those cited source chunks for a valid faithfulness assessment.
    """
    identifiers = list(analysis.retrieved_chunk_ids)
    for comparison in analysis.comparisons:
        identifiers.extend(e.chunk_id for e in comparison.evidence)
    for claim in analysis.explanation:
        identifiers.extend(e.chunk_id for e in claim.evidence)
    identifiers = list(dict.fromkeys(identifiers))
    if any(chunk_id not in chunks for chunk_id in identifiers):
        raise ValueError("Evaluation references a missing source chunk")
    return [chunks[chunk_id].text for chunk_id in identifiers]



def load_evaluation_profile(source, provider, cv_id, chunks):
    """Use live extraction or verified human-labeled facts for a synthetic CV.

    Labeled fixture mode is an explanation-only evaluation: it deliberately
    avoids repeated and variable LLM candidate fact extraction. The stored
    document must match the *entire* versioned synthetic CV, including default
    chunk boundaries, before human-labeled candidate facts may be used.
    """
    from src.matching.facts import extract_candidate_profile
    if source == "extracted":
        return extract_candidate_profile(provider, cv_id, chunks)
    if source != "labeled-fixture":
        raise ValueError("Unknown evaluation profile source")

    from src.matching.fixtures import ROOT, fixture_profile
    from src.rag.ingestion import chunk_cv, read_cv
    expected = chunk_cv(read_cv(ROOT / "sample_cv.md", cv_id))
    if len(chunks) != len(expected) or any(
        actual.model_dump() != reference.model_dump()
        for actual, reference in zip(chunks, expected)
    ):
        raise ValueError("Labeled fixture mode requires exact stored sample CV and default chunks")
    return fixture_profile(cv_id, chunks)


def main():
    parser = argparse.ArgumentParser(description="Paid Ragas evaluation on synthetic stored CV")
    parser.add_argument("--cv-id", required=True, help="ID of an ingested synthetic test CV")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--profile-source", choices=("extracted", "labeled-fixture"),
                        default="extracted",
                        help="Use live candidate extraction or verified synthetic fixture facts")
    parser.add_argument("--output", default="evaluation/results/matching_ragas_live.json")
    args = parser.parse_args()
    if not 1 <= args.limit <= 6:
        parser.error("--limit must be in 1..6")
    load_dotenv()
    stage = "initialization"
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy
        from scripts.cv import live_service
        from src.llm.openai_provider import OpenAIProvider
        from src.matching.fixtures import fixture_jobs
        from src.matching.service import render_match_explanation

        stage = "loading CV"
        service, client = live_service()
        chunks = {x.chunk_id: x for x in service.store.list_chunks(args.cv_id)}
        if not chunks:
            raise ValueError("Synthetic CV is not ingested")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        provider = OpenAIProvider(client, model)
        stage = "candidate facts extraction" if args.profile_source == "extracted" else "labeled fixture validation"
        profile = load_evaluation_profile(args.profile_source, provider, args.cv_id, list(chunks.values()))

        stage = "ranking and selected narrative generation"
        matches = select_ranked_matches(
            fixture_jobs(), args.cv_id, service, profile, provider, args.limit
        )
        if not matches:
            raise ValueError("No jobs selected for evaluation")

        stage = "Ragas initialization"
        evaluator_client = AsyncOpenAI(timeout=60.0, max_retries=2)
        llm = llm_factory(model, client=evaluator_client)
        embeddings = embedding_factory(
            "openai", model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            client=evaluator_client
        )
        faithfulness = Faithfulness(llm=llm)
        relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

        records = []
        for result in matches:
            question = evaluation_question(result)
            response = render_match_explanation(result)
            contexts = evidence_contexts(result, chunks)
            if not contexts or not response:
                raise ValueError("Cannot evaluate empty answer or contexts")
            stage = f"Ragas faithfulness for {result.job_id}"
            f = float(faithfulness.score(
                user_input=question, response=response, retrieved_contexts=contexts
            ).value)
            stage = f"Ragas relevancy for {result.job_id}"
            r = float(relevancy.score(user_input=question, response=response).value)
            if not math.isfinite(f) or not math.isfinite(r):
                raise ValueError("Nonfinite Ragas metric")
            records.append({"job_id": result.job_id,
                            "faithfulness": round(f, 4),
                            "answer_relevancy": round(r, 4),
                            "context_chunks": len(contexts)})

        report = {
            "mode": "paid_live_ragas",
            "scope": "Synthetic CV only; Ragas judge estimates, not ground truth",
            "method": "deterministic_rank_then_explain_top_n",
            "candidate_profile_source": args.profile_source,
            "context_policy": "retrieved_plus_approved_cited_source_chunks",
            "jobs": len(records),
            "metrics": {
                "faithfulness": round(sum(r["faithfulness"] for r in records) / len(records), 4),
                "answer_relevancy": round(sum(r["answer_relevancy"] for r in records) / len(records), 4),
            },
            "per_job": records,
        }
        stage = "writing evaluation report"
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Faithfulness:", report["metrics"]["faithfulness"])
        print("Answer relevancy:", report["metrics"]["answer_relevancy"])
        print("Numbers-only Ragas report:", output)
        return 0
    except Exception as exc:
        # Do not print exception messages: they might contain CV text, prompts,
        # credential-bearing DB connection strings, or provider response data.
        print(f"Ragas evaluation failed at {stage} ({type(exc).__name__}).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
