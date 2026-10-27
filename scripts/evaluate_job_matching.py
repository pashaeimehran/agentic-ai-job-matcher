"""Reproducible offline evaluation of *synthetic* job/CV matching.

Metrics refer to a single labeled synthetic CV, not employment suitability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.match_jobs import run_offline
from src.matching.fixtures import ROOT, fixture_jobs, fixture_profile
from src.matching.retrieval import hybrid_search
from src.matching.scoring import rank_analyses, score_match
from src.rag.embeddings import HashEmbeddings
from src.rag.service import CVService
from src.rag.store import MemoryVectorStore


def evaluate() -> dict:
    labels = json.loads((ROOT / "matching_jobs.json").read_text(encoding="utf-8"))
    jobs = fixture_jobs()
    service = CVService(HashEmbeddings(), MemoryVectorStore())
    cv_id = "evaluation-job-cv"
    chunks = service.ingest(ROOT / "sample_cv.md", cv_id)
    profile = fixture_profile(cv_id, chunks)
    matches = run_offline()
    by_id = {x.job_id: x for x in matches}
    status_checks = 0
    status_correct = 0
    retrieval_metrics = {name: {"hits": 0, "precision_sum": 0.0, "recall_sum": 0.0}
                         for name in ("vector", "hybrid")}
    results = []
    for row, job in zip(labels, jobs):
        match = by_id[job.job_id]
        actual = {f"{c.kind}:{c.requirement}": c.status for c in match.comparisons}
        expected = row["expected_status"]
        status_checks += len(expected)
        status_correct += sum(actual.get(key) == wanted for key, wanted in expected.items())
        label_chunks = {c.chunk_id for c in chunks if c.section == row["reference_section"]
                        and row["reference_quote"] in c.text}
        if not label_chunks:
            raise ValueError("Reference quote not preserved by chunking")
        query = ", ".join([job.role, *job.skills, *[f"{x.name} {x.minimum_level or ''}" for x in job.languages]])
        vector = service.search_cv(query, cv_id, k=3)
        hybrid = hybrid_search(service, cv_id, query, k=3)
        found = {}
        for name, retrieved in (("vector", vector), ("hybrid", hybrid)):
            ids = {h.chunk.chunk_id for h in retrieved}
            relevant = len(ids & label_chunks)
            retrieval_metrics[name]["hits"] += int(relevant > 0)
            retrieval_metrics[name]["precision_sum"] += relevant / len(retrieved) if retrieved else 0
            retrieval_metrics[name]["recall_sum"] += relevant / len(label_chunks)
            found[name] = relevant > 0
        calculated, _ = score_match(job, profile, match.score.retrieval_cosine)
        if calculated != match.score:
            raise AssertionError("Match scoring changed on repeated calculation")
        results.append({"job_id": job.job_id, "score": match.score.value,
                        "status_correct": sum(actual.get(key) == value for key, value in expected.items()),
                        "status_total": len(expected), "retrieval_hit_at_3": found})
    n = len(jobs)
    summary = {name: {"hit_at_3": round(x["hits"] / n, 6),
                      "precision_at_3": round(x["precision_sum"] / n, 6),
                      "recall_at_3": round(x["recall_sum"] / n, 6)}
               for name, x in retrieval_metrics.items()}
    return {"mode": "offline_lexical_hash_synthetic",
            "limitations": "Single public synthetic CV; lexical hash is not a semantic embedding; "
                           "these labels are development data, not independent hiring decisions.",
            "job_cases": n, "labeled_requirement_checks": status_checks,
            "requirement_status_accuracy": round(status_correct / status_checks, 6),
            "score_reproducibility": True,
            "ranked_job_ids": [item.job_id for item in matches],
            "retrieval": summary, "cases": results}


def main():
    parser = argparse.ArgumentParser(description="Run offline job/CV matching regression benchmark")
    parser.add_argument("--output", default="evaluation/results/matching_offline.json")
    args = parser.parse_args()
    result = evaluate()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Status accuracy: {result['requirement_status_accuracy']:.1%} "
          f"({result['labeled_requirement_checks']} checks)")
    for mode, metrics in result["retrieval"].items():
        print(f"{mode}: hit@3={metrics['hit_at_3']:.1%} "
              f"precision@3={metrics['precision_at_3']:.1%} recall@3={metrics['recall_at_3']:.1%}")
    print("Ranking:", ", ".join(result["ranked_job_ids"]))
    print("Report:", path)
    return 0 if result["requirement_status_accuracy"] == 1.0 and result["score_reproducibility"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
