"""Explicit, human-labeled *synthetic* evidence for a zero-API offline demonstration.

Not a general CV parser. The live path uses extract_candidate_profile().
"""
from __future__ import annotations

import json
from pathlib import Path
from src.extraction.schemas import JobPosting
from src.matching.facts import validate_facts
from src.matching.requirements import prepare_job
from src.matching.schemas import (CandidateFacts, CandidateProfile, Evidence,
                                  SkillFact, LanguageFact, EmploymentPeriod)
from src.rag.schemas import CVChunk

ROOT = Path(__file__).resolve().parents[2] / "data" / "fixtures"


def fixture_jobs():
    rows = json.loads((ROOT / "matching_jobs.json").read_text(encoding="utf-8"))
    return [prepare_job(row["id"], JobPosting.model_validate(row["posting"]),
                        row.get("language_levels")) for row in rows]


def fixture_profile(cv_id: str, chunks: list[CVChunk]) -> CandidateProfile:
    labels = json.loads((ROOT / "matching_candidate_facts.json").read_text(encoding="utf-8"))
    def evidence(quote):
        found = [c for c in chunks if quote in c.text]
        if not found:
            raise ValueError("Labeled fixture quote is not preserved in CV chunks")
        return Evidence(chunk_id=found[0].chunk_id, quote=quote)
    profile = CandidateProfile(cv_id=cv_id,
        skills=[SkillFact(name=x["name"], evidence=evidence(x["quote"])) for x in labels["skills"]],
        languages=[LanguageFact(name=x["name"], level=x["level"],
                                evidence=evidence(x["quote"])) for x in labels["languages"]],
        employment=[EmploymentPeriod(start=x["start"], end=x["end"],
                                     evidence=evidence(x["quote"])) for x in labels["employment"]])
    validate_facts(CandidateFacts.model_validate(profile.model_dump(exclude={"cv_id"})), chunks)
    return profile
