"""Strict, source-attributed contracts for candidate facts and job comparisons."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


CEFR = Literal["A1", "A2", "B1", "B2", "C1", "C2", "native"]
Status = Literal["supported", "unmet", "unknown"]
Kind = Literal["skill", "language", "experience"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Evidence(StrictModel):
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class SkillFact(StrictModel):
    name: str = Field(min_length=1)
    evidence: Evidence


class LanguageFact(StrictModel):
    name: str = Field(min_length=1)
    level: CEFR | None = None
    evidence: Evidence


class EmploymentPeriod(StrictModel):
    """Explicit YYYY-MM boundaries only; unknown boundaries are excluded."""
    start: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    end: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    evidence: Evidence

    @model_validator(mode="after")
    def chronological(self):
        if self.end < self.start:
            raise ValueError("Employment end precedes start")
        return self


class CandidateFacts(StrictModel):
    """Per-batch extraction result. Validated against source chunks in Python."""
    skills: list[SkillFact] = Field(default_factory=list)
    languages: list[LanguageFact] = Field(default_factory=list)
    employment: list[EmploymentPeriod] = Field(default_factory=list)


class CandidateProfile(CandidateFacts):
    cv_id: str = Field(min_length=1)


class LanguageRequirement(StrictModel):
    name: str = Field(min_length=1)
    minimum_level: CEFR | None = None


class JobRequirements(StrictModel):
    job_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    skills: list[str] = Field(default_factory=list)
    experience_years: int | None = Field(default=None, ge=0, le=50)
    languages: list[LanguageRequirement] = Field(default_factory=list)
    location: str | None = None


class JobLanguageEvidence(StrictModel):
    name: str = Field(min_length=1)
    level: CEFR
    quote: str = Field(min_length=1)


class JobLanguageExtraction(StrictModel):
    requirements: list[JobLanguageEvidence] = Field(default_factory=list)


class RequirementResult(StrictModel):
    kind: Kind
    requirement: str
    status: Status
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)


class ComponentScore(StrictModel):
    name: Literal["skills", "experience", "languages", "retrieval"]
    raw: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    weighted_contribution: float = Field(ge=0, le=100)


class MatchScore(StrictModel):
    value: float = Field(ge=0, le=100)
    components: list[ComponentScore]
    evidenced_requirements: int = Field(ge=0)
    total_requirements: int = Field(ge=0)
    retrieval_cosine: float | None = Field(default=None, ge=-1, le=1)


class NarrativeClaim(StrictModel):
    kind: Kind
    requirement: str
    status: Status
    text: str = Field(min_length=1, max_length=400)
    evidence: list[Evidence] = Field(default_factory=list)


class NarrativeDraft(StrictModel):
    claims: list[NarrativeClaim] = Field(default_factory=list, max_length=12)


class MatchAnalysis(StrictModel):
    job_id: str
    role: str
    score: MatchScore
    comparisons: list[RequirementResult]
    explanation: list[NarrativeClaim]
    retrieved_chunk_ids: list[str]
