"""Strict argument and result contracts for the tool layer.

Argument models are the only path from model-generated JSON to application
code: unknown keys are rejected and types are not coerced.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.extraction.schemas import JobPosting
from src.matching.schemas import CEFR, MatchScore, RequirementResult
from src.tools.errors import ErrorCode


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def is_empty(self) -> bool:
        return False


# --- Arguments -----------------------------------------------------------------

class SearchJobsArgs(StrictArgs):
    query: str = Field(min_length=1, max_length=200,
                       description="Role title or skills to search for, e.g. 'AI Engineer'.")
    location: str | None = Field(default=None, min_length=1, max_length=100,
                                 description="City, 'Remote', or country such as 'Germany'.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of jobs to return.")


class GetJobDetailsArgs(StrictArgs):
    job_id: str = Field(min_length=1, max_length=120, description="A job_id returned by search_jobs.")


class SearchCVArgs(StrictArgs):
    query: str = Field(min_length=1, max_length=500,
                       description="What evidence to look for in the user's CV, e.g. 'Python experience'.")
    k: int = Field(default=3, ge=1, le=10, description="Maximum number of CV excerpts.")
    section: str | None = Field(default=None, min_length=1, max_length=120,
                                description="Optional exact CV section heading to restrict the search.")


class CalculateMatchArgs(StrictArgs):
    job_ids: list[str] = Field(min_length=1, max_length=10,
                               description="One to ten job_ids to score against the user's CV.")

    @field_validator("job_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or len(v) > 120 for v in values):
            raise ValueError("job_ids must be non-empty strings of at most 120 characters")
        if len(set(values)) != len(values):
            raise ValueError("job_ids must be unique")
        return values


class SaveJobArgs(StrictArgs):
    job_id: str = Field(min_length=1, max_length=120, description="The job_id to save.")
    note: str | None = Field(default=None, max_length=500, description="Optional short user note.")


# --- Results -------------------------------------------------------------------

class JobSummary(BaseModel):
    job_id: str
    role: str
    location: str | None
    seniority: str | None
    skills: list[str]
    relevance: float = Field(ge=0, description="Deterministic keyword relevance, not a CV match score.")


class SearchJobsResult(ToolResult):
    jobs: list[JobSummary]
    total_matches: int = Field(ge=0)
    message: str | None = None

    def is_empty(self) -> bool:
        return not self.jobs


class JobDetailsResult(ToolResult):
    job_id: str
    source: str
    posting: JobPosting
    language_levels: dict[str, CEFR] = Field(default_factory=dict)


class CVEvidence(BaseModel):
    chunk_id: str
    section: str
    text: str
    fusion_score: float
    vector_cosine: float | None
    token_coverage: float


class SearchCVResult(ToolResult):
    evidence: list[CVEvidence]
    message: str | None = None

    def is_empty(self) -> bool:
        return not self.evidence


class JobMatch(BaseModel):
    rank: int = Field(ge=1)
    job_id: str
    role: str
    score: MatchScore
    supported: list[str]
    unmet: list[str]
    unknown: list[str]
    comparisons: list[RequirementResult]
    retrieved_chunk_ids: list[str]


class CalculateMatchResult(ToolResult):
    """Scores are produced by src.matching.scoring; the model may only quote them."""
    matches: list[JobMatch]
    scoring_note: str = ("score.value is calculated by deterministic Python code (0-100). "
                         "Quote it exactly; never estimate or adjust it.")


class SaveJobResult(ToolResult):
    job_id: str
    status: Literal["saved", "already_saved"]
    saved_at: str


# --- Envelope ------------------------------------------------------------------

class ToolError(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool
    details: list[dict[str, Any]] = Field(default_factory=list)


class ToolOutcome(BaseModel):
    """Uniform observation for every tool execution, successful or not."""
    tool_name: str
    status: Literal["ok", "empty", "error"]
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    attempts: int = Field(ge=0)
    duration_ms: float = Field(ge=0)

    @property
    def ok(self) -> bool:
        return self.status != "error"
