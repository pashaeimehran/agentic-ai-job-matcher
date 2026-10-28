"""HTTP request and response models.

Search request models subclass the tool argument models, so field limits are
defined once (in src.tools.schemas) and the tool registry re-validates them.
Result items reuse the tool result models unchanged.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.tools.schemas import CVEvidence, JobDetailsResult, JobMatch, JobSummary, SearchCVArgs, SearchJobsArgs


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"
    version: str
    agent: Literal["langgraph"] = "langgraph"
    chat_available: bool = Field(description="False when no LLM provider is configured.")
    tools: list[str]
    catalog_jobs: int = Field(ge=0)


# --- Jobs ----------------------------------------------------------------------

class JobListItem(ApiModel):
    job_id: str
    role: str
    location: str | None
    seniority: str | None
    skills: list[str]
    source: str


class JobListResponse(ApiModel):
    request_id: str
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    jobs: list[JobListItem]


class JobDetailsResponse(ApiModel):
    request_id: str
    job: JobDetailsResult


# --- Search --------------------------------------------------------------------

class JobSearchRequest(SearchJobsArgs):
    target: Literal["jobs"]


class CVSearchRequest(SearchCVArgs):
    target: Literal["cv"]


SearchRequest = Annotated[JobSearchRequest | CVSearchRequest, Field(discriminator="target")]


class JobSearchResponse(ApiModel):
    request_id: str
    target: Literal["jobs"] = "jobs"
    status: Literal["ok", "empty"]
    message: str | None = None
    total_matches: int = Field(ge=0)
    jobs: list[JobSummary]


class CVSearchResponse(ApiModel):
    request_id: str
    target: Literal["cv"] = "cv"
    status: Literal["ok", "empty"]
    message: str | None = None
    evidence: list[CVEvidence]


SearchResponse = Annotated[JobSearchResponse | CVSearchResponse, Field(discriminator="target")]


# --- Analysis ------------------------------------------------------------------

class AnalyzeJobRequest(ApiModel):
    job_id: str = Field(min_length=1, max_length=120, description="Catalog job_id to analyze against the CV.")


class AnalyzeJobResponse(ApiModel):
    request_id: str
    score_source: Literal["deterministic"] = "deterministic"
    match: JobMatch
    scoring_note: str


# --- Chat ----------------------------------------------------------------------

class ChatRequest(ApiModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$",
                                  description="Continue an earlier conversation; omit to start a new one.")


class ToolCallSummary(ApiModel):
    step: int
    tool: str
    status: Literal["ok", "empty", "error"]
    error_code: str | None = None
    attempts: int


class ChatResponse(ApiModel):
    request_id: str
    run_id: str
    thread_id: str
    status: Literal["completed"] = "completed"
    answer: str
    model_turns: int
    tool_calls: list[ToolCallSummary]
    ungrounded_scores: list[str] = Field(
        description="Score-like numbers in the answer that no calculate_match call produced.")
