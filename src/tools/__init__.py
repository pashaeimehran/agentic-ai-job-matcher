"""Framework-independent tool layer: schemas, registry and implementations."""
from __future__ import annotations

from src.tools.cv import search_cv
from src.tools.jobs import get_job_details, search_jobs
from src.tools.matching import calculate_match
from src.tools.registry import RetryPolicy, Tool, ToolRegistry
from src.tools.saved_jobs import save_job
from src.tools.schemas import (CalculateMatchArgs, GetJobDetailsArgs, SaveJobArgs,
                               SearchCVArgs, SearchJobsArgs, ToolOutcome)


def default_registry() -> ToolRegistry:
    return ToolRegistry([
        Tool("search_jobs",
             "Search the job catalog by role or skill keywords, optionally filtered by location. "
             "Returns job_ids and short summaries; relevance is keyword relevance, not CV fit.",
             SearchJobsArgs, search_jobs),
        Tool("get_job_details",
             "Get the full structured posting for one job_id: role, skills, experience years, "
             "languages and levels, seniority and location.",
             GetJobDetailsArgs, get_job_details),
        Tool("search_cv",
             "Retrieve verbatim excerpts from the user's own CV that are relevant to a query. "
             "Use for questions about the user's experience or skills.",
             SearchCVArgs, search_cv),
        Tool("calculate_match",
             "Calculate deterministic CV compatibility for one to ten job_ids. Returns ranked "
             "scores (0-100), supported/unmet/unknown requirements and evidence. The only "
             "valid source of match scores.",
             CalculateMatchArgs, calculate_match),
        Tool("save_job",
             "Save one job_id to the user's saved-jobs list, with an optional note. Only call "
             "when the user explicitly asks to save a specific job.",
             SaveJobArgs, save_job),
    ])


__all__ = ["RetryPolicy", "Tool", "ToolOutcome", "ToolRegistry", "default_registry"]
