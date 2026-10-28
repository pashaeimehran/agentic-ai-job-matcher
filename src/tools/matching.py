"""calculate_match tool: delegates scoring and ranking to src.matching.

No score is computed here. analyze_jobs() validates the profile against the
stored chunks, runs score_match() and orders with rank_analyses().
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.matching.service import analyze_jobs
from src.tools.errors import ToolNotFoundError, ToolPreconditionError
from src.tools.schemas import CalculateMatchArgs, CalculateMatchResult, JobMatch

if TYPE_CHECKING:
    from src.tools.context import ToolContext


def calculate_match(args: CalculateMatchArgs, ctx: "ToolContext") -> CalculateMatchResult:
    missing = [job_id for job_id in args.job_ids if ctx.catalog.get(job_id) is None]
    if missing:
        raise ToolNotFoundError(f"Unknown job_id(s): {', '.join(missing)}. Use search_jobs first.")
    if not ctx.cv_service.store.list_chunks(ctx.cv_id):
        raise ToolPreconditionError("The user's CV has not been ingested.")
    jobs = [ctx.catalog.get(job_id).requirements() for job_id in args.job_ids]
    analyses = analyze_jobs(jobs, ctx.cv_id, ctx.cv_service, profile=ctx.profile())
    return CalculateMatchResult(matches=[
        JobMatch(rank=rank, job_id=a.job_id, role=a.role, score=a.score,
                 supported=[c.requirement for c in a.comparisons if c.status == "supported"],
                 unmet=[c.requirement for c in a.comparisons if c.status == "unmet"],
                 unknown=[c.requirement for c in a.comparisons if c.status == "unknown"],
                 comparisons=a.comparisons, retrieved_chunk_ids=a.retrieved_chunk_ids)
        for rank, a in enumerate(analyses, 1)])
