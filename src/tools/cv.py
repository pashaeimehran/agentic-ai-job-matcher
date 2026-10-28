"""search_cv tool: a thin wrapper over the existing CV-scoped hybrid retrieval."""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.matching.retrieval import hybrid_search
from src.tools.schemas import CVEvidence, SearchCVArgs, SearchCVResult

if TYPE_CHECKING:
    from src.tools.context import ToolContext


def search_cv(args: SearchCVArgs, ctx: "ToolContext") -> SearchCVResult:
    # cv_id comes from the application context, never from model arguments.
    hits = hybrid_search(ctx.cv_service, ctx.cv_id, args.query, k=args.k, section=args.section)
    evidence = [CVEvidence(chunk_id=h.chunk.chunk_id, section=h.chunk.section, text=h.chunk.text,
                           fusion_score=round(h.fusion_score, 6),
                           vector_cosine=None if h.vector_cosine is None else round(h.vector_cosine, 6),
                           token_coverage=round(h.token_coverage, 6))
                for h in hits]
    message = None if evidence else "No CV evidence was found for this query."
    return SearchCVResult(evidence=evidence, message=message)
