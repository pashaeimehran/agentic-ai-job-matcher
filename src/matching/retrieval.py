"""Simple token-aware hybrid CV retrieval and deterministic reranking."""
from __future__ import annotations

import re
from dataclasses import dataclass
from src.rag.embeddings import validate_vectors
from src.rag.schemas import CVChunk
from src.rag.service import CVService


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w+#.]+", text.casefold()))


@dataclass(frozen=True)
class HybridHit:
    chunk: CVChunk
    fusion_score: float
    vector_cosine: float | None
    token_coverage: float


def hybrid_search(service: CVService, cv_id: str, query: str, k: int = 5,
                  section: str | None = None, rerank: bool = True) -> list[HybridHit]:
    """RRF merges cosine and lexical rankings. Reranking adds query-token coverage.

    This is exact small-CV search, not a learned reranker. Section is filtered
    before fusion, so excluded sections cannot enter the returned evidence.
    """
    if not query.strip() or len(query) > 2000 or not 1 <= k <= 50:
        raise ValueError("Query length must be 1..2000 and k must be 1..50")
    chunks = service.store.list_chunks(cv_id)
    if section:
        chunks = [c for c in chunks if c.section.casefold() == section.casefold()]
    if not chunks:
        return []
    vector = validate_vectors(service.embeddings.embed([query]), 1, service.embeddings.dimensions)[0]
    # Search all available chunks to avoid dropping lexical-only hits from fusion.
    # The existing store uses an exact cosine search, scoped by CV ID.
    # For large CVs, cap the vector candidate pool explicitly to a known limit.
    vector_hits = service.store.search(cv_id, vector, service.embeddings.model,
                                       service.embeddings.dimensions, min(len(service.store.list_chunks(cv_id)), 500),
                                       section=section)
    lexical = _tokens(query)
    coverages = {c.chunk_id: (len(lexical & _tokens(c.text)) / len(lexical) if lexical else 0.0)
                 for c in chunks}
    lexical_hits = sorted((c for c in chunks if coverages[c.chunk_id] > 0),
                          key=lambda c: (-coverages[c.chunk_id], c.ordinal))
    fusion = {}
    raw_vectors = {}
    for rank, hit in enumerate(vector_hits, 1):
        fusion[hit.chunk.chunk_id] = fusion.get(hit.chunk.chunk_id, 0.0) + 1 / (60 + rank)
        raw_vectors[hit.chunk.chunk_id] = hit.score
    for rank, chunk in enumerate(lexical_hits, 1):
        fusion[chunk.chunk_id] = fusion.get(chunk.chunk_id, 0.0) + 1 / (60 + rank)
    found = [HybridHit(chunk=c, fusion_score=fusion[c.chunk_id],
                       vector_cosine=raw_vectors.get(c.chunk_id), token_coverage=coverages[c.chunk_id])
             for c in chunks if c.chunk_id in fusion]
    if rerank:
        found.sort(key=lambda hit: (-(hit.fusion_score + 0.02 * hit.token_coverage), hit.chunk.ordinal))
    else:
        found.sort(key=lambda hit: (-hit.fusion_score, hit.chunk.ordinal))
    return found[:k]
