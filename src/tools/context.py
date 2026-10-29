"""Application-owned dependencies passed to every tool.

Scoping decisions (which CV, which catalog, where to save) live here and are
never taken from model-generated arguments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.matching.schemas import CandidateProfile
from src.rag.service import CVService
from src.tools.jobs import FixtureJobCatalog, JobCatalog
from src.tools.saved_jobs import MemorySavedJobStore, SavedJobStore


@dataclass
class ToolContext:
    catalog: JobCatalog
    cv_service: CVService
    cv_id: str
    profile_loader: Callable[[], CandidateProfile]
    saved_jobs: SavedJobStore
    _profile: CandidateProfile | None = field(default=None, init=False, repr=False)

    def profile(self) -> CandidateProfile:
        """Load the sourced candidate profile once (live loaders may call an LLM)."""
        if self._profile is None:
            profile = self.profile_loader()
            if profile.cv_id != self.cv_id:
                raise ValueError("Profile CV ID must match the context CV ID")
            self._profile = profile
        return self._profile


def ensure_ingested(service: CVService, path, cv_id: str):
    """Ingest a CV unless a persistent store already holds the identical document and embedding setup.

    Avoids re-embedding (and paying for) the same CV on every restart.
    """
    from src.rag.ingestion import read_cv
    document_info = getattr(service.store, "document_info", None)
    if document_info is not None:
        stored = document_info(cv_id)
        current = (read_cv(path, cv_id).content_hash, service.embeddings.model, service.embeddings.dimensions)
        if stored is not None and tuple(stored) == current:
            chunks = service.store.list_chunks(cv_id)
            if chunks:
                return chunks
    return service.ingest(path, cv_id)


def build_synthetic_context(service: CVService, saved_jobs: SavedJobStore | None = None,
                            cv_id: str = "synthetic-demo") -> ToolContext:
    """Synthetic CV with its human-labeled profile and the fixture catalog, on any CV service/store."""
    from src.matching.fixtures import ROOT, fixture_profile

    chunks = ensure_ingested(service, ROOT / "sample_cv.md", cv_id)
    return ToolContext(catalog=FixtureJobCatalog(), cv_service=service, cv_id=cv_id,
                       profile_loader=lambda: fixture_profile(cv_id, chunks),
                       saved_jobs=saved_jobs or MemorySavedJobStore())


def build_offline_context(saved_jobs: SavedJobStore | None = None,
                          cv_id: str = "synthetic-demo") -> ToolContext:
    """Synthetic CV, lexical hash embeddings and fixture catalog; no network."""
    from src.rag.embeddings import HashEmbeddings
    from src.rag.store import MemoryVectorStore

    return build_synthetic_context(CVService(HashEmbeddings(), MemoryVectorStore()), saved_jobs, cv_id)
