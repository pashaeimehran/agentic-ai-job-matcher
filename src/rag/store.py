from __future__ import annotations

from typing import Protocol
from src.rag.embeddings import cosine, validate_vectors
from src.rag.schemas import CVDocument, CVChunk, SearchHit


class VectorStore(Protocol):
    def replace(self, document: CVDocument, chunks: list[CVChunk], vectors: list[list[float]], model: str, dimensions: int) -> None: ...
    def search(self, cv_id: str, vector: list[float], model: str, dimensions: int, k: int, section: str | None = None) -> list[SearchHit]: ...
    def list_chunks(self, cv_id: str) -> list[CVChunk]: ...
    def delete(self, cv_id: str) -> None: ...


def validate_replacement(document, chunks, vectors, dimensions):
    validate_vectors(vectors, len(chunks), dimensions)
    if not chunks or len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("Chunks must be nonempty with unique IDs.")
    if any(c.cv_id != document.cv_id or document.text[c.start:c.end] != c.text for c in chunks):
        raise ValueError("Chunk does not match its source document.")


class MemoryVectorStore:
    def __init__(self):
        self.documents = {}

    def replace(self, document, chunks, vectors, model, dimensions):
        validate_replacement(document, chunks, vectors, dimensions)
        self.documents[document.cv_id] = (model, dimensions, list(zip(chunks, vectors)))

    def list_chunks(self, cv_id):
        return [c for c, _ in self.documents[cv_id][2]] if cv_id in self.documents else []

    def search(self, cv_id, vector, model, dimensions, k, section=None):
        if cv_id not in self.documents:
            return []
        stored_model, stored_dimensions, entries = self.documents[cv_id]
        if (model, dimensions) != (stored_model, stored_dimensions):
            raise ValueError("Embedding configuration changed. Re-ingest this CV.")
        validate_vectors([vector], 1, dimensions)
        return sorted((SearchHit(chunk=c, score=cosine(vector, v)) for c, v in entries
                       if section is None or c.section.casefold() == section.casefold()),
                      key=lambda h: (-h.score, h.chunk.ordinal))[:k]

    def delete(self, cv_id):
        self.documents.pop(cv_id, None)
