"""Opt-in integration test using a disposable CV ID, never clears shared tables."""
import os
import uuid
from pathlib import Path
import pytest

DSN = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="Set TEST_DATABASE_URL for PostgreSQL integration tests")


def test_postgres_roundtrip_replace_isolation_rollback_delete(tmp_path):
    pytest.importorskip("psycopg")
    from src.rag.postgres import PostgresVectorStore
    from src.rag.embeddings import HashEmbeddings
    from src.rag.service import CVService
    from src.rag.ingestion import read_cv, chunk_cv
    store=PostgresVectorStore(DSN); store.initialize()
    service=CVService(HashEmbeddings(),store)
    prefix="integration-"+uuid.uuid4().hex
    ids=[prefix+"-a",prefix+"-b"]
    fixture=Path(__file__).resolve().parents[2]/"data/fixtures/sample_cv.md"
    try:
        first=service.ingest(fixture,ids[0]);service.ingest(fixture,ids[1])
        service.ingest(fixture,ids[0])
        stored = store.list_chunks(ids[0])
        assert len(stored) == len(first)
        assert all(c.cv_id == ids[0] for c in stored)
        section_hits = store.search(ids[0], HashEmbeddings().embed(["German language"])[0],
                                    "lexical-hash-v1", 512, 50, section="Languages")
        assert section_hits and all(h.chunk.section == "Languages" for h in section_hits)
        hits=service.search_cv("German language level",ids[0],50)
        assert len(hits)==len(first)
        assert all(h.chunk.cv_id==ids[0] for h in hits)
        assert hits[0].chunk.section=="Languages"
        with pytest.raises(ValueError,match="configuration"):
            store.search(ids[0],[1.0]*512,"wrong-model",512,3)
        doc=read_cv(fixture,ids[0]);chunks=chunk_cv(doc)
        # Duplicate ordinals trigger a DB constraint AFTER deletion; transaction must roll back.
        chunks[1]=chunks[1].model_copy(update={"ordinal":0})
        with pytest.raises(Exception):
            store.replace(doc,chunks,HashEmbeddings().embed([c.text for c in chunks]),"lexical-hash-v1",512)
        assert len(service.search_cv("German",ids[0],50))==len(first)
        replacement=tmp_path/"replacement.txt";replacement.write_text("Rust programming specialist")
        service.ingest(replacement,ids[0])
        assert len(service.search_cv("Rust",ids[0],50))==1
        store.delete(ids[0]);assert service.search_cv("German",ids[0])==[]
        assert service.search_cv("German",ids[1])
    finally:
        for cv_id in ids:
            store.delete(cv_id)
