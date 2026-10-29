from __future__ import annotations

import json
import psycopg
from psycopg.rows import dict_row
from src.rag.embeddings import validate_vectors
from src.rag.schemas import CVChunk, SearchHit
from src.rag.store import validate_replacement


class PostgresVectorStore:
    """Exact cosine search scoped to one CV; each operation owns its transaction."""
    def __init__(self, dsn: str):
        self.dsn = dsn

    def connect(self):
        return psycopg.connect(self.dsn, connect_timeout=10, row_factory=dict_row)

    def initialize(self):
        """Apply the full application schema (CV tables, saved jobs, cost log); see src.db."""
        from src.db import initialize_database
        initialize_database(self.dsn)

    def replace(self, document, chunks, vectors, model, dimensions):
        validate_replacement(document, chunks, vectors, dimensions)
        with self.connect() as conn:
            # The upsert locks this document, serializing simultaneous replacements.
            conn.execute("""INSERT INTO cv_documents
                (cv_id, source_name, content_hash, embedding_model, dimensions)
                VALUES (%s,%s,%s,%s,%s) ON CONFLICT (cv_id) DO UPDATE SET
                source_name=EXCLUDED.source_name, content_hash=EXCLUDED.content_hash,
                embedding_model=EXCLUDED.embedding_model, dimensions=EXCLUDED.dimensions,
                updated_at=now()""", (document.cv_id, document.source_name, document.content_hash, model, dimensions))
            conn.execute("DELETE FROM cv_chunks WHERE cv_id=%s", (document.cv_id,))
            with conn.cursor() as cur:
                cur.executemany("""INSERT INTO cv_chunks
                    (chunk_id,cv_id,section,ordinal,text,start_offset,end_offset,embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)""",
                    [(c.chunk_id,c.cv_id,c.section,c.ordinal,c.text,c.start,c.end,json.dumps(v))
                     for c,v in zip(chunks,vectors)])

    def document_info(self, cv_id):
        """(content_hash, embedding_model, dimensions) of a stored CV, or None."""
        with self.connect() as conn:
            row = conn.execute("SELECT content_hash,embedding_model,dimensions FROM cv_documents WHERE cv_id=%s",
                               (cv_id,)).fetchone()
        return None if row is None else (row["content_hash"], row["embedding_model"], row["dimensions"])

    def list_chunks(self, cv_id):
        with self.connect() as conn:
            rows = conn.execute("""SELECT chunk_id,cv_id,section,ordinal,text,
                start_offset AS start,end_offset AS end FROM cv_chunks
                WHERE cv_id=%s ORDER BY ordinal""", (cv_id,)).fetchall()
        return [CVChunk(**row) for row in rows]

    def search(self, cv_id, vector, model, dimensions, k, section=None):
        validate_vectors([vector], 1, dimensions)
        with self.connect() as conn:
            # Share lock keeps metadata and chunks consistent across both queries.
            doc = conn.execute("SELECT embedding_model,dimensions FROM cv_documents WHERE cv_id=%s FOR SHARE", (cv_id,)).fetchone()
            if doc is None:
                return []
            if (model, dimensions) != (doc["embedding_model"], doc["dimensions"]):
                raise ValueError("Embedding configuration changed. Re-ingest this CV.")
            rows = conn.execute("""SELECT chunk_id,cv_id,section,ordinal,text,
                start_offset AS start,end_offset AS end,
                1 - (embedding <=> %s::vector) AS score FROM cv_chunks
                WHERE cv_id=%s AND (%s::text IS NULL OR lower(section)=lower(%s::text))
                ORDER BY embedding <=> %s::vector, ordinal LIMIT %s""",
                (json.dumps(vector), cv_id, section, section, json.dumps(vector), k)).fetchall()
        return [SearchHit(chunk=CVChunk(**{k:v for k,v in row.items() if k != "score"}),
                          score=max(-1.0,min(1.0,row["score"]))) for row in rows]

    def delete(self, cv_id):
        with self.connect() as conn:
            conn.execute("DELETE FROM cv_documents WHERE cv_id=%s", (cv_id,))
