"""Database schema initialization and readiness checks.

The repository's mechanism is idempotent SQL files (CREATE ... IF NOT EXISTS)
applied in one transaction. Applying them again is safe; there is no separate
migration framework because the schema only grows by new tables.
"""
from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parent

SCHEMA_FILES = (
    SRC / "rag" / "schema.sql",            # CV documents and pgvector chunks
    SRC / "tools" / "saved_jobs.sql",      # saved jobs
    SRC / "observability" / "schema.sql",  # LLM usage / cost log
)
REQUIRED_TABLES = ("cv_documents", "cv_chunks", "saved_jobs", "cost_log")


class SchemaInitializationError(RuntimeError):
    """Raised when the schema cannot be applied; the message never contains the DSN."""


def initialize_database(dsn: str) -> None:
    import psycopg
    sql = "\n".join(path.read_text(encoding="utf-8") for path in SCHEMA_FILES)
    try:
        with psycopg.connect(dsn, connect_timeout=10) as conn:
            conn.execute(sql)
    except psycopg.Error as exc:
        raise SchemaInitializationError(f"Database schema initialization failed ({type(exc).__name__}).") from None


def missing_tables(dsn: str) -> list[str]:
    """Required tables that do not exist; raises psycopg errors if the database is unreachable."""
    import psycopg
    with psycopg.connect(dsn, connect_timeout=3) as conn:
        rows = conn.execute("SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema = current_schema() AND table_name = ANY(%s)",
                            (list(REQUIRED_TABLES),)).fetchall()
    present = {row[0] for row in rows}
    return [table for table in REQUIRED_TABLES if table not in present]
