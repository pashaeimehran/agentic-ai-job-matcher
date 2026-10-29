"""cost_log persistence: SQL contract with a fake connection, plus an opt-in PostgreSQL round trip."""
import os
import uuid

import pytest

from src.db import REQUIRED_TABLES, SCHEMA_FILES
from src.observability.metrics import CompositeUsageSink, LLMCallRecord, MemoryUsageSink
from src.observability.postgres import INSERT_SQL, PostgresUsageSink


class FakeConnection:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.log.append((sql, params))


def record(**overrides):
    values = {"run_id": "run-1", "request_id": "req-1", "provider": "openai", "model": "gpt-4o-mini",
              "operation": "complete", "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
              "estimated_cost_usd": 0.000027, "latency_ms": 812.5, "status": "ok"}
    return LLMCallRecord(**{**values, **overrides})


def test_record_is_inserted_with_all_columns_and_no_content():
    log = []
    sink = PostgresUsageSink("postgresql://unused", connect=lambda dsn: FakeConnection(log))
    rec = record()
    sink.record(rec)
    (sql, params), = log
    assert sql == INSERT_SQL and "ON CONFLICT (record_id) DO NOTHING" in sql
    assert params[0] == rec.record_id and "gpt-4o-mini" in params and 0.000027 in params
    assert len(params) == sql.count("%s")


def test_database_outage_does_not_fail_or_block_other_sinks():
    def down(dsn):
        raise ConnectionError("database unavailable")
    memory = MemoryUsageSink()
    CompositeUsageSink([PostgresUsageSink("postgresql://x", connect=down), memory]).record(record())
    assert len(memory.records) == 1


def test_schema_files_define_every_required_table():
    sql = "\n".join(path.read_text(encoding="utf-8") for path in SCHEMA_FILES)
    for table in REQUIRED_TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    assert "prompt" not in sql.split("CREATE TABLE IF NOT EXISTS cost_log")[1].split("prompt_version")[0]


DSN = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not DSN, reason="Set TEST_DATABASE_URL for PostgreSQL integration tests")
def test_postgres_cost_log_round_trip():
    import psycopg
    from src.db import initialize_database, missing_tables
    initialize_database(DSN)
    initialize_database(DSN)  # idempotent
    assert missing_tables(DSN) == []
    run_id = "cost-it-" + uuid.uuid4().hex
    sink = PostgresUsageSink(DSN)
    first = record(run_id=run_id)
    sink.record(first)
    sink.record(first)  # duplicate delivery is ignored
    sink.record(record(run_id=run_id, status="error", error_category="timeout", estimated_cost_usd=None))
    try:
        with psycopg.connect(DSN) as conn:
            rows = conn.execute("SELECT status, total_tokens, estimated_cost_usd FROM cost_log WHERE run_id=%s "
                                "ORDER BY status DESC", (run_id,)).fetchall()
        assert [(r[0], r[1]) for r in rows] == [("ok", 120), ("error", 120)]
        assert float(rows[0][2]) == pytest.approx(0.000027) and rows[1][2] is None
        assert sink.summary(1)["llm_calls"] >= 2
    finally:
        with psycopg.connect(DSN) as conn:
            conn.execute("DELETE FROM cost_log WHERE run_id=%s", (run_id,))
