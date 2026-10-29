"""PostgreSQL persistence for LLMCallRecord (the cost_log table, created by src.db)."""
from __future__ import annotations

from typing import Any, Callable

from src.observability.metrics import LLMCallRecord

_COLUMNS = ("record_id", "created_at", "run_id", "request_id", "provider", "model", "operation",
            "prompt_version", "input_tokens", "output_tokens", "cached_input_tokens", "total_tokens",
            "estimated_cost_usd", "latency_ms", "status", "cache_hit", "attempts", "error_category")

INSERT_SQL = (f"INSERT INTO cost_log ({', '.join(_COLUMNS)}) VALUES ({', '.join(['%s'] * len(_COLUMNS))}) "
              "ON CONFLICT (record_id) DO NOTHING")


class PostgresUsageSink:
    """One short transaction per record. Wrap in CompositeUsageSink so a database outage never fails a call."""

    def __init__(self, dsn: str, connect: Callable[[str], Any] | None = None) -> None:
        self.dsn = dsn
        self._connect = connect

    def _connection(self):
        if self._connect is not None:
            return self._connect(self.dsn)
        import psycopg
        return psycopg.connect(self.dsn, connect_timeout=5)

    def record(self, record: LLMCallRecord) -> None:
        row = record.model_dump()
        with self._connection() as conn:
            conn.execute(INSERT_SQL, tuple(row[column] for column in _COLUMNS))

    def summary(self, since_hours: int = 24) -> dict[str, Any]:
        """Aggregate usage over a recent window (used by scripts and smoke checks)."""
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(self.dsn, connect_timeout=5, row_factory=dict_row) as conn:
            return conn.execute(
                """SELECT count(*) AS llm_calls, coalesce(sum(total_tokens), 0) AS total_tokens,
                          coalesce(sum(estimated_cost_usd), 0)::float AS estimated_cost_usd,
                          coalesce(avg(cache_hit::int), 0)::float AS cache_hit_rate
                   FROM cost_log WHERE created_at >= now() - make_interval(hours => %s)""",
                (since_hours,)).fetchone()
