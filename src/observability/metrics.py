"""Operational metric records and sinks.

LLMCallRecord holds counts, timings and labels only. Prompts, messages, model
outputs and CV text are never part of it, so it is safe to log and persist.
"""
from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Iterable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("src.observability")

CallStatus = Literal["ok", "error", "budget_exceeded"]


class LLMCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str | None = None
    request_id: str | None = None
    provider: str
    model: str
    operation: str                       # complete | generate:<Schema> | embed
    prompt_version: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = None   # None: no usage reported or model not in the pricing table
    latency_ms: float = Field(ge=0)
    status: CallStatus
    cache_hit: bool = False
    attempts: int = Field(default=1, ge=0)
    error_category: str | None = None


class UsageSink(Protocol):
    def record(self, record: LLMCallRecord) -> None: ...


class MemoryUsageSink:
    """Keeps records in memory; used by evaluations and tests."""

    def __init__(self) -> None:
        self.records: list[LLMCallRecord] = []

    def record(self, record: LLMCallRecord) -> None:
        self.records.append(record)

    def for_run(self, run_id: str) -> list[LLMCallRecord]:
        return [r for r in self.records if r.run_id == run_id]


class LoggingUsageSink:
    """One structured JSON log line per LLM call."""

    def record(self, record: LLMCallRecord) -> None:
        logger.info(json.dumps({"event": "llm_call", **record.model_dump(mode="json")}, sort_keys=True))


def record_safely(sink: UsageSink, record: LLMCallRecord) -> None:
    """Deliver a record; a failing sink is logged (exception type only) and never raises."""
    try:
        sink.record(record)
    except Exception as exc:  # noqa: BLE001 - observability must not break requests
        logger.warning(json.dumps({"event": "usage_sink_failed", "sink": type(sink).__name__,
                                   "exception": type(exc).__name__}))


class CompositeUsageSink:
    """Fans out to several sinks; a failing sink never affects the others or the caller."""

    def __init__(self, sinks: Iterable[UsageSink]) -> None:
        self.sinks = list(sinks)

    def record(self, record: LLMCallRecord) -> None:
        for sink in self.sinks:
            record_safely(sink, record)


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0..100); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(records: list[LLMCallRecord]) -> dict:
    """Aggregate usage over records (e.g. one evaluation run)."""
    costs = [r.estimated_cost_usd for r in records if r.estimated_cost_usd is not None]
    provider_latencies = [r.latency_ms for r in records if not r.cache_hit and r.status == "ok"]
    hits = sum(r.cache_hit for r in records)
    return {
        "llm_calls": len(records),
        "errors": sum(r.status != "ok" for r in records),
        "input_tokens": sum(r.input_tokens for r in records),
        "output_tokens": sum(r.output_tokens for r in records),
        "total_tokens": sum(r.total_tokens for r in records),
        "estimated_cost_usd": round(sum(costs), 8) if costs else None,
        "cache_hits": hits,
        "cache_hit_rate": round(hits / len(records), 6) if records else None,
        "provider_latency_ms_p50": _round(percentile(provider_latencies, 50)),
        "provider_latency_ms_p95": _round(percentile(provider_latencies, 95)),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)
