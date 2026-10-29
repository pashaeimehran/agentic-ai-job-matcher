"""Build observability components from Settings (used by the API, CLI scripts and evaluations)."""
from __future__ import annotations

import json
import logging

from src.config import Settings
from src.observability.cache import MemoryResponseCache
from src.observability.metrics import CompositeUsageSink, LoggingUsageSink, UsageSink
from src.observability.tracing import LangfuseTracer, NoopTracer, Tracer, set_tracer

logger = logging.getLogger("src.observability")


def configure_tracing(settings: Settings) -> Tracer:
    """Install the process tracer. Falls back to a no-op tracer if Langfuse cannot be initialized."""
    tracer: Tracer = NoopTracer()
    if settings.tracing_active:
        try:
            tracer = LangfuseTracer.from_settings(
                public_key=settings.langfuse_public_key or "",
                secret_key=settings.langfuse_secret_key.get_secret_value() if settings.langfuse_secret_key else "",
                host=settings.langfuse_host, environment=settings.app_env, release="job-match-agent",
                capture_content=settings.langfuse_capture_content)
        except Exception as exc:  # noqa: BLE001 - tracing must never block startup
            logger.warning(json.dumps({"event": "tracing_disabled", "exception": type(exc).__name__}))
    set_tracer(tracer)
    return tracer


def build_usage_sink(settings: Settings, extra: list[UsageSink] | None = None) -> CompositeUsageSink:
    sinks: list[UsageSink] = [LoggingUsageSink(), *(extra or [])]
    if settings.cost_log_enabled and settings.database_url is not None:
        from src.observability.postgres import PostgresUsageSink
        sinks.append(PostgresUsageSink(settings.database_url.get_secret_value()))
    return CompositeUsageSink(sinks)


def build_cache(settings: Settings) -> MemoryResponseCache | None:
    if not settings.llm_cache_enabled:
        return None
    return MemoryResponseCache(max_entries=settings.llm_cache_max_entries, ttl_seconds=settings.llm_cache_ttl_seconds)
