"""Application services for the API and FastAPI dependency providers.

Services are created once per app (create_app) and stored on app.state, so
tests build an app with their own services instead of mutating globals.
build_default_services() turns Settings (environment variables) into the
production object graph; build_services() is the explicit constructor tests use.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, Request

from src.agent.prompts import SYSTEM_PROMPT, system_prompt
from src.api.errors import ApiError, from_tool_error
from src.config import Settings
from src.graph import GraphConfig, LangGraphAgent, memory_checkpointer
from src.llm.tool_calling import ToolCallingLLMProvider
from src.observability.tracing import NoopTracer, Tracer
from src.tools import default_registry
from src.tools.context import ToolContext, build_offline_context
from src.tools.jobs import JobCatalog
from src.tools.registry import RetryPolicy, ToolRegistry

logger = logging.getLogger("src.api")


@dataclass
class ApiServices:
    context: ToolContext
    registry: ToolRegistry
    agent: LangGraphAgent | None  # None: no LLM provider configured, /chat is unavailable
    retry: RetryPolicy = RetryPolicy()
    mode: str = "custom"           # fixture | live | custom (tests)
    database_url: str | None = field(default=None, repr=False)
    tracer: Tracer = field(default_factory=NoopTracer)
    summary: dict[str, Any] = field(default_factory=dict)

    def call_tool(self, name: str, arguments: dict[str, Any], request_id: str) -> tuple[str, dict]:
        """Run one shared tool; return (status, data) or raise the mapped ApiError."""
        outcome = self.registry.execute(name, arguments, self.context, self.retry, run_id=request_id)
        if outcome.error is not None:
            raise from_tool_error(outcome.error)
        return outcome.status, outcome.data

    def close(self) -> None:
        """Flush and stop tracing; database connections are per operation and need no pool shutdown."""
        self.tracer.flush()
        self.tracer.shutdown()


def build_services(context: ToolContext | None = None, provider: ToolCallingLLMProvider | None = None,
                   registry: ToolRegistry | None = None, config: GraphConfig = GraphConfig(),
                   prompt: str = SYSTEM_PROMPT) -> ApiServices:
    context = context or build_offline_context()
    registry = registry or default_registry()
    agent = (LangGraphAgent(provider, registry, context, config, system_prompt=prompt,
                            checkpointer=memory_checkpointer())
             if provider is not None else None)
    return ApiServices(context=context, registry=registry, agent=agent, retry=config.tool_retry)


def build_default_services(settings: Settings | None = None) -> ApiServices:
    """Synthetic CV and fixture catalog with the configured provider, storage and observability.

    fixture mode: deterministic scripted provider, hash embeddings, no paid calls.
    live mode:    OpenAI provider and embeddings (paid), same tools and data.
    DATABASE_URL: schema is initialized, CV chunks / saved jobs / cost log live in PostgreSQL.
    Raises ConfigurationError or SchemaInitializationError so startup fails with a clear message.
    """
    from src.db import initialize_database
    from src.llm.factory import build_embeddings, build_provider
    from src.observability.setup import build_cache, build_usage_sink, configure_tracing
    from src.rag.service import CVService
    from src.tools.context import build_synthetic_context

    if settings is None:
        from dotenv import load_dotenv
        load_dotenv()
        settings = Settings.from_env()
    tracer = configure_tracing(settings)
    dsn = settings.database_url.get_secret_value() if settings.database_url else None
    if dsn and settings.db_init_on_startup:
        initialize_database(dsn)
    sink = build_usage_sink(settings)
    if dsn:
        from src.rag.postgres import PostgresVectorStore
        from src.tools.saved_jobs import PostgresSavedJobStore
        store, saved_jobs = PostgresVectorStore(dsn), PostgresSavedJobStore(dsn)
    else:
        from src.rag.store import MemoryVectorStore
        from src.tools.saved_jobs import JsonFileSavedJobStore
        store = MemoryVectorStore()
        saved_jobs = (JsonFileSavedJobStore(settings.saved_jobs_path) if settings.saved_jobs_path
                      else JsonFileSavedJobStore())
    context = build_synthetic_context(CVService(build_embeddings(settings, sink), store), saved_jobs, settings.cv_id)
    provider = build_provider(settings, sink, build_cache(settings))
    config = GraphConfig(max_model_turns=settings.agent_max_model_turns,
                         max_tool_calls=settings.agent_max_tool_calls,
                         max_run_tokens=settings.agent_max_run_tokens,
                         max_run_cost_usd=settings.agent_max_run_cost_usd)
    services = build_services(context, provider, config=config,
                              prompt=system_prompt(settings.agent_prompt_version))
    services.mode, services.database_url, services.tracer = settings.app_mode, dsn, tracer
    services.summary = settings.public_summary()
    logger.info(json.dumps({"event": "services_ready", **services.summary}, sort_keys=True))
    return services


def get_services(request: Request) -> ApiServices:
    return request.app.state.services


def get_request_id(request: Request) -> str:
    return request.state.request_id


def get_catalog(services: ApiServices = Depends(get_services)) -> JobCatalog:
    return services.context.catalog


def get_agent(services: ApiServices = Depends(get_services)) -> LangGraphAgent:
    if services.agent is None:
        raise ApiError(503, "chat_unavailable", "No LLM provider is configured for the chat endpoint.")
    return services.agent
