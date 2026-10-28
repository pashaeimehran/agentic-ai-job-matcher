"""Application services for the API and FastAPI dependency providers.

Services are created once per app (create_app) and stored on app.state, so
tests build an app with their own services instead of mutating globals.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Request

from src.api.errors import ApiError, from_tool_error
from src.graph import GraphConfig, LangGraphAgent, memory_checkpointer
from src.llm.tool_calling import ToolCallingLLMProvider
from src.tools import default_registry
from src.tools.context import ToolContext, build_offline_context
from src.tools.jobs import JobCatalog
from src.tools.registry import RetryPolicy, ToolRegistry


@dataclass
class ApiServices:
    context: ToolContext
    registry: ToolRegistry
    agent: LangGraphAgent | None  # None: no LLM provider configured, /chat is unavailable
    retry: RetryPolicy = RetryPolicy()

    def call_tool(self, name: str, arguments: dict[str, Any], request_id: str) -> tuple[str, dict]:
        """Run one shared tool; return (status, data) or raise the mapped ApiError."""
        outcome = self.registry.execute(name, arguments, self.context, self.retry, run_id=request_id)
        if outcome.error is not None:
            raise from_tool_error(outcome.error)
        return outcome.status, outcome.data


def build_services(context: ToolContext | None = None, provider: ToolCallingLLMProvider | None = None,
                   registry: ToolRegistry | None = None, config: GraphConfig = GraphConfig()) -> ApiServices:
    context = context or build_offline_context()
    registry = registry or default_registry()
    agent = (LangGraphAgent(provider, registry, context, config, checkpointer=memory_checkpointer())
             if provider is not None else None)
    return ApiServices(context=context, registry=registry, agent=agent, retry=config.tool_retry)


def build_default_services() -> ApiServices:
    """Synthetic CV and fixture catalog; saved jobs in data/private; OpenAI only if a key is set."""
    from dotenv import load_dotenv
    from src.tools.saved_jobs import JsonFileSavedJobStore
    load_dotenv()
    provider = None
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        from src.llm.openai_provider import OpenAIProvider
        provider = OpenAIProvider(OpenAI(), os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    return build_services(build_offline_context(saved_jobs=JsonFileSavedJobStore()), provider)


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
