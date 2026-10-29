"""Construct the configured, instrumented LLM provider from Settings.

Every LLM call made through these providers is traced, measured, costed,
optionally cached and checked against the per-run budget (src.observability).
"""
from __future__ import annotations

from src.config import Settings
from src.observability.cache import ResponseCache
from src.observability.instrumented import InstrumentedEmbeddings, InstrumentedProvider, LLMRetryPolicy
from src.observability.metrics import UsageSink
from src.observability.pricing import PricingTable


def build_provider(settings: Settings, sink: UsageSink, cache: ResponseCache | None = None,
                   prompt_version: str | None = None) -> InstrumentedProvider:
    """OpenAI in live mode, the deterministic fixture provider in fixture mode."""
    if settings.app_mode == "live":
        from openai import OpenAI
        from src.llm.openai_provider import OpenAIProvider
        # SDK retries are disabled so that every retry is visible to (and counted by) the wrapper.
        client = OpenAI(api_key=settings.openai_api_key.get_secret_value(), max_retries=0,
                        timeout=settings.openai_timeout_seconds)
        inner = OpenAIProvider(client, settings.openai_model, max_output_tokens=settings.openai_max_output_tokens)
    else:
        from src.llm.fixture_provider import FixtureToolCallingProvider
        inner = FixtureToolCallingProvider()
        cache = None  # deterministic and free: caching would only hide the real code path
    return InstrumentedProvider(
        inner, sink=sink, pricing=PricingTable.from_json(settings.llm_pricing_json), cache=cache,
        cache_operations=settings.llm_cache_operations, cache_namespace=settings.llm_cache_version,
        prompt_version=prompt_version or settings.agent_prompt_version,
        retry=LLMRetryPolicy(max_retries=settings.llm_max_retries, backoff_seconds=settings.llm_retry_backoff_seconds))


def build_embeddings(settings: Settings, sink: UsageSink):
    if settings.embeddings == "hash":
        from src.rag.embeddings import HashEmbeddings
        return HashEmbeddings()
    if settings.openai_api_key is None:
        from src.config import ConfigurationError
        raise ConfigurationError("Invalid configuration: EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
    from openai import OpenAI
    from src.rag.embeddings import OpenAIEmbeddings
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value(), timeout=settings.openai_timeout_seconds,
                    max_retries=settings.llm_max_retries)
    return InstrumentedEmbeddings(OpenAIEmbeddings(client, settings.embedding_model, settings.embedding_dimensions),
                                  sink=sink, pricing=PricingTable.from_json(settings.llm_pricing_json))
