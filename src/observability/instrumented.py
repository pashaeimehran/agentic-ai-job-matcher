"""Provider wrapper that adds observability, caching, retries and budget checks to every LLM call.

InstrumentedProvider implements both provider protocols (generate() and
complete()), so agents, graph nodes and matching code use it unchanged. Per call:

    budget check -> cache lookup -> provider call with bounded retries
    -> usage + deterministic cost -> LLMCallRecord to the sink -> cache store

and one "generation" span with model, usage and cost. Business logic never
touches tracing, metrics or the cache directly.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from src.llm.provider import LLMMessage, LLMProviderError, LLMUsage
from src.llm.tool_calling import AssistantTurn, ConversationItem, ToolSpec
from src.observability.cache import ResponseCache, cache_key
from src.observability.context import BudgetExceededError, current_run
from src.observability.metrics import CallStatus, LLMCallRecord, LoggingUsageSink, UsageSink, record_safely
from src.observability.pricing import PricingTable
from src.observability.tracing import get_tracer

logger = logging.getLogger("src.observability")

T = TypeVar("T", bound=BaseModel)

# Error classes are matched by name so this module never imports a provider SDK.
_ERROR_CATEGORIES = {
    "APITimeoutError": "timeout", "TimeoutError": "timeout", "ReadTimeout": "timeout",
    "RateLimitError": "rate_limit",
    "APIConnectionError": "connection", "ConnectionError": "connection", "ConnectError": "connection",
    "InternalServerError": "server_error",
    "AuthenticationError": "auth", "PermissionDeniedError": "auth",
    "BadRequestError": "bad_request", "UnprocessableEntityError": "bad_request", "NotFoundError": "bad_request",
    "LLMProviderError": "invalid_response", "ValidationError": "invalid_response",
    "BudgetExceededError": "budget_exceeded",
}
RETRYABLE_CATEGORIES = frozenset({"timeout", "rate_limit", "connection", "server_error"})


def error_category(exc: BaseException) -> str:
    for cls in type(exc).__mro__:
        if cls.__name__ in _ERROR_CATEGORIES:
            return _ERROR_CATEGORIES[cls.__name__]
    return "other"


class LLMRetryPolicy(BaseModel):
    """Retries for transient provider failures (timeouts, rate limits, connection and 5xx errors)."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_retries: int = Field(default=2, ge=0, le=5)
    backoff_seconds: float = Field(default=1.0, ge=0, le=30)


class InstrumentedProvider:
    def __init__(self, inner: Any, *, sink: UsageSink | None = None, pricing: PricingTable | None = None,
                 cache: ResponseCache | None = None,
                 cache_operations: frozenset[str] = frozenset({"generate", "complete"}),
                 cache_namespace: str = "v1", prompt_version: str | None = None,
                 retry: LLMRetryPolicy = LLMRetryPolicy(), provider_name: str | None = None,
                 clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.inner = inner
        self.sink = sink or LoggingUsageSink()
        self.pricing = pricing or PricingTable()
        self.cache, self.cache_operations, self.cache_namespace = cache, cache_operations, cache_namespace
        self.prompt_version, self.retry = prompt_version, retry
        self.provider_name = provider_name or getattr(inner, "provider_name", type(inner).__name__)
        self._clock, self._sleep = clock, sleep

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "unknown")

    # --- provider protocols ------------------------------------------------------

    def complete(self, messages: Sequence[ConversationItem], tools: Sequence[ToolSpec]) -> AssistantTurn:
        def call() -> tuple[AssistantTurn, LLMUsage | None]:
            if hasattr(self.inner, "complete_with_usage"):
                return self.inner.complete_with_usage(messages, tools)
            return self.inner.complete(messages, tools), None

        key = self._key("complete", messages, tools=tools)
        return self._run("complete", "complete", key, call,
                         dump=lambda turn: turn.model_dump_json(),
                         load=AssistantTurn.model_validate_json,
                         span_metadata={"tools": len(tools), "messages": len(messages)})

    def generate(self, messages: Sequence[LLMMessage], schema: type[T]) -> T:
        def call() -> tuple[T, LLMUsage | None]:
            if hasattr(self.inner, "generate_with_usage"):
                return self.inner.generate_with_usage(messages, schema)
            return self.inner.generate(messages, schema), None

        operation = f"generate:{schema.__name__}"
        key = self._key("generate", messages, output_schema=schema)
        return self._run("generate", operation, key, call,
                         dump=lambda result: schema.model_validate(result).model_dump_json(),
                         load=schema.model_validate_json,
                         span_metadata={"schema": schema.__name__, "messages": len(messages)})

    # --- internals ---------------------------------------------------------------

    def _key(self, kind: str, messages: Sequence[Any], tools: Sequence[Any] = (),
             output_schema: type[BaseModel] | None = None) -> str | None:
        if self.cache is None or kind not in self.cache_operations:
            return None
        return cache_key(namespace=self.cache_namespace, provider=self.provider_name, model=self.model,
                         operation=kind, messages=messages, tools=tools, output_schema=output_schema,
                         params=getattr(self.inner, "model_parameters", {}), prompt_version=self.prompt_version)

    def _run(self, kind: str, operation: str, key: str | None, call: Callable[[], tuple[Any, LLMUsage | None]],
             dump: Callable[[Any], str], load: Callable[[str], Any], span_metadata: dict[str, Any]) -> Any:
        run = current_run()
        started = self._clock()
        metadata = {"operation": operation, "run_id": run.run_id, "prompt_version": self.prompt_version,
                    **span_metadata}
        with get_tracer().span(f"llm.{kind}", kind="generation", model=self.model, metadata=metadata) as span:
            try:
                run.budget.check()
            except BudgetExceededError:
                self._record(operation, started, "budget_exceeded", attempts=0, category="budget_exceeded")
                span.update(level="WARNING", status_message="budget_exceeded")
                raise

            if key is not None:
                cached = self.cache.get(key)
                if cached is not None:
                    self._record(operation, started, "ok", attempts=0, cache_hit=True, usage=LLMUsage())
                    span.update(metadata={"cache_hit": True})
                    return load(cached)

            attempts = 0
            while True:
                attempts += 1
                try:
                    result, usage = call()
                    break
                except Exception as exc:
                    category = error_category(exc)
                    will_retry = category in RETRYABLE_CATEGORIES and attempts <= self.retry.max_retries
                    logger.warning(json.dumps({"event": "llm_call_failed", "run_id": run.run_id,
                                               "operation": operation, "attempt": attempts,
                                               "error_category": category, "exception": type(exc).__name__,
                                               "will_retry": will_retry}, sort_keys=True))
                    if will_retry:
                        if self.retry.backoff_seconds:
                            self._sleep(self.retry.backoff_seconds * 2 ** (attempts - 1))
                        continue
                    self._record(operation, started, "error", attempts=attempts, category=category)
                    span.update(level="ERROR", status_message=category,
                                metadata={"attempts": attempts, "cache_hit": False})
                    raise

            cost = self.pricing.estimate(self.model, usage)
            run.budget.add(usage.total_tokens if usage else 0, cost)
            self._record(operation, started, "ok", attempts=attempts, usage=usage, cost=cost)
            if key is not None:
                try:
                    self.cache.set(key, dump(result))
                except Exception as exc:  # noqa: BLE001 - a cache failure must not fail the call
                    logger.warning(json.dumps({"event": "llm_cache_store_failed",
                                               "exception": type(exc).__name__}))
            update: dict[str, Any] = {"metadata": {"attempts": attempts, "cache_hit": False}}
            if usage is not None:
                update["usage_details"] = {"input": usage.input_tokens, "output": usage.output_tokens,
                                           "total": usage.total_tokens}
            if cost is not None:
                update["cost_details"] = {"total": cost}
            if get_tracer().capture_content:
                update["output"] = dump(result)
            span.update(**update)
            return result

    def _record(self, operation: str, started: float, status: CallStatus, *, attempts: int,
                usage: LLMUsage | None = None, cost: float | None = None, cache_hit: bool = False,
                category: str | None = None) -> None:
        run = current_run()
        record = LLMCallRecord(
            run_id=run.run_id, request_id=run.request_id, provider=self.provider_name, model=self.model,
            operation=operation, prompt_version=self.prompt_version,
            input_tokens=usage.input_tokens if usage else 0, output_tokens=usage.output_tokens if usage else 0,
            cached_input_tokens=usage.cached_input_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            estimated_cost_usd=0.0 if cache_hit else cost,
            latency_ms=round((self._clock() - started) * 1000, 3), status=status, cache_hit=cache_hit,
            attempts=attempts, error_category=category)
        record_safely(self.sink, record)


class InstrumentedEmbeddings:
    """EmbeddingProvider wrapper: records usage, cost and latency of embedding calls (never the texts)."""

    def __init__(self, inner: Any, *, sink: UsageSink | None = None, pricing: PricingTable | None = None,
                 provider_name: str = "openai", clock: Callable[[], float] = time.perf_counter) -> None:
        self.inner, self.sink = inner, sink or LoggingUsageSink()
        self.pricing, self.provider_name, self._clock = pricing or PricingTable(), provider_name, clock

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        run, started = current_run(), self._clock()
        status: CallStatus = "ok"
        usage: LLMUsage | None = None
        category = None
        with get_tracer().span("llm.embed", kind="embedding", model=self.model,
                               metadata={"texts": len(texts), "run_id": run.run_id}) as span:
            try:
                if hasattr(self.inner, "embed_with_usage"):
                    vectors, usage = self.inner.embed_with_usage(texts)
                else:
                    vectors = self.inner.embed(texts)
                return vectors
            except Exception as exc:
                status, category = "error", error_category(exc)
                span.update(level="ERROR", status_message=category)
                raise
            finally:
                cost = self.pricing.estimate(self.model, usage)
                if usage is not None:
                    span.update(usage_details={"input": usage.input_tokens}, **(
                        {"cost_details": {"total": cost}} if cost is not None else {}))
                record_safely(self.sink, LLMCallRecord(
                    run_id=run.run_id, request_id=run.request_id, provider=self.provider_name,
                    model=self.model, operation="embed", input_tokens=usage.input_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0, estimated_cost_usd=cost,
                    latency_ms=round((self._clock() - started) * 1000, 3), status=status,
                    error_category=category))


__all__ = ["InstrumentedEmbeddings", "InstrumentedProvider", "LLMProviderError", "LLMRetryPolicy",
           "RETRYABLE_CATEGORIES", "error_category"]
