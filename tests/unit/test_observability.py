"""Usage extraction, cost estimation, tracing, retries, budgets and log hygiene (no network, no Langfuse)."""
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.agent import AgentConfig, ManualAgent
from src.api import build_services, create_app
from src.extraction.schemas import JobPosting
from src.graph import GraphConfig, LangGraphAgent
from src.llm.openai_provider import OpenAIProvider, usage_from_response
from src.llm.provider import LLMProviderError, LLMUsage
from src.llm.tool_calling import AssistantTurn, ChatMessage
from src.matching.fixtures import ROOT
from src.observability import (BudgetExceededError, InstrumentedProvider, LLMRetryPolicy, MemoryUsageSink,
                               NoopTracer, PricingTable, RecordingTracer, RunBudget, run_scope, summarize,
                               use_tracer)
from src.observability.instrumented import InstrumentedEmbeddings, error_category
from src.observability.metrics import CompositeUsageSink, LLMCallRecord, LoggingUsageSink, percentile
from src.observability.pricing import ModelPrice
from src.observability.tracing import LangfuseTracer
from src.tools import default_registry
from src.tools.context import build_offline_context
from tests.unit.agent_fakes import ScriptedProvider, answer, call, tools

USAGE = LLMUsage(input_tokens=1000, output_tokens=200, cached_input_tokens=400)


class UsageProvider:
    """Tool-calling fake that reports token usage like OpenAIProvider.complete_with_usage."""
    model = "gpt-4o-mini"

    def __init__(self, steps, usage=USAGE):
        self.inner, self.usage = ScriptedProvider(steps), usage

    def complete_with_usage(self, messages, tool_specs):
        return self.inner.complete(messages, tool_specs), self.usage


class FlakyProvider:
    model = "gpt-4o-mini"

    def __init__(self, errors, then=None):
        self.errors, self.then, self.calls = list(errors), then or answer("ok"), 0

    def complete(self, messages, tool_specs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.then


class RateLimitError(Exception):
    """Named like the OpenAI SDK error; classification is by class name."""


class AuthenticationError(Exception):
    pass


MESSAGES = [ChatMessage(role="user", content="hi")]


# --- usage extraction and cost ------------------------------------------------------

def test_usage_extracted_from_responses_api_payload():
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=120, output_tokens=30,
                                                     input_tokens_details=SimpleNamespace(cached_tokens=20)))
    assert usage_from_response(response) == LLMUsage(input_tokens=120, output_tokens=30, cached_input_tokens=20)
    assert usage_from_response(SimpleNamespace()) is None


def test_openai_provider_returns_usage_and_applies_output_limit():
    class Responses:
        def parse(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(output_parsed=JobPosting(role="AI Engineer"),
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    responses = Responses()
    provider = OpenAIProvider(SimpleNamespace(responses=responses), "gpt-4o-mini", max_output_tokens=300)
    result, usage = provider.generate_with_usage([{"role": "user", "content": "x"}], JobPosting)
    assert result.role == "AI Engineer" and usage == LLMUsage(input_tokens=10, output_tokens=5)
    assert responses.kwargs["max_output_tokens"] == 300 and provider.model_parameters == {"max_output_tokens": 300}


def test_cost_estimate_is_deterministic_and_uses_cached_rate():
    pricing = PricingTable({"m": ModelPrice(input_per_1m=1.0, output_per_1m=4.0, cached_input_per_1m=0.5)})
    # (600 * 1.0 + 400 * 0.5 + 200 * 4.0) / 1e6
    assert pricing.estimate("m", USAGE) == pytest.approx(0.0016)
    assert pricing.estimate("m", USAGE) == pricing.estimate("m", USAGE)


def test_cost_unknown_model_or_usage_is_none_not_guessed():
    pricing = PricingTable()
    assert pricing.estimate("some-new-model", USAGE) is None
    assert pricing.estimate("gpt-4o-mini", None) is None


def test_dated_snapshot_uses_longest_prefix_and_json_override():
    pricing = PricingTable.from_json('{"gpt-4o-mini": {"input": 1, "output": 2}}')
    assert pricing.price_for("gpt-4o-mini-2024-07-18") == ModelPrice(1.0, 2.0, None)
    assert pricing.price_for("gpt-4o-2024-08-06").input_per_1m == 2.50
    with pytest.raises(ValueError):
        PricingTable.from_json("{not json")


def test_summary_and_percentiles():
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([], 50) is None
    records = [LLMCallRecord(provider="p", model="m", operation="complete", input_tokens=10, output_tokens=5,
                             total_tokens=15, estimated_cost_usd=0.001, latency_ms=100, status="ok"),
               LLMCallRecord(provider="p", model="m", operation="complete", latency_ms=1, status="ok",
                             cache_hit=True, estimated_cost_usd=0.0)]
    summary = summarize(records)
    assert summary["total_tokens"] == 15 and summary["cache_hit_rate"] == 0.5
    assert summary["estimated_cost_usd"] == 0.001 and summary["provider_latency_ms_p50"] == 100


# --- instrumented provider ----------------------------------------------------------

def test_instrumented_call_records_usage_cost_latency_and_run_ids():
    sink = MemoryUsageSink()
    provider = InstrumentedProvider(UsageProvider([answer("hi")]), sink=sink, prompt_version="v2")
    with run_scope("run-1", request_id="req-1"):
        assert provider.complete(MESSAGES, []).text == "hi"
    (record,) = sink.records
    assert (record.run_id, record.request_id, record.operation, record.status) == ("run-1", "req-1", "complete", "ok")
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (1000, 200, 1200)
    assert record.estimated_cost_usd == PricingTable().estimate("gpt-4o-mini", USAGE)
    assert record.prompt_version == "v2" and record.attempts == 1 and not record.cache_hit
    assert record.latency_ms >= 0


def test_transient_failures_are_retried_and_counted():
    sink, sleeps = MemoryUsageSink(), []
    inner = FlakyProvider([RateLimitError("slow down"), TimeoutError()])
    provider = InstrumentedProvider(inner, sink=sink, retry=LLMRetryPolicy(max_retries=2, backoff_seconds=0.5),
                                    sleep=sleeps.append)
    assert provider.complete(MESSAGES, []).text == "ok"
    assert inner.calls == 3 and sleeps == [0.5, 1.0]
    assert sink.records[0].attempts == 3 and sink.records[0].status == "ok"


def test_provider_failure_is_logged_and_recorded_without_payload(caplog):
    sink = MemoryUsageSink()
    provider = InstrumentedProvider(FlakyProvider([AuthenticationError("sk-live-SECRET bad key")]), sink=sink)
    with caplog.at_level(logging.INFO), pytest.raises(AuthenticationError):
        provider.complete(MESSAGES, [])
    (record,) = sink.records
    assert record.status == "error" and record.error_category == "auth" and record.attempts == 1
    assert "llm_call_failed" in caplog.text and "SECRET" not in caplog.text


def test_error_categories():
    assert error_category(TimeoutError()) == "timeout"
    assert error_category(LLMProviderError("x")) == "invalid_response"
    assert error_category(BudgetExceededError("x")) == "budget_exceeded"
    assert error_category(KeyError("x")) == "other"


def test_failing_sink_never_breaks_the_call():
    class Broken:
        def record(self, record):
            raise ConnectionError("db down")
    provider = InstrumentedProvider(UsageProvider([answer("hi")]), sink=CompositeUsageSink([Broken()]))
    assert provider.complete(MESSAGES, []).text == "hi"


def test_budget_stops_the_run_before_the_next_model_call():
    budget = RunBudget(max_tokens=1000)
    budget.add(1200, 0.01)
    with pytest.raises(BudgetExceededError):
        budget.check()


@pytest.mark.parametrize("engine", ["manual", "langgraph"])
def test_agents_end_with_budget_exceeded_status(engine):
    sink = MemoryUsageSink()
    steps = [tools(call("search_jobs", {"query": "AI"})), tools(call("search_jobs", {"query": "ML"})), answer("x")]
    provider = InstrumentedProvider(UsageProvider(steps), sink=sink)  # 1200 tokens per call
    limits = {"max_run_tokens": 2000}
    agent = (ManualAgent(provider, default_registry(), build_offline_context(), AgentConfig(**limits))
             if engine == "manual" else
             LangGraphAgent(provider, default_registry(), build_offline_context(), GraphConfig(**limits)))
    result = agent.run("find jobs", run_id="budget-run")
    assert result.status == "budget_exceeded" and result.model_turns == 2
    assert [r.status for r in sink.records] == ["ok", "ok", "budget_exceeded"]


def test_embeddings_usage_is_recorded_without_texts():
    class FakeEmbeddings:
        model, dimensions = "text-embedding-3-small", 3

        def embed_with_usage(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts], LLMUsage(input_tokens=50)
    sink = MemoryUsageSink()
    vectors = InstrumentedEmbeddings(FakeEmbeddings(), sink=sink).embed(["German B1 secret CV line"])
    assert vectors == [[1.0, 0.0, 0.0]]
    (record,) = sink.records
    assert record.operation == "embed" and record.input_tokens == 50 and record.estimated_cost_usd == 0.000001
    assert "German" not in record.model_dump_json()


# --- tracing -------------------------------------------------------------------------

def test_trace_hierarchy_for_graph_run_tools_and_model_calls():
    tracer = RecordingTracer()
    provider = InstrumentedProvider(UsageProvider([tools(call("calculate_match", {"job_ids": ["fixture-01"]})),
                                                   answer("done")]), sink=MemoryUsageSink())
    with use_tracer(tracer):
        LangGraphAgent(provider, default_registry(), build_offline_context(), GraphConfig()).run("x", run_id="r9")
    parents = {(s.name, s.parent) for s in tracer.spans}
    assert ("agent.langgraph", None) in parents
    assert ("llm.complete", "graph.agent") in parents
    assert ("tool.calculate_match", "graph.execute_tools") in parents
    assert ("graph.finalize", "agent.langgraph") in parents
    llm = next(s for s in tracer.spans if s.name == "llm.complete")
    assert llm.kind == "generation" and llm.model == "gpt-4o-mini" and llm.metadata["run_id"] == "r9"
    assert llm.updates["usage_details"] == {"input": 1000, "output": 200, "total": 1200}
    tool = next(s for s in tracer.spans if s.name == "tool.calculate_match")
    assert tool.metadata["status"] == "ok" and tool.input is None  # content capture is off by default


def test_manual_agent_and_api_request_spans():
    tracer = RecordingTracer()
    services = build_services(provider=ScriptedProvider([answer("hello")]))
    with use_tracer(tracer):
        client = TestClient(create_app(services))
        assert client.post("/chat", json={"message": "hi"}).status_code == 200
        ManualAgent(ScriptedProvider([answer("x")]), default_registry(), build_offline_context()).run("q")
    names = tracer.names()
    assert "POST /chat" in names and "agent.manual" in names
    assert next(s for s in tracer.spans if s.name == "agent.langgraph").parent == "POST /chat"
    assert next(s for s in tracer.spans if s.name == "POST /chat").metadata["status_code"] == 200


def test_content_capture_is_opt_in():
    tracer = RecordingTracer(capture_content=True)
    with use_tracer(tracer):
        default_registry().execute("search_jobs", {"query": "AI"}, build_offline_context())
    assert tracer.spans[0].input == {"query": "AI"} and tracer.spans[0].updates["output"]["status"] == "ok"


def test_tracing_disabled_is_a_noop():
    with use_tracer(NoopTracer()):
        result = LangGraphAgent(ScriptedProvider([answer("ok")]), default_registry(), build_offline_context(),
                                GraphConfig()).run("x")
    assert result.status == "completed"


def test_langfuse_tracer_degrades_when_backend_fails():
    class BrokenClient:
        def start_as_current_observation(self, **kwargs):
            raise ConnectionError("langfuse unreachable")

        def flush(self):
            raise ConnectionError("down")

        def shutdown(self):
            raise ConnectionError("down")
    tracer = LangfuseTracer(BrokenClient())
    with use_tracer(tracer):
        result = LangGraphAgent(ScriptedProvider([answer("ok")]), default_registry(), build_offline_context(),
                                GraphConfig()).run("x")
    tracer.flush(), tracer.shutdown()
    assert result.status == "completed"


def test_langfuse_flush_and_shutdown_are_time_bounded(caplog):
    import threading
    import time
    release = threading.Event()

    class HangingClient:  # exporter retrying against an unreachable backend
        def flush(self):
            release.wait(10)

        shutdown = flush
    tracer = LangfuseTracer(HangingClient(), export_timeout_seconds=0.05)
    started = time.perf_counter()
    with caplog.at_level("WARNING", logger="src.observability"):
        tracer.flush(), tracer.shutdown()
    release.set()
    assert time.perf_counter() - started < 2
    assert caplog.text.count('"exception": "Timeout"') == 2


def test_langfuse_tracer_passes_metadata_but_not_content_by_default():
    calls = []

    class Observation:
        def update(self, **fields):
            calls.append(("update", fields))

    class Client:
        def start_as_current_observation(self, **kwargs):
            calls.append(("start", kwargs))
            from contextlib import nullcontext
            return nullcontext(Observation())
    with LangfuseTracer(Client()).span("tool.search_cv", kind="tool", metadata={"run_id": "r"},
                                       input={"query": "secret"}) as span:
        span.update(output={"text": "cv"}, metadata={"status": "ok"})
    start, update = calls
    assert start[1]["as_type"] == "tool" and "input" not in start[1]
    assert "output" not in update[1] and update[1]["metadata"] == {"status": "ok"}


# --- log hygiene ---------------------------------------------------------------------

def test_no_raw_cv_text_or_secrets_in_operational_logs(caplog):
    cv_text = (ROOT / "sample_cv.md").read_text(encoding="utf-8")
    cv_lines = [line.strip() for line in cv_text.splitlines() if len(line.strip()) > 25]
    secret = "sk-test-DO-NOT-LOG-123"
    sink = MemoryUsageSink()
    provider = InstrumentedProvider(UsageProvider([tools(call("search_cv", {"query": "German level"})),
                                                   tools(call("calculate_match", {"job_ids": ["match-03-robotics"]})),
                                                   answer("Your German is B1.")]),
                                    sink=CompositeUsageSink([sink, LoggingUsageSink()]))
    with caplog.at_level(logging.DEBUG):
        result = LangGraphAgent(provider, default_registry(), build_offline_context(), GraphConfig()).run(
            f"My key is {secret}. What is my German level?")
    assert result.status == "completed"
    logged = caplog.text + "".join(r.model_dump_json() for r in sink.records)
    assert "llm_call" in caplog.text and "tool_succeeded" in caplog.text
    assert secret not in logged
    assert not [line for line in cv_lines if line in logged]
    assert all(json.loads(r.model_dump_json()).keys().isdisjoint({"messages", "prompt", "output"})
               for r in sink.records)
