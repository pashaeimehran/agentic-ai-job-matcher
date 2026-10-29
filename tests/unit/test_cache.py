"""Response cache: key stability, hits, misses, invalidation, TTL and LRU (deterministic, no network)."""
import pytest

from src.extraction.schemas import JobPosting
from src.llm.provider import LLMUsage
from src.llm.tool_calling import AssistantTurn, ChatMessage, ToolCall, ToolSpec
from src.observability import InstrumentedProvider, MemoryResponseCache, MemoryUsageSink, cache_key

MESSAGES = [ChatMessage(role="system", content="rules"), ChatMessage(role="user", content="find AI jobs")]
SPEC = ToolSpec(name="search_jobs", description="Search", parameters={"type": "object"})


def key(**overrides):
    params = {"namespace": "v1", "provider": "openai", "model": "gpt-4o-mini", "operation": "complete",
              "messages": MESSAGES, "tools": [SPEC], "params": {"max_output_tokens": 500}, "prompt_version": "v1"}
    return cache_key(**{**params, **overrides})


def test_key_is_stable_and_hex_only():
    assert key() == key() and len(key()) == 64 and all(c in "0123456789abcdef" for c in key())
    # Equivalent dicts in a different order produce the same key.
    assert key(params={"b": 1, "a": 2}) == key(params={"a": 2, "b": 1})


def test_key_contains_no_prompt_text():
    assert "find AI jobs" not in key()


@pytest.mark.parametrize("change", [
    {"namespace": "v2"}, {"model": "gpt-4o"}, {"provider": "other"}, {"operation": "generate"},
    {"prompt_version": "v2"}, {"tools": []}, {"params": {"max_output_tokens": 600}},
    {"messages": [*MESSAGES, ChatMessage(role="user", content="more")]},
    {"output_schema": JobPosting},
])
def test_any_relevant_input_change_is_a_different_key(change):
    assert key(**change) != key()


def test_memory_cache_hit_miss_ttl_and_lru():
    now = [0.0]
    cache = MemoryResponseCache(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    assert cache.get("a") is None
    cache.set("a", "1"), cache.set("b", "2")
    assert cache.get("a") == "1"
    cache.set("c", "3")                      # evicts least recently used "b"
    assert cache.get("b") is None and cache.get("c") == "3"
    now[0] = 11
    assert cache.get("a") is None            # expired
    assert cache.stats()["hits"] == 2 and cache.stats()["misses"] == 3
    with pytest.raises(ValueError):
        MemoryResponseCache(max_entries=0)


class CountingProvider:
    model = "gpt-4o-mini"
    model_parameters = {"max_output_tokens": 500}

    def __init__(self):
        self.calls = 0

    def complete_with_usage(self, messages, tools):
        self.calls += 1
        return AssistantTurn(tool_calls=[ToolCall(call_id=f"c{self.calls}", name="search_jobs",
                                                  arguments='{"query": "AI"}')]), LLMUsage(input_tokens=900,
                                                                                           output_tokens=40)

    def generate_with_usage(self, messages, schema):
        self.calls += 1
        return schema(role="AI Engineer"), LLMUsage(input_tokens=100, output_tokens=20)


def test_repeated_request_is_served_from_cache_at_zero_cost():
    inner, sink, cache = CountingProvider(), MemoryUsageSink(), MemoryResponseCache()
    provider = InstrumentedProvider(inner, sink=sink, cache=cache, prompt_version="v1")
    first = provider.complete(MESSAGES, [SPEC])
    second = provider.complete(MESSAGES, [SPEC])
    assert first == second and inner.calls == 1
    miss, hit = sink.records
    assert not miss.cache_hit and miss.total_tokens == 940 and miss.estimated_cost_usd > 0
    assert hit.cache_hit and hit.total_tokens == 0 and hit.estimated_cost_usd == 0.0 and hit.attempts == 0


def test_structured_generation_is_cached_per_schema():
    inner, cache = CountingProvider(), MemoryResponseCache()
    provider = InstrumentedProvider(inner, sink=MemoryUsageSink(), cache=cache)
    messages = [{"role": "user", "content": "posting"}]
    assert provider.generate(messages, JobPosting) == provider.generate(messages, JobPosting)
    assert inner.calls == 1 and isinstance(provider.generate(messages, JobPosting), JobPosting)


def test_prompt_version_or_namespace_bump_invalidates():
    inner, cache = CountingProvider(), MemoryResponseCache()
    for version, namespace in (("v1", "a"), ("v2", "a"), ("v2", "b")):
        InstrumentedProvider(inner, sink=MemoryUsageSink(), cache=cache, prompt_version=version,
                             cache_namespace=namespace).complete(MESSAGES, [SPEC])
    assert inner.calls == 3


def test_operations_can_be_excluded_from_caching():
    inner = CountingProvider()
    provider = InstrumentedProvider(inner, sink=MemoryUsageSink(), cache=MemoryResponseCache(),
                                    cache_operations=frozenset({"generate"}))
    provider.complete(MESSAGES, [SPEC]), provider.complete(MESSAGES, [SPEC])
    assert inner.calls == 2


def test_failed_calls_are_not_cached():
    class Failing(CountingProvider):
        def complete_with_usage(self, messages, tools):
            self.calls += 1
            raise ValueError("bad")
    inner, cache = Failing(), MemoryResponseCache()
    provider = InstrumentedProvider(inner, sink=MemoryUsageSink(), cache=cache)
    for _ in range(2):
        with pytest.raises(ValueError):
            provider.complete(MESSAGES, [SPEC])
    assert inner.calls == 2 and len(cache) == 0
