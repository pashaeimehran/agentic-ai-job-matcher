"""Deterministic response cache for LLM calls.

Key: SHA-256 over a canonical JSON document of everything that determines the
response: cache namespace version, provider, model, model parameters,
operation, prompt version, the full message list (including tool observations),
tool schemas and the output schema. Any change to any of them is a miss.

Invalidation and versioning:
  - CACHE_SCHEMA_VERSION changes when the key layout or the stored format changes;
  - the configured namespace (LLM_CACHE_VERSION) is bumped to drop all entries;
  - prompt_version and the prompt text itself are part of the key;
  - entries expire after a TTL and the store is bounded (LRU eviction).

Storage is process-local memory only: entries can contain model output derived
from CV text, so they are never written to a shared location. Keys are hashes,
so no prompt or CV text appears in them.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Protocol, Sequence

from pydantic import BaseModel

CACHE_SCHEMA_VERSION = "1"


def _canonical(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {"__type__": type(value).__name__, **value.model_dump(mode="json")}
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def cache_key(*, namespace: str, provider: str, model: str, operation: str, messages: Sequence[Any],
              tools: Sequence[Any] = (), output_schema: type[BaseModel] | None = None,
              params: dict[str, Any] | None = None, prompt_version: str | None = None) -> str:
    document = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "provider": provider,
        "model": model,
        "params": _canonical(params or {}),
        "operation": operation,
        "prompt_version": prompt_version,
        "messages": _canonical(list(messages)),
        "tools": _canonical(list(tools)),
        "output_schema": None if output_schema is None else {
            "name": f"{output_schema.__module__}.{output_schema.__qualname__}",
            "json_schema": output_schema.model_json_schema()},
    }
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...


class MemoryResponseCache:
    """Thread-safe bounded LRU cache with per-entry TTL; values are serialized JSON strings."""

    def __init__(self, max_entries: int = 512, ttl_seconds: float = 3600,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if max_entries < 1 or ttl_seconds <= 0:
            raise ValueError("Cache needs max_entries >= 1 and ttl_seconds > 0")
        self.max_entries, self.ttl_seconds, self._clock = max_entries, ttl_seconds, clock
        self._entries: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> str | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0] <= self._clock():
                if entry is not None:
                    del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry[1]

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._entries[key] = (self._clock() + self.ttl_seconds, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        lookups = self.hits + self.misses
        return {"entries": len(self), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / lookups, 6) if lookups else None}
