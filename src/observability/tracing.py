"""Tracing abstraction with a no-op default, a recording tracer for tests and a Langfuse backend.

Instrumentation calls get_tracer().span(...) at a few central points (API
request, agent run, graph node, tool execution, LLM call). Tracing is
best-effort: if the backend is missing, misconfigured or unreachable, spans
degrade to no-ops and the application keeps working.

Content policy: by default spans carry metadata only (names, IDs, statuses,
counts, model, usage). Inputs and outputs, which can contain CV text, are
attached only when capture_content is enabled for a trusted, self-hosted
Langfuse.
"""
from __future__ import annotations

import contextvars
import json
import logging
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Protocol

logger = logging.getLogger("src.observability")

SpanKind = Literal["span", "generation", "agent", "tool", "chain", "embedding"]


class Span(Protocol):
    def update(self, **fields: Any) -> None:
        """Accepted fields: output, metadata, level, status_message, model, usage_details, cost_details."""
        ...


class Tracer(Protocol):
    enabled: bool
    capture_content: bool

    def span(self, name: str, *, kind: SpanKind = "span", metadata: dict[str, Any] | None = None,
             input: Any = None, model: str | None = None) -> Any: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...


class _NoopSpan:
    def update(self, **fields: Any) -> None:
        return None


class NoopTracer:
    enabled = False
    capture_content = False

    @contextmanager
    def span(self, name: str, *, kind: SpanKind = "span", metadata: dict[str, Any] | None = None,
             input: Any = None, model: str | None = None) -> Iterator[Span]:
        yield _NoopSpan()

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


@dataclass
class SpanRecord:
    name: str
    kind: str
    parent: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    input: Any = None
    model: str | None = None
    updates: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def update(self, **fields: Any) -> None:
        metadata = fields.pop("metadata", None)
        if metadata:
            self.metadata.update(metadata)
        self.updates.update(fields)


class RecordingTracer:
    """In-memory tracer that records the span tree; used by tests and offline evaluations."""
    enabled = True

    def __init__(self, capture_content: bool = False) -> None:
        self.capture_content = capture_content
        self.spans: list[SpanRecord] = []
        self._stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
            f"recording_tracer_{id(self)}", default=())

    @contextmanager
    def span(self, name: str, *, kind: SpanKind = "span", metadata: dict[str, Any] | None = None,
             input: Any = None, model: str | None = None) -> Iterator[SpanRecord]:
        stack = self._stack.get()
        record = SpanRecord(name=name, kind=kind, parent=stack[-1] if stack else None,
                            metadata=dict(metadata or {}), input=input if self.capture_content else None,
                            model=model)
        self.spans.append(record)
        token = self._stack.set((*stack, name))
        try:
            yield record
        except BaseException as exc:
            record.error = type(exc).__name__
            raise
        finally:
            self._stack.reset(token)

    def names(self) -> list[str]:
        return [s.name for s in self.spans]

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class _LangfuseSpan:
    def __init__(self, observation: Any, capture_content: bool) -> None:
        self._observation, self._capture = observation, capture_content

    def update(self, **fields: Any) -> None:
        if not self._capture:
            fields.pop("output", None)
        try:
            self._observation.update(**fields)
        except Exception as exc:  # noqa: BLE001 - tracing is best-effort
            _tracing_failed("update", exc)


class LangfuseTracer:
    """Langfuse (OpenTelemetry-based SDK) backend. All SDK failures degrade to no-ops."""
    enabled = True

    def __init__(self, client: Any, capture_content: bool = False, export_timeout_seconds: float = 3.0) -> None:
        self._client, self.capture_content = client, capture_content
        self.export_timeout_seconds = export_timeout_seconds

    @classmethod
    def from_settings(cls, public_key: str, secret_key: str, host: str, environment: str,
                      release: str, capture_content: bool) -> "LangfuseTracer":
        from langfuse import Langfuse
        client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=host,
                          environment=environment, release=release, timeout=5)
        return cls(client, capture_content)

    @contextmanager
    def span(self, name: str, *, kind: SpanKind = "span", metadata: dict[str, Any] | None = None,
             input: Any = None, model: str | None = None) -> Iterator[Span]:
        with ExitStack() as stack:
            try:
                kwargs: dict[str, Any] = {"name": name, "as_type": kind, "metadata": metadata or None}
                if model:
                    kwargs["model"] = model
                if self.capture_content and input is not None:
                    kwargs["input"] = input
                observation = stack.enter_context(self._client.start_as_current_observation(**kwargs))
                span: Span = _LangfuseSpan(observation, self.capture_content)
            except Exception as exc:  # noqa: BLE001 - tracing is best-effort
                _tracing_failed("start", exc)
                span = _NoopSpan()
            try:
                yield span
            except BaseException as exc:
                span.update(level="ERROR", status_message=type(exc).__name__)
                raise

    def flush(self) -> None:
        self._bounded("flush", self._client.flush)

    def shutdown(self) -> None:
        self._bounded("shutdown", self._client.shutdown)

    def _bounded(self, stage: str, call: Callable[[], Any]) -> None:
        """Run an export call with a deadline. With the backend unreachable the exporter retries
        for much longer than a container's stop grace period; unsent spans are dropped instead."""
        def run() -> None:
            try:
                call()
            except Exception as exc:  # noqa: BLE001
                _tracing_failed(stage, exc)

        worker = threading.Thread(target=run, name=f"langfuse-{stage}", daemon=True)
        worker.start()
        worker.join(self.export_timeout_seconds)
        if worker.is_alive():
            logger.warning(json.dumps({"event": "tracing_failed", "stage": stage, "exception": "Timeout",
                                       "timeout_seconds": self.export_timeout_seconds}))


def _tracing_failed(stage: str, exc: Exception) -> None:
    logger.warning(json.dumps({"event": "tracing_failed", "stage": stage, "exception": type(exc).__name__}))


_TRACER: Tracer = NoopTracer()


def get_tracer() -> Tracer:
    return _TRACER


def set_tracer(tracer: Tracer) -> Tracer:
    """Install a process-wide tracer; returns the previous one."""
    global _TRACER
    previous, _TRACER = _TRACER, tracer
    return previous


@contextmanager
def use_tracer(tracer: Tracer) -> Iterator[Tracer]:
    previous = set_tracer(tracer)
    try:
        yield tracer
    finally:
        set_tracer(previous)
