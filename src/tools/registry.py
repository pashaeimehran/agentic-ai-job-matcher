"""Tool registry: name -> argument schema -> implementation, plus safe execution.

execute() is the single entry point used by the manual agent loop and is
intended to be reused unchanged by LangGraph nodes, FastAPI routes or an MCP
server. It never raises for tool-level problems; it returns a ToolOutcome.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.llm.tool_calling import ToolSpec
from src.observability.tracing import get_tracer
from src.tools.errors import classify_exception
from src.tools.schemas import StrictArgs, ToolError, ToolOutcome, ToolResult

logger = logging.getLogger("src.tools")

A = TypeVar("A", bound=StrictArgs)
R = TypeVar("R", bound=ToolResult)


class RetryPolicy(BaseModel):
    """Bounded retries for failures explicitly classified as retryable.

    max_retries counts re-executions after the first attempt. Validation
    errors, unknown tools, not_found, precondition and internal failures are
    never retried by the executor.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_retries: int = Field(default=1, ge=0, le=5)
    backoff_seconds: float = Field(default=0.0, ge=0, le=10)


@dataclass(frozen=True)
class Tool(Generic[A, R]):
    name: str
    description: str
    args_model: type[A]
    handler: Callable[[A, Any], R]

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, parameters=schema)


def _log(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def validate(self, name: str, raw_arguments: str | dict[str, Any]) -> StrictArgs | ToolError:
        """Parse model arguments into the tool's schema without executing anything."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolError(code="unknown_tool", retryable=False,
                             message=f"Tool '{name}' does not exist. Available: {', '.join(self._tools)}.")
        try:
            if isinstance(raw_arguments, str):
                return tool.args_model.model_validate_json(raw_arguments or "{}")
            return tool.args_model.model_validate(raw_arguments)
        except ValidationError as exc:
            # Omit 'input' values: they may echo long or sensitive text into logs.
            details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                       for e in exc.errors(include_url=False, include_input=False)]
            return ToolError(code="invalid_arguments", retryable=False, details=details,
                             message="Arguments do not match the tool schema; correct them and call again.")

    def execute(self, name: str, raw_arguments: str | dict[str, Any], context: Any,
                retry: RetryPolicy = RetryPolicy(), *, run_id: str | None = None,
                step: int | None = None) -> ToolOutcome:
        """Validate and run one tool inside a "tool.<name>" span (metadata only unless content capture is on)."""
        tracer = get_tracer()
        with tracer.span(f"tool.{name}", kind="tool", metadata={"run_id": run_id, "step": step},
                         input=raw_arguments) as span:
            outcome = self._execute(name, raw_arguments, context, retry, run_id=run_id, step=step)
            span.update(metadata={"status": outcome.status, "attempts": outcome.attempts,
                                  "error_code": outcome.error.code if outcome.error else None,
                                  "duration_ms": outcome.duration_ms},
                        **({"level": "ERROR", "status_message": outcome.error.code} if outcome.error else {}),
                        **({"output": outcome.model_dump(mode="json")} if tracer.capture_content else {}))
            return outcome

    def _execute(self, name: str, raw_arguments: str | dict[str, Any], context: Any,
                 retry: RetryPolicy, *, run_id: str | None, step: int | None) -> ToolOutcome:
        started = time.perf_counter()
        meta = {"run_id": run_id, "step": step, "tool": name}
        args = self.validate(name, raw_arguments)
        if isinstance(args, ToolError):
            _log("tool_rejected", **meta, error_code=args.code, arguments_valid=False)
            return ToolOutcome(tool_name=name, status="error", error=args, attempts=0,
                               duration_ms=round((time.perf_counter() - started) * 1000, 3))
        tool = self._tools[name]
        attempts = 0
        while True:
            attempts += 1
            try:
                result = tool.handler(args, context)
            except Exception as exc:  # noqa: BLE001 - classified, never propagated
                code, retryable, message = classify_exception(exc)
                will_retry = retryable and attempts <= retry.max_retries
                _log("tool_failed", **meta, arguments_valid=True, attempt=attempts,
                     error_code=code, exception=type(exc).__name__, retryable=retryable,
                     will_retry=will_retry)
                if will_retry:
                    if retry.backoff_seconds:
                        time.sleep(retry.backoff_seconds * attempts)
                    continue
                duration = round((time.perf_counter() - started) * 1000, 3)
                return ToolOutcome(tool_name=name, status="error", attempts=attempts, duration_ms=duration,
                                   error=ToolError(code=code, message=message, retryable=retryable))
            duration = round((time.perf_counter() - started) * 1000, 3)
            status = "empty" if result.is_empty() else "ok"
            _log("tool_succeeded", **meta, arguments_valid=True, attempts=attempts,
                 status=status, duration_ms=duration)
            return ToolOutcome(tool_name=name, status=status, data=result.model_dump(mode="json"),
                               attempts=attempts, duration_ms=duration)
