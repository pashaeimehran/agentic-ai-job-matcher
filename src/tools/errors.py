"""Tool failure classification and the bounded retry policy.

Only failures classified as retryable are retried. Unknown exceptions are
non-retryable and are reported with a generic message, because exception text
may contain CV content, SQL parameters or provider payloads.
"""
from __future__ import annotations

from typing import Literal

ErrorCode = Literal[
    "unknown_tool",        # model asked for a tool that is not registered
    "invalid_arguments",   # JSON/schema validation failed; tool not executed
    "not_found",           # referenced entity (e.g. job_id) does not exist
    "precondition_failed", # application state prevents execution (e.g. CV not ingested)
    "transient",           # temporary dependency failure; retryable
    "internal",            # unexpected exception; not retryable
]


class ToolExecutionError(Exception):
    """Raised by tool implementations with a message that is safe to show the model."""
    code: ErrorCode = "internal"
    retryable = False

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.safe_message = message


class ToolNotFoundError(ToolExecutionError):
    code = "not_found"


class ToolPreconditionError(ToolExecutionError):
    code = "precondition_failed"


class TransientToolError(ToolExecutionError):
    code = "transient"
    retryable = True


def _transient_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [TimeoutError, ConnectionError]
    try:
        from src.data_sources.ba_jobsuche import BAJobsucheUnavailable
        types.append(BAJobsucheUnavailable)
    except ImportError:  # pragma: no cover - optional dependency path
        pass
    try:
        import psycopg
        types.append(psycopg.OperationalError)
    except ImportError:  # pragma: no cover - psycopg is optional for offline use
        pass
    return tuple(types)


_TRANSIENT = _transient_types()


def classify_exception(exc: BaseException) -> tuple[ErrorCode, bool, str]:
    """Return (code, retryable, safe_message) for an exception raised by a tool."""
    if isinstance(exc, ToolExecutionError):
        return exc.code, exc.retryable, exc.safe_message
    if isinstance(exc, _TRANSIENT):
        return "transient", True, f"Temporary dependency failure ({type(exc).__name__})."
    return "internal", False, f"Tool failed unexpectedly ({type(exc).__name__})."
