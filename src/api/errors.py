"""Application-level API error contract.

Every non-2xx response has the same body:

    {"request_id": "...", "run_id": null, "error": {"code": "...", "message": "...",
     "retryable": false, "details": []}}

Messages are safe to show to clients: they never contain stack traces, provider
payloads or CV text. Tool errors keep the tool layer's classification.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.tools.schemas import ToolError

logger = logging.getLogger("src.api")

ApiErrorCode = Literal[
    "validation_error",        # 422 request body/query does not match the schema
    "job_not_found",           # 404 unknown job_id
    "cv_not_available",        # 409 the configured CV is not ingested
    "dependency_unavailable",  # 503 transient tool dependency failure (after bounded retries)
    "tool_failure",            # 500 unexpected tool failure
    "llm_provider_error",      # 502 the LLM provider call failed
    "agent_limit_exceeded",    # 422 the agent hit a model-turn, tool-call or invalid-request limit
    "chat_unavailable",        # 503 no LLM provider configured
    "not_found",               # 404 unknown route
    "method_not_allowed",      # 405
    "internal_error",          # 500 anything else
]


class ErrorBody(BaseModel):
    code: ApiErrorCode
    message: str
    retryable: bool = False
    details: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    request_id: str
    run_id: str | None = None
    error: ErrorBody


class ApiError(Exception):
    def __init__(self, status_code: int, code: ApiErrorCode, message: str, *, retryable: bool = False,
                 details: list[dict[str, Any]] | None = None, run_id: str | None = None) -> None:
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message
        self.retryable, self.details, self.run_id = retryable, details or [], run_id

    def body(self, request_id: str) -> ErrorResponse:
        return ErrorResponse(request_id=request_id, run_id=self.run_id, error=ErrorBody(
            code=self.code, message=self.message, retryable=self.retryable, details=self.details))


_TOOL_ERRORS: dict[str, tuple[int, ApiErrorCode]] = {
    "not_found": (404, "job_not_found"),
    "precondition_failed": (409, "cv_not_available"),
    "transient": (503, "dependency_unavailable"),
    "internal": (500, "tool_failure"),
    # Request models mirror the tool argument models, so these indicate a server-side defect.
    "invalid_arguments": (500, "internal_error"),
    "unknown_tool": (500, "internal_error"),
}

_AGENT_ERRORS: dict[str, tuple[int, ApiErrorCode, bool]] = {
    "provider_error": (502, "llm_provider_error", True),
    "max_model_turns_exceeded": (422, "agent_limit_exceeded", False),
    "max_tool_calls_exceeded": (422, "agent_limit_exceeded", False),
    "invalid_tool_requests": (422, "agent_limit_exceeded", False),
    "graph_recursion_limit": (422, "agent_limit_exceeded", False),
}


def from_tool_error(error: ToolError) -> ApiError:
    status, code = _TOOL_ERRORS[error.code]
    message = error.message if status < 500 or code == "dependency_unavailable" else "The tool failed unexpectedly."
    return ApiError(status, code, message, retryable=error.retryable,
                    details=[{"tool_error_code": error.code}])


def from_agent_status(status: str, message: str | None, run_id: str) -> ApiError:
    http_status, code, retryable = _AGENT_ERRORS[status]
    return ApiError(http_status, code, message or "The agent run did not complete.", retryable=retryable,
                    details=[{"agent_status": status}], run_id=run_id)


def log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))
