"""Configuration and structured results for a manual agent run."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.tools.errors import ErrorCode
from src.tools.registry import RetryPolicy
from src.tools.schemas import ToolOutcome

RunStatus = Literal["completed", "max_model_turns_exceeded", "max_tool_calls_exceeded", "provider_error",
                    "budget_exceeded"]


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # A model turn is one provider.complete() call.
    max_model_turns: int = Field(default=8, ge=1, le=50)
    # Counts every requested call, including rejected or failed ones.
    max_tool_calls: int = Field(default=12, ge=0, le=100)
    tool_retry: RetryPolicy = RetryPolicy()
    # Per-run LLM spending limits, enforced before each model call by the instrumented
    # provider (src.observability). None disables the limit.
    max_run_tokens: int | None = Field(default=None, ge=1)
    max_run_cost_usd: float | None = Field(default=None, gt=0)

    def budget(self):
        from src.observability.context import RunBudget
        return RunBudget(max_tokens=self.max_run_tokens, max_cost_usd=self.max_run_cost_usd)


class ToolCallRecord(BaseModel):
    step: int = Field(ge=1)
    call_id: str
    tool_name: str
    arguments_valid: bool
    status: Literal["ok", "empty", "error"]
    error_code: ErrorCode | None = None
    attempts: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    outcome: ToolOutcome


class AgentRunResult(BaseModel):
    run_id: str
    status: RunStatus
    final_answer: str | None = None
    model_turns: int = Field(ge=0)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    error: str | None = None
    # Numbers presented as scores in the answer that no calculate_match call produced.
    ungrounded_scores: list[str] = Field(default_factory=list)

    @property
    def tool_sequence(self) -> list[str]:
        return [record.tool_name for record in self.tool_calls]
