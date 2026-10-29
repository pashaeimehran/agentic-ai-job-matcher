"""Explicit LangGraph state, configuration and run result.

The state holds orchestration data only. Business results (tool outcomes,
deterministic match scores) are produced by src.tools and copied in unchanged.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import Field

from src.agent.schemas import AgentConfig, AgentRunResult, ToolCallRecord
from src.llm.tool_calling import ConversationItem, ToolCall

GraphRunStatus = Literal["completed", "max_model_turns_exceeded", "max_tool_calls_exceeded",
                         "provider_error", "budget_exceeded", "invalid_tool_requests", "graph_recursion_limit"]

# Terminal routes taken from the graph's conditional edges.
Route = Literal["execute_tools", "finalize", "fail", "agent"]


class GraphConfig(AgentConfig):
    """AgentConfig plus graph-only guards.

    max_invalid_tool_turns: consecutive model turns in which every requested
    call was rejected (unknown tool or invalid arguments) before the run is
    terminated. The manual loop has no such guard and spends its model-turn
    budget instead.
    """
    max_invalid_tool_turns: int = Field(default=2, ge=1, le=10)

    def recursion_limit(self) -> int:
        # prepare + (agent + execute_tools) per model turn + final node, with headroom.
        # Routing ends the run long before this; it only backstops a wiring defect.
        return 2 * self.max_model_turns + 4


class AgentState(TypedDict, total=False):
    # --- input ---
    request: str
    run_id: str
    # --- conversation (persisted per thread when a checkpointer is configured) ---
    messages: list[ConversationItem]
    # --- per-run orchestration, reset by the prepare node ---
    pending_calls: list[ToolCall]
    tool_calls: list[ToolCallRecord]
    model_turns: int
    invalid_tool_turns: int
    path: list[str]
    # --- outcome ---
    status: GraphRunStatus | None
    final_answer: str | None
    error: str | None
    ungrounded_scores: list[str]


class GraphRunResult(AgentRunResult):
    """AgentRunResult with graph-specific statuses and the visited node path."""
    status: GraphRunStatus
    thread_id: str | None = None
    path: list[str] = Field(default_factory=list)
