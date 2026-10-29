"""LangGraph nodes and conditional routing.

Nodes orchestrate; they contain no retrieval, matching or scoring logic. Tool
work goes through ToolRegistry.execute() exactly as in the manual loop, and
answers are checked with the manual loop's ungrounded_scores().

    START -> prepare -> agent --(route_after_agent)--> execute_tools | finalize | fail
                          ^                                  |
                          +------(route_after_tools)---------+--> fail
    finalize -> END,  fail -> END

Routing functions are pure functions of (state, limits) so they can be tested
without a model or a compiled graph.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.loop import ungrounded_scores
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.schemas import ToolCallRecord
from src.llm.tool_calling import (AssistantTurn, ChatMessage, ConversationItem, ToolCallingLLMProvider,
                                  ToolResultMessage)
from src.graph.state import AgentState, GraphConfig, Route
from src.observability.context import BudgetExceededError
from src.observability.tracing import get_tracer
from src.tools.registry import ToolRegistry

logger = logging.getLogger("src.graph")

_REJECTED = ("invalid_arguments", "unknown_tool")


def _log(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))


def close_dangling_calls(messages: list[ConversationItem]) -> list[ConversationItem]:
    """Answer tool calls a previous run requested but never executed.

    A run stopped by a limit can end with an assistant turn whose calls have no
    results. Providers reject such a history, so a continued thread gets an
    explicit 'not executed' observation for each of them.
    """
    answered = {m.call_id for m in messages if isinstance(m, ToolResultMessage)}
    missing = [c for m in messages if isinstance(m, AssistantTurn) for c in m.tool_calls
               if c.call_id not in answered]
    return [*messages, *(ToolResultMessage(call_id=c.call_id, name=c.name, content=json.dumps(
        {"tool_name": c.name, "status": "not_executed",
         "message": "The previous run stopped before this call was executed."}))
        for c in missing)]


# --- Routing (pure) --------------------------------------------------------------

def route_after_agent(state: AgentState) -> Route:
    if state.get("status") in ("provider_error", "budget_exceeded"):
        return "fail"
    return "execute_tools" if state.get("pending_calls") else "finalize"


def route_after_tools(state: AgentState, limits: GraphConfig) -> Route:
    if state.get("status") is not None:  # tool budget or invalid-request guard tripped
        return "fail"
    if state.get("model_turns", 0) >= limits.max_model_turns:
        return "fail"
    return "agent"


def failure_status(state: AgentState) -> str:
    """Status recorded by the fail node; a missing status means the turn budget ran out."""
    return state.get("status") or "max_model_turns_exceeded"


def traced_node(name: str, node):
    """Wrap a node in a "graph.<name>" span; the node itself stays free of tracing code."""
    def run(state: AgentState) -> dict:
        metadata = {"run_id": state.get("run_id"), "model_turns": state.get("model_turns", 0)}
        with get_tracer().span(f"graph.{name}", kind="chain", metadata=metadata) as span:
            update = node(state)
            span.update(metadata={"status": update.get("status"),
                                  "pending_calls": len(update.get("pending_calls") or [])})
            return update
    run.__name__ = name
    return run


# --- Nodes ----------------------------------------------------------------------

class GraphNodes:
    def __init__(self, provider: ToolCallingLLMProvider, registry: ToolRegistry, context: Any,
                 config: GraphConfig = GraphConfig(), system_prompt: str = SYSTEM_PROMPT) -> None:
        self.provider, self.registry, self.context = provider, registry, context
        self.config, self.system_prompt = config, system_prompt

    def prepare(self, state: AgentState) -> dict:
        """Understand Request: start a run on a new or continued thread; no LLM call."""
        history = list(state.get("messages") or [])
        messages = close_dangling_calls(history) if history else [
            ChatMessage(role="system", content=self.system_prompt)]
        messages.append(ChatMessage(role="user", content=state["request"]))
        cfg = self.config
        # The request text is not logged: it may contain personal data.
        _log("agent_run_started", run_id=state["run_id"], engine="langgraph",
             continued_thread=bool(history), max_model_turns=cfg.max_model_turns,
             max_tool_calls=cfg.max_tool_calls, tools=self.registry.names())
        return {"messages": messages, "pending_calls": [], "tool_calls": [], "model_turns": 0,
                "invalid_tool_turns": 0, "path": ["prepare"], "status": None, "final_answer": None,
                "error": None, "ungrounded_scores": []}

    def agent(self, state: AgentState) -> dict:
        """One model turn: the model either requests tools or answers."""
        step = state.get("model_turns", 0) + 1
        path = [*state.get("path", []), "agent"]
        try:
            reply = self.provider.complete(state["messages"], self.registry.specs())
        except BudgetExceededError as exc:
            _log("model_turn_failed", run_id=state["run_id"], step=step, exception=type(exc).__name__)
            return {"model_turns": step - 1, "path": path, "pending_calls": [], "status": "budget_exceeded",
                    "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - provider payloads are not echoed
            _log("model_turn_failed", run_id=state["run_id"], step=step, exception=type(exc).__name__)
            return {"model_turns": step, "path": path, "pending_calls": [], "status": "provider_error",
                    "error": f"Model call failed ({type(exc).__name__})."}
        _log("model_turn", run_id=state["run_id"], step=step,
             tool_calls=[c.name for c in reply.tool_calls], has_text=bool(reply.text))
        return {"messages": [*state["messages"], reply], "pending_calls": list(reply.tool_calls),
                "model_turns": step, "path": path}

    def execute_tools(self, state: AgentState) -> dict:
        """Run every requested call through the shared registry, within the tool budget."""
        cfg, step, run_id = self.config, state["model_turns"], state["run_id"]
        messages, records = list(state["messages"]), list(state.get("tool_calls", []))
        executed: list[ToolCallRecord] = []
        update: dict = {}
        for call in state.get("pending_calls", []):
            if len(records) >= cfg.max_tool_calls:
                update = {"status": "max_tool_calls_exceeded",
                          "error": f"Tool-call limit of {cfg.max_tool_calls} reached."}
                break
            outcome = self.registry.execute(call.name, call.arguments, self.context, cfg.tool_retry,
                                            run_id=run_id, step=step)
            code = outcome.error.code if outcome.error else None
            record = ToolCallRecord(step=step, call_id=call.call_id, tool_name=call.name,
                                    arguments_valid=code not in _REJECTED, status=outcome.status,
                                    error_code=code, attempts=outcome.attempts,
                                    duration_ms=outcome.duration_ms, outcome=outcome)
            records.append(record)
            executed.append(record)
            messages.append(ToolResultMessage(call_id=call.call_id, name=call.name,
                                              content=outcome.model_dump_json(exclude={"duration_ms"})))
        all_rejected = bool(executed) and all(not r.arguments_valid for r in executed)
        invalid_turns = state.get("invalid_tool_turns", 0) + 1 if all_rejected else 0
        if not update and invalid_turns >= cfg.max_invalid_tool_turns:
            update = {"status": "invalid_tool_requests",
                      "error": f"Stopped after {invalid_turns} consecutive turns with only invalid "
                               "tool requests."}
        return {"messages": messages, "tool_calls": records, "pending_calls": [],
                "invalid_tool_turns": invalid_turns, "path": [*state.get("path", []), "execute_tools"],
                **update}

    def finalize(self, state: AgentState) -> dict:
        """Produce the final recommendation from the model's last answer."""
        last = state["messages"][-1]
        final = (last.text if isinstance(last, AssistantTurn) else None) or ""
        flagged = ungrounded_scores(final, state.get("tool_calls", [])) if final else []
        self._finished(state, "completed", flagged)
        return {"status": "completed", "final_answer": final, "ungrounded_scores": flagged,
                "path": [*state.get("path", []), "finalize"]}

    def fail(self, state: AgentState) -> dict:
        """Terminate with a structured failure (provider error or exhausted limit)."""
        status = failure_status(state)
        error = state.get("error") or (f"Model-turn limit of {self.config.max_model_turns} reached "
                                       "without a final answer.")
        self._finished(state, status, [])
        return {"status": status, "error": error, "final_answer": None,
                "path": [*state.get("path", []), "fail"]}

    @staticmethod
    def _finished(state: AgentState, status: str, flagged: list[str]) -> None:
        _log("agent_run_finished", run_id=state["run_id"], engine="langgraph", status=status,
             model_turns=state.get("model_turns", 0), tool_calls=len(state.get("tool_calls", [])),
             ungrounded_scores=len(flagged))
