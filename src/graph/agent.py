"""Graph assembly, checkpointing and the LangGraphAgent entry point."""
from __future__ import annotations

import uuid
from functools import partial
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from src.agent.prompts import SYSTEM_PROMPT
from src.graph.nodes import GraphNodes, route_after_agent, route_after_tools, traced_node
from src.graph.state import AgentState, GraphConfig, GraphRunResult
from src.llm.tool_calling import ToolCallingLLMProvider
from src.observability.context import run_scope
from src.observability.tracing import get_tracer
from src.tools.registry import ToolRegistry

# Every application type that can appear in checkpointed state. Anything else is
# refused on load instead of being deserialized implicitly.
CHECKPOINT_TYPES = [
    ("src.llm.tool_calling", "ChatMessage"),
    ("src.llm.tool_calling", "AssistantTurn"),
    ("src.llm.tool_calling", "ToolCall"),
    ("src.llm.tool_calling", "ToolResultMessage"),
    ("src.agent.schemas", "ToolCallRecord"),
    ("src.tools.schemas", "ToolOutcome"),
    ("src.tools.schemas", "ToolError"),
]


def memory_checkpointer() -> InMemorySaver:
    """Process-local conversation memory keyed by thread_id (not durable, not multi-user)."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


def build_graph(nodes: GraphNodes, checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(AgentState)
    for name in ("prepare", "agent", "execute_tools", "finalize", "fail"):
        graph.add_node(name, traced_node(name, getattr(nodes, name)))
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["execute_tools", "finalize", "fail"])
    graph.add_conditional_edges("execute_tools", partial(route_after_tools, limits=nodes.config),
                                ["agent", "fail"])
    graph.add_edge("finalize", END)
    graph.add_edge("fail", END)
    return graph.compile(checkpointer=checkpointer)


class LangGraphAgent:
    """Same contract as ManualAgent.run(), orchestrated by an explicit state graph."""

    def __init__(self, provider: ToolCallingLLMProvider, registry: ToolRegistry, context: Any,
                 config: GraphConfig = GraphConfig(), system_prompt: str = SYSTEM_PROMPT,
                 checkpointer: BaseCheckpointSaver | None = None) -> None:
        self.config = config
        self.checkpointer = checkpointer
        self.graph = build_graph(GraphNodes(provider, registry, context, config, system_prompt), checkpointer)

    def run(self, request: str, run_id: str | None = None, thread_id: str | None = None) -> GraphRunResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        if thread_id is not None and self.checkpointer is None:
            raise ValueError("thread_id requires a checkpointer")
        if self.checkpointer is not None:
            thread_id = thread_id or run_id
        invoke_config: dict = {"recursion_limit": self.config.recursion_limit()}
        if thread_id:
            invoke_config["configurable"] = {"thread_id": thread_id}
        with run_scope(run_id, budget=self.config.budget()), get_tracer().span(
                "agent.langgraph", kind="agent", input=request,
                metadata={"run_id": run_id, "thread_id": thread_id}) as span:
            result = self._invoke(request, run_id, thread_id, invoke_config)
            span.update(metadata={"status": result.status, "model_turns": result.model_turns,
                                  "tool_calls": len(result.tool_calls), "path": result.path},
                        output=result.final_answer or result.error)
            return result

    def _invoke(self, request: str, run_id: str, thread_id: str | None, invoke_config: dict) -> GraphRunResult:
        try:
            state = self.graph.invoke({"request": request, "run_id": run_id}, invoke_config)
        except GraphRecursionError:
            # Unreachable while routing is correct; reported rather than raised if it ever is.
            state = self.graph.get_state(invoke_config).values if self.checkpointer else {}
            state = {**state, "status": "graph_recursion_limit",
                     "error": f"Graph recursion limit of {self.config.recursion_limit()} reached."}
        return GraphRunResult(run_id=run_id, thread_id=thread_id, status=state["status"],
                              final_answer=state.get("final_answer"),
                              model_turns=state.get("model_turns", 0),
                              tool_calls=state.get("tool_calls", []), error=state.get("error"),
                              ungrounded_scores=state.get("ungrounded_scores", []),
                              path=state.get("path", []))
