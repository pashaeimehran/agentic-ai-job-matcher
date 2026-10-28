"""MCP server adapter over the shared tool registry.

tools/list returns registry.specs() (the JSON schemas generated from the tool
argument models). tools/call forwards to ToolRegistry.execute(), which remains
the validation authority: the SDK does not validate arguments here, and any
schema check an MCP client performs is advisory. Results are the same
ToolOutcome envelope the agents observe.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server

from src.tools.registry import RetryPolicy, ToolRegistry
from src.tools.schemas import ToolError, ToolOutcome

logger = logging.getLogger("src.mcp_server")

SERVER_NAME = "job-match-agent"
INSTRUCTIONS = ("Job search, CV evidence retrieval and deterministic CV-job matching for one configured CV. "
                "Match scores come only from calculate_match; quote score.value exactly.")

# Behaviour hints for MCP clients. Only save_job changes state; it is idempotent.
_ANNOTATIONS = {
    "search_jobs": types.ToolAnnotations(read_only_hint=True, open_world_hint=False),
    "get_job_details": types.ToolAnnotations(read_only_hint=True, open_world_hint=False),
    "search_cv": types.ToolAnnotations(read_only_hint=True, open_world_hint=False),
    "calculate_match": types.ToolAnnotations(read_only_hint=True, open_world_hint=False),
    "save_job": types.ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True,
                                      open_world_hint=False),
}
DEFAULT_EXPOSED = tuple(_ANNOTATIONS)


def _log(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))


def build_mcp_server(registry: ToolRegistry, context: Any, retry: RetryPolicy = RetryPolicy(),
                     exposed: tuple[str, ...] = DEFAULT_EXPOSED) -> Server:
    missing = [name for name in exposed if registry.get(name) is None]
    if missing:
        raise ValueError(f"Cannot expose unregistered tools: {missing}")

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=spec.name, description=spec.description, input_schema=spec.parameters,
                       annotations=_ANNOTATIONS.get(spec.name))
            for spec in registry.specs() if spec.name in exposed])

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        run_id = f"mcp-{ctx.request_id}"
        if params.name in exposed:
            arguments = params.arguments or {}
            # Tool handlers are synchronous (database, embeddings); keep them off the event loop.
            outcome = await anyio.to_thread.run_sync(
                lambda: registry.execute(params.name, arguments, context, retry, run_id=run_id))
        else:
            # Unexposed tools are reported like unknown ones, without naming hidden tools.
            outcome = ToolOutcome(tool_name=params.name, status="error", attempts=0, duration_ms=0,
                                  error=ToolError(code="unknown_tool", retryable=False,
                                                  message=f"Tool '{params.name}' does not exist. "
                                                          f"Available: {', '.join(exposed)}."))
        _log("mcp_tool_call", run_id=run_id, tool=params.name, status=outcome.status,
             error_code=outcome.error.code if outcome.error else None)
        payload = outcome.model_dump(mode="json", exclude={"duration_ms"})
        return types.CallToolResult(content=[types.TextContent(text=json.dumps(payload))],
                                    structured_content=payload, is_error=outcome.status == "error")

    return Server(SERVER_NAME, version="0.1.0", instructions=INSTRUCTIONS,
                  on_list_tools=list_tools, on_call_tool=call_tool)
