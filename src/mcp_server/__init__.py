"""MCP server exposing the shared tool registry to external LLM clients."""
from src.mcp_server.server import DEFAULT_EXPOSED, build_mcp_server

__all__ = ["DEFAULT_EXPOSED", "build_mcp_server"]
