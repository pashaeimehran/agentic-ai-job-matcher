"""Run the MCP server over stdio for local MCP clients.

    python -m scripts.serve_mcp

Client configuration (command + args, run from the repository root):

    {"command": "<repo>/venv/Scripts/python.exe", "args": ["-m", "scripts.serve_mcp"], "cwd": "<repo>"}

Uses the synthetic CV and fixture catalog; saved jobs go to data/private/.
No LLM is called by the server: the MCP client's own model chooses the tools.
Logs go to stderr because stdout carries the protocol.
"""
from __future__ import annotations

import argparse
import logging
import sys

import anyio
from mcp.server.stdio import stdio_server

from src.mcp_server import build_mcp_server
from src.tools import default_registry
from src.tools.context import build_offline_context
from src.tools.saved_jobs import JsonFileSavedJobStore, MemorySavedJobStore


async def serve(memory_saved_jobs: bool) -> None:
    store = MemorySavedJobStore() if memory_saved_jobs else JsonFileSavedJobStore()
    server = build_mcp_server(default_registry(), build_offline_context(saved_jobs=store))
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the shared job-matching tools over MCP (stdio)")
    parser.add_argument("--memory-saved-jobs", action="store_true",
                        help="Keep saved jobs in memory instead of data/private/saved_jobs.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(name)s %(message)s")
    anyio.run(serve, args.memory_saved_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
