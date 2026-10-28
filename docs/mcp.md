# MCP Server

The MCP server exposes the five shared tools to external LLM-driven clients (desktop assistants,
IDE agents, other agent frameworks) over the Model Context Protocol.

```
MCP client → src/mcp_server/server.py (adapter) → ToolRegistry.execute() → shared services
```

## Design

- `tools/list` returns `registry.specs()`: the names, descriptions and JSON schemas generated from
  the tool argument models. The MCP SDK does not generate a second schema.
- `tools/call` forwards the name and arguments to `ToolRegistry.execute()`. The registry remains
  the validation authority: strict types, no unknown keys, bounded values. The low-level MCP SDK
  server does not validate arguments, and any validation an MCP client performs is advisory.
- Results are the same `ToolOutcome` envelope the agents observe, returned both as
  `structured_content` and as JSON text. `is_error` is `true` exactly when `status == "error"`.
- Synchronous tool handlers run in a worker thread (`anyio.to_thread`), not on the event loop.
- The CV scope comes from the server's `ToolContext`; clients cannot select a CV.
- No LLM runs inside the server. The client's own model decides which tools to call.

The adapter uses the SDK's low-level `Server` rather than the decorator API (`MCPServer`). The
decorator API derives schemas from Python function signatures; that would duplicate the argument
models and create a second validation path.

## Exposed tools

| Tool | MCP annotations |
|---|---|
| `search_jobs` | read-only, closed-world |
| `get_job_details` | read-only, closed-world |
| `search_cv` | read-only, closed-world |
| `calculate_match` | read-only, closed-world; the only source of match scores |
| `save_job` | state-changing, non-destructive, idempotent |

`build_mcp_server(registry, context, exposed=(...))` can expose a subset. A tool outside the
exposed set is answered as `unknown_tool`, and the error lists only exposed tools.

Not exposed: `/health`, catalog paging, and the LangGraph agent (`/chat`). An MCP client is already
an agent; giving it a nested agent would put one LLM loop inside another and hide the tool calls
from the client. MCP exposes capabilities, not every REST endpoint.

## Running

```bash
python -m scripts.serve_mcp                       # stdio transport
python -m scripts.serve_mcp --memory-saved-jobs   # do not write data/private/saved_jobs.json
```

Generic client configuration (paths are placeholders; run from the repository root):

```json
{"mcpServers": {"job-match-agent": {
  "command": "<repo>/venv/Scripts/python.exe",
  "args": ["-m", "scripts.serve_mcp"],
  "cwd": "<repo>"}}}
```

Logs go to stderr because stdout carries the protocol. The server uses the synthetic CV and the
fixture catalog.

Example call and result (`calculate_match`):

```json
{"name": "calculate_match", "arguments": {"job_ids": ["fixture-01"]}}
```
```json
{"tool_name": "calculate_match", "status": "ok", "attempts": 1, "error": null,
 "data": {"matches": [{"rank": 1, "job_id": "fixture-01", "score": {"value": 69.59, "...": "..."}, "...": "..."}],
          "scoring_note": "score.value is calculated by deterministic Python code (0-100). ..."}}
```

Invalid arguments are returned as a tool error, without executing the handler:

```json
{"tool_name": "search_jobs", "status": "error", "attempts": 0, "data": null,
 "error": {"code": "invalid_arguments", "retryable": false,
           "message": "Arguments do not match the tool schema; correct them and call again.",
           "details": [{"loc": ["limit"], "msg": "Input should be a valid integer", "type": "int_type"}]}}
```

## REST and MCP

| | REST (`src/api`) | MCP (`src/mcp_server`) |
|---|---|---|
| Consumer | frontends, services, scripts | LLM-driven clients choosing tools |
| Contract | stable HTTP resources, status codes, `ErrorResponse` | tool list with JSON schemas; `ToolOutcome` results |
| Agent | includes the LangGraph agent (`/chat`) | none: the client is the agent |
| Errors | mapped to HTTP codes | returned as tool results (`is_error`) so the client's model can react |
| Shared | `ToolRegistry.execute()`, `ToolContext`, the tools and services behind them | identical |

## Tests

`tests/unit/test_mcp_server.py` runs an MCP `Client` against the server in-process and also launches
the real stdio entry point as a subprocess. It verifies that:

- listed schemas and descriptions equal `registry.specs()` exactly;
- each of the five tools reaches the registry (a spy records the call) and returns the same data as a
  direct `registry.execute()`;
- `calculate_match` scores equal a direct `analyze_jobs()` computation;
- `save_job` writes to the context's shared store;
- strict validation still rejects string-typed integers, unknown keys and missing fields, with zero
  handler attempts;
- domain errors are structured, and unexposed or unknown tools are rejected.

No desktop MCP host is required.

## Limitations

- stdio transport only; no authentication (a local, single-user process).
- MCP resources and prompts are not implemented; only tools.
- A PostgreSQL-backed CV requires constructing the server with a live `ToolContext`; there is no CLI flag for it yet.
