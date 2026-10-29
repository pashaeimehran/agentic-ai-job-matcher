# Observability

Instrumentation sits at four boundaries, so business logic (matching, retrieval, tools) contains no
tracing or metrics code:

| Boundary | Where | Produces |
|---|---|---|
| HTTP request | middleware in `src/api/app.py` | root span `METHOD /path`, `http_request` log line, `X-Request-ID` |
| Agent run | `LangGraphAgent.run`, `ToolCallingAgent.run` | `agent.langgraph` / `agent.manual` span, run scope with `run_id` and budget |
| Graph node / tool | `src/graph/nodes.py`, `ToolRegistry.execute` | `graph.<node>` and `tool.<name>` spans with status, error code, attempts |
| Model call | `InstrumentedProvider` / `InstrumentedEmbeddings` (`src/observability/instrumented.py`) | `llm.complete` / `llm.generate` / `llm.embed` generation span and one `LLMCallRecord` |

`request_id` and `run_id` travel in a `contextvars` scope (`src/observability/context.py`) instead of
being passed through function signatures.

## Trace hierarchy

Recorded in the local Langfuse stack (fixture mode, `POST /chat` "Find AI Engineer jobs in Germany
that fit my CV"):

```
SPAN       POST /chat
└─ AGENT   agent.langgraph
   ├─ CHAIN graph.prepare
   ├─ CHAIN graph.agent
   │  └─ GENERATION llm.complete        model, usage, cost, cache_hit, attempts, prompt_version
   ├─ CHAIN graph.execute_tools
   │  └─ TOOL tool.search_jobs
   ├─ CHAIN graph.agent
   │  └─ GENERATION llm.complete
   ├─ CHAIN graph.execute_tools
   │  └─ TOOL tool.calculate_match
   ├─ CHAIN graph.agent
   │  └─ GENERATION llm.complete
   └─ CHAIN graph.finalize
```

Deterministic endpoints produce `POST /analyze-job → tool.calculate_match` with no generation.
`/health` and `/ready` are not traced, so probes don't flood the trace store.

## Langfuse

- Backend: Langfuse SDK (OpenTelemetry-based) in `LangfuseTracer`. It is enabled when
  `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, or with `TRACING_ENABLED=true`.
- Self-hosted: `docker compose --profile observability up -d --wait` starts Langfuse v4 (web,
  worker, PostgreSQL, ClickHouse, Redis, MinIO) with a headless-initialized local project. Set
  `TRACING_ENABLED=true` for the API service. UI: <http://localhost:3000> (local admin from
  `compose.yaml`; change before any non-local use).
- Optional dependency: if Langfuse is missing, misconfigured or unreachable, spans degrade to
  no-ops. Flush and shutdown are capped at 3 s each, so a down backend cannot block container
  shutdown. Validated: with `langfuse-web` stopped, `/chat` still returned 200 and the API exited
  cleanly (exit 0) on `docker compose stop`.
- Content policy: spans carry names, IDs, statuses, counts, model and usage only. Inputs and
  outputs (prompts, tool results, CV excerpts) are attached only with
  `LANGFUSE_CAPTURE_CONTENT=true`. Use that only with a trusted self-hosted instance.

## Operational metrics and cost log

`LLMCallRecord` (`src/observability/metrics.py`) holds: record/run/request IDs, timestamp,
provider, model, operation, prompt version, input/cached/output/total tokens, estimated cost,
latency, status (`ok | error | budget_exceeded`), cache hit, attempts and error category
(`timeout`, `rate_limit`, `connection`, `server_error`, `auth`, `bad_request`,
`invalid_response`, `budget_exceeded`, `other`). It has no field for prompt, message, output or CV
text.

Sinks (`CompositeUsageSink`; a failing sink never fails the call):

- JSON log line `{"event": "llm_call", ...}` on logger `src.observability`;
- PostgreSQL `cost_log` table (`src/observability/schema.sql`) when `DATABASE_URL` is set;
- in-memory sink used by evaluations to compute tokens, cost, latency percentiles and hit rate.

Other structured log events: `http_request`, `chat_run`, `api_error`, `agent_run_started` /
`agent_run_finished`, `model_turn`, `tool_succeeded` / `tool_rejected` / `tool_failed`,
`mcp_tool_call`, `llm_call_failed` (with `will_retry`), `tracing_failed`, `usage_sink_failed` and
`services_ready` (non-secret config summary). Request bodies, query strings and validation-error input values are never logged.

## Tests

`tests/unit/test_observability.py`, `test_cache.py`, `test_cost_log.py` and `test_config.py` cover
usage extraction, cost calculation, retries and error categories, span hierarchy for both agents
and the API, tracing disabled, Langfuse failure and bounded flush, content-capture opt-in, cache
keys/hits/misses/TTL/versioning, budget enforcement, cost-log persistence (PostgreSQL, opt-in via
`TEST_DATABASE_URL`), and the absence of CV text and secrets in logs. None need a Langfuse server.
