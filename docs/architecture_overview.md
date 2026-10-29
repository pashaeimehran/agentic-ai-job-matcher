# Architecture Overview

One tool layer, four entry points.

```
                    ┌── Manual agent loop   src/agent/        (CLI baseline: scripts.run_agent)
                    │
User / Client ──────┼── LangGraph agent     src/graph/        (scripts.run_agent --agent langgraph, POST /chat)
                    │
                    ├── FastAPI             src/api/          (scripts.serve_api)
                    │
                    └── MCP server          src/mcp_server/   (scripts.serve_mcp)
                             │
                             ▼
            Shared tool layer  src/tools/   ToolRegistry.execute(): validate → run → classify → bounded retry
                             │
              ┌──────────────┼───────────────────┬──────────────────┐
              ▼              ▼                   ▼                  ▼
        Job catalog      CV / RAG           Matching           Saved jobs
        src/tools/jobs   src/rag,           src/matching       src/tools/saved_jobs
        (fixtures)       matching.retrieval      │
                                                 ▼
                                   deterministic scoring (score_match, rank_analyses)
```

LLM access goes through `src/llm/tool_calling.py` (provider-neutral types) and
`src/llm/provider.py`. `src/llm/openai_provider.py` adapts the OpenAI SDK; `src/llm/fixture_provider.py`
is the deterministic zero-key stand-in. `src/llm/factory.py` builds one of them from `Settings` and
wraps it in `InstrumentedProvider`. Graph nodes, routes and the MCP adapter never import a provider;
the API's `build_default_services` and the CLI scripts construct it.

## Cross-cutting layers

| Layer | Module | Applied at |
|---|---|---|
| Configuration | `src/config.py` (`Settings.from_env`, secrets as `SecretStr`) | API startup, CLI and evaluation scripts |
| Schema initialization | `src/db.py` (idempotent SQL files) | API startup when `DATABASE_URL` is set |
| Run scope (request/run ID, budget) | `src/observability/context.py` | API middleware, agent `run()` |
| Tracing (Langfuse or no-op) | `src/observability/tracing.py` | request, agent run, graph node, tool, model call |
| Retries, budget, cache, usage and cost | `src/observability/instrumented.py`, `cache.py`, `pricing.py` | every LLM and embedding call |
| Operational records / cost log | `src/observability/metrics.py`, `postgres.py` | JSON logs, `cost_log` table, evaluation sinks |

Business modules (`src/tools`, `src/matching`, `src/rag`) contain no tracing, metrics or cache
code. See [observability.md](observability.md) and [cost-controls.md](cost-controls.md).

## Single-source rules

| Concern | Implemented once in | Used by |
|---|---|---|
| Tool argument schemas and validation | `src/tools/schemas.py`, `ToolRegistry.validate` | both agents, API (`/search` request models subclass them), MCP (`tools/list`) |
| Tool execution, error codes, retries | `ToolRegistry.execute`, `src/tools/errors.py` | all four entry points |
| Match score | `src/matching/scoring.py` | `calculate_match` only |
| Score grounding check | `src/agent/loop.py::ungrounded_scores` | manual loop, LangGraph `finalize` |
| CV scope | `ToolContext.cv_id` | never a request or tool argument |

Verified by tests: a spy registry records that the API and MCP calls reach `ToolRegistry.execute`.
Graph, API and MCP tests compare `calculate_match` scores with a direct `analyze_jobs()` computation.

## Runtime and deployment

| Environment | How it runs | Data store |
|---|---|---|
| Local, no Docker | `python -m scripts.serve_api` (fixture mode without a key) | in-memory CV store, JSON saved jobs |
| Docker Compose | `api` + `pgvector/pgvector:pg16`, optional self-hosted Langfuse profile | PostgreSQL + pgvector |
| Azure (deployed) | Container App `jobmatch-api` (Consumption, 0–1 replicas, HTTPS), image from ACR pulled with a managed identity | Azure Database for PostgreSQL Flexible Server 16 (B1ms) with pgvector |

The same image and the same `Settings` drive all three; only environment variables differ.
Infrastructure is defined in `infra/azure/main.bicep` (`deploy.sh`). Application images go through
Docker → ACR (short-lived token) → `az containerapp update` (`deploy_app.sh`, `deploy.yml`), which
never touches PostgreSQL. See [deployment.md](deployment.md).

## Entry points

| Entry point | Who chooses the tools | LLM | Docs |
|---|---|---|---|
| Manual loop | the model, in a `for` loop | yes | [tool_calling_agent_architecture.md](tool_calling_agent_architecture.md) |
| LangGraph | the model, in an explicit state graph | yes | [why-langgraph.md](why-langgraph.md) |
| REST `/search`, `/analyze-job`, `/jobs` | the HTTP client (fixed per endpoint) | no | [api.md](api.md) |
| REST `/chat` | the model via LangGraph | yes | [api.md](api.md) |
| MCP | the external client's model | not in-process | [mcp.md](mcp.md) |

## Evaluation and comparison

| Command | What it measures | Deterministic |
|---|---|---|
| `python -m pytest` | unit, graph, API and MCP tests | yes |
| `python -m scripts.evaluate_tool_selection --mode replay [--agent langgraph]` | dataset and harness, through either agent | yes |
| `python -m scripts.compare_agents` | manual vs LangGraph orchestration equivalence (29 scenarios) | yes |
| `python -m scripts.evaluate_job_matching` | matching status accuracy, retrieval hit rate | yes |
| `python -m scripts.evaluate_cv_retrieval` | offline lexical retrieval | yes |
| `python -m scripts.evaluate_tool_selection --mode live [--agent langgraph]` | model tool selection | no (paid) |
| `python -m scripts.compare_versions --mode offline` | prompt v1 vs v2 harness: labels and score determinism | yes |
| `python -m scripts.compare_versions --mode live --repeats 5` | prompt v1 vs v2 grounding, tokens, cost, latency | no (paid) |
| `python -m scripts.evaluate_system [--live]` | all of the above in one report, with gates; `--live` adds cost, latency and cache measurements | offline: yes |

Details and recorded results: [evaluation.md](evaluation.md).
