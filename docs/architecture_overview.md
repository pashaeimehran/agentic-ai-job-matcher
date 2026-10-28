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
`src/llm/provider.py`. `src/llm/openai_provider.py` is the only module that calls the OpenAI SDK.
Graph nodes, routes and the MCP adapter never import it; the API's `build_default_services` and the
CLI scripts construct the provider.

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
