# Agentic AI Job Matcher

[![CI](https://github.com/pashaeimehran/agentic-ai-job-matcher/actions/workflows/ci.yml/badge.svg)](https://github.com/pashaeimehran/agentic-ai-job-matcher/actions/workflows/ci.yml)

An evidence-grounded job-matching agent for the German job market. It extracts structured
requirements from job postings, retrieves supporting passages from a CV, scores fit with
deterministic code, and lets an LLM agent search, compare and explain matches using validated
tools. It runs behind a REST API and an MCP server, and is measured, traced, costed,
containerized and ready to deploy to Azure.

**Python · LangGraph · FastAPI · MCP · PostgreSQL/pgvector · Langfuse · Docker · Azure Container Apps**

## The problem

Job postings and CVs describe the same things differently. A useful match tells a candidate
which requirements their CV *shows* they meet, which it shows they *don't* meet, and which it
simply doesn't mention. It must never invent evidence or a score. So this project keeps the
judgement in deterministic, testable code and uses the LLM only to choose tools and explain the
results, with every claim traceable to a CV quotation.

## Architecture

```
             REST client                    MCP client (its own LLM)
                  │                                  │
        FastAPI  (src/api)                 MCP server (src/mcp_server)
                  │  /chat                           │  tools/call
         LangGraph agent (src/graph)                 │
   prepare → agent ⇄ execute_tools → finalize        │
                  │                                  │
                  └──────────► ToolRegistry ◄────────┘   validation, retries, error contract
                               (src/tools)
        search_jobs · get_job_details · search_cv · calculate_match · save_job
                  │                 │                   │
            job catalog      CV retrieval (src/rag)   deterministic matching (src/matching)
                                    │
                       PostgreSQL + pgvector  ── cv_documents · cv_chunks · saved_jobs · cost_log

 Cross-cutting (src/observability, src/llm):
   InstrumentedProvider → retries · per-run budget · response cache · usage + cost · Langfuse spans
   OpenAI provider (live)  |  deterministic fixture provider (zero-key demo)
```

The REST API, MCP server, LangGraph agent and the framework-free baseline agent
(`src/agent/loop.py`) all call the same `ToolRegistry`. None of them re-implements search,
matching or scoring. More detail: [architecture overview](docs/architecture_overview.md).

## Technologies

| Technology | Role |
|---|---|
| Python 3.14, Pydantic | application code, schema validation for every LLM output and tool argument |
| OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) | tool calling, structured extraction, embeddings (live mode) |
| LangGraph | agent orchestration: explicit state, routing, limits, per-thread checkpoints |
| FastAPI, Uvicorn | typed REST API with health and readiness probes |
| MCP Python SDK | exposes the five tools to external LLM clients |
| PostgreSQL 16 + pgvector | CV chunks and vectors, saved jobs, cost log |
| Langfuse (self-hosted) | traces of requests, graph nodes, tool calls and model calls |
| Ragas | LLM-judge faithfulness / relevancy (isolated environment) |
| Docker, Docker Compose | image and local stack (API, PostgreSQL, Langfuse) |
| GitHub Actions, Bicep, Azure CLI | CI, Azure Container Apps + PostgreSQL Flexible Server deployment |
| pytest | 306 offline tests plus 2 opt-in PostgreSQL integration tests |

## Quick start: fixture mode, no API key

Fixture mode runs the real API, LangGraph agent, tools, retrieval, matching and PostgreSQL
persistence with a deterministic scripted model in place of OpenAI. It makes no paid calls and
needs no credentials. [What it does and doesn't simulate](docs/limitations.md#fixture-mode).

### With Docker (recommended)

Requires Docker with Compose v2.

```bash
git clone https://github.com/pashaeimehran/agentic-ai-job-matcher.git
cd agentic-ai-job-matcher
docker compose up -d --build --wait        # API + PostgreSQL/pgvector, about 1-2 minutes on first build
```

Open <http://127.0.0.1:8000/docs> for the interactive API, or (on Windows PowerShell, type
`curl.exe` instead of `curl`):

```bash
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/analyze-job -H "Content-Type: application/json" \
     -d '{"job_id": "match-01-python-backend"}'
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "Find AI Engineer jobs in Germany that fit my CV"}'
```

`/analyze-job` returns a deterministic score (95.11 for this job) with supported, unmet and
unknown requirements. `/chat` runs the agent: `search_jobs` → `calculate_match` → answer.

Add tracing with a self-hosted Langfuse at <http://localhost:3000> (login `admin@example.com` /
`local-admin-password`, local only):

```bash
TRACING_ENABLED=true docker compose --profile observability up -d --build --wait
```

Stop with `docker compose down` (add `-v` to delete the database and trace volumes).

### Without Docker

Requires Python 3.14. Without `DATABASE_URL`, CV chunks stay in memory and saved jobs go to
`data/private/saved_jobs.json`.

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m scripts.serve_api                # http://127.0.0.1:8000/docs
```

## Live mode

Copy `.env.example` to `.env`, set `OPENAI_API_KEY` and `APP_MODE=live` (or leave `APP_MODE`
unset: it becomes `live` when a key is present), then start either way. Live mode uses OpenAI
for the agent and for CV embeddings, so these calls are paid. Every variable is documented in
[`.env.example`](.env.example). The CLI covers CV ingestion and Q&A (`python -m scripts.cv
--help`) and running the agents directly (`python -m scripts.run_agent --help`). See the
[retrieval setup guide](docs/apply_cv_retrieval.md).

## Tests

```bash
python -m pytest                                         # offline: no key, no network, no database
TEST_DATABASE_URL=postgresql://user:pass@127.0.0.1:5432/db python -m pytest   # + PostgreSQL integration
```

The tests cover tools, matching, retrieval, both agents, the API, the MCP server, configuration,
fixture mode, observability, cache, cost accounting and log hygiene. On Windows, use `127.0.0.1`
rather than `localhost` in connection URLs: `localhost` tries IPv6 first, and Compose publishes on
IPv4 only.

## Evaluations

```bash
python -m scripts.evaluate_system                        # all deterministic evaluations + gates (CI)
python -m scripts.compare_versions --mode offline        # version-comparison harness check
python -m scripts.evaluate_system --live                 # paid: live tool selection, cost, latency, cache
python -m scripts.compare_versions --mode live --repeats 5   # paid: prompt v1 vs v2
```

The dedicated evaluators (`evaluate_cv_retrieval`, `evaluate_job_matching`,
`evaluate_tool_selection`, `compare_agents`, `evaluate_job_extraction`, `evaluate_matching_ragas`)
still run on their own. Ragas runs in a separate environment ([how](docs/apply_job_cv_matching.md)).
Methodology and every metric: [docs/evaluation.md](docs/evaluation.md).

## Results

Synthetic data throughout. Deterministic results are reproducible. Live results are single runs
unless marked, and Ragas is an LLM judge.

| Area | Result | Kind |
|---|---|---|
| Requirement-status accuracy | 100% (25 labeled checks, 6 jobs); scores reproducible | deterministic |
| CV retrieval Hit@3 / MRR | 95% / 0.867 offline (lexical); 100% / 0.975 live (semantic) | deterministic / single run |
| Matching retrieval @3, vector → hybrid | recall 83% → 100%, precision 28% → 33% | deterministic |
| Tool selection, 18 cases | first-tool accuracy 100%, manual loop and LangGraph, live | single run |
| Manual loop vs LangGraph, 29 scenarios | 28 identical, 1 documented guard difference, identical scores | deterministic |
| Prompt v1 → v2 (grounding of "unknown") | task success 86.7% → 96.7%; runs with deficit wording 4/30 → 1/30 (p = 0.35, not significant) | 5 repeats × 6 scenarios × 2 |
| Cost per query, live agent | $0.000402 cold; $0.000027 with a warm cache (91% hit rate) | single run |
| Latency, live agent | p50 2.4 s / p95 6.9 s cold; p50 6.5 ms warm | single run |
| Ragas faithfulness / answer relevancy | 0.68–0.74 / 0.67–0.70 over three runs | LLM judge |
| Job extraction | 40/40 fixture variants exact | single run |

Full version comparison: [evaluation/results/version_comparison.md](evaluation/results/version_comparison.md).

## Tool calling vs LangGraph

Both agents use the same tools, provider interface and limits. The **manual loop**
(`src/agent/loop.py`) is a plain bounded `while` loop: the model requests tools, Python validates
and executes them, and failures are classified and retried only when retryable. The **LangGraph
agent** (`src/graph/`) expresses the same logic as explicit state and small nodes
(`prepare → agent ⇄ execute_tools → finalize | fail`) with pure routing functions, per-thread
checkpointing and one extra guard that stops repeated invalid tool requests. A replay harness
shows they behave identically on 28 of 29 scenarios. The LangGraph version is the one served by
the API. Why: [docs/why-langgraph.md](docs/why-langgraph.md). Tool contracts:
[docs/tool_calling_agent_architecture.md](docs/tool_calling_agent_architecture.md).

## MCP

`python -m scripts.serve_mcp` serves `search_jobs`, `get_job_details`, `search_cv`,
`calculate_match` and `save_job` over stdio. Schemas come from the shared registry, which also
remains the validation authority. Read-only and idempotent hints are set per tool. The server calls
no LLM: the MCP client's model decides. [docs/mcp.md](docs/mcp.md).

## Observability and cost controls

- Langfuse traces: `POST /chat → agent.langgraph → graph.* → llm.complete | tool.*`, carrying
  model, tokens, cost, cache hit, retries and run IDs. Prompts and CV text are excluded unless
  content capture is explicitly enabled. Tracing is optional: if Langfuse is down, requests still
  succeed and shutdown stays bounded.
- One structured record per LLM call (tokens, cost, latency, status, cache hit, error category)
  goes to JSON logs and the PostgreSQL `cost_log` table. Prices live in a single dated table.
- Cost controls: max model turns and tool calls, per-run token/cost budget, output-token cap,
  bounded retries, and an in-memory response cache keyed by everything that determines the
  answer, with explicit versioning.

[docs/observability.md](docs/observability.md) · [docs/cost-controls.md](docs/cost-controls.md)

## Deployment

One image (multi-stage, non-root, no secrets) runs locally with Compose and on **Azure Container
Apps** with **Azure Database for PostgreSQL Flexible Server** (pgvector), provisioned by
`infra/azure/main.bicep` and `infra/azure/deploy.sh`. CI (`.github/workflows/ci.yml`) runs tests,
deterministic evaluation gates, the image build and container and Compose smoke tests on every pull
request without secrets. The image is built with Docker from the working tree and pushed to ACR
with a short-lived token (no ACR Tasks, no registry admin user). The app pulls it with a managed
identity. Deployment (`deploy.yml`) is manual and approval-gated, with OIDC login.

**Live demo (fixture mode, synthetic data):**
<https://jobmatch-api.blueglacier-8c9485b6.northeurope.azurecontainerapps.io/docs>. It runs in
`northeurope` and scales to zero, so the first request after idle takes about 30 s. Verified there:
health and readiness, pgvector, deterministic `/analyze-job` (95.11), and `/chat` running
`search_jobs → calculate_match`. Bicep (`infra/azure/deploy.sh`) creates and changes the
infrastructure. Image updates (`infra/azure/deploy_app.sh`, the `deploy.yml` workflow) only update
the Container App. A live OpenAI run in Azure was deliberately not executed, because the public API
has no authentication. Details: [docs/deployment.md](docs/deployment.md).

## Limitations

No authentication or multi-user isolation. Conversation memory and the cache are in-process, so
the API runs as one replica. The job catalog is a small synthetic fixture set. Evaluation sets are
small and synthetic, and most live results are single runs. The match score is an advisory
heuristic, not a hiring decision. Full list: [docs/limitations.md](docs/limitations.md). Personal
data handling and the EU AI Act: [docs/data-protection.md](docs/data-protection.md). What comes
next: [docs/future-improvements.md](docs/future-improvements.md).

## Repository guide

| Path | Purpose |
|---|---|
| `src/extraction/` | job-text cleaning and structured extraction |
| `src/rag/` | CV parsing, chunking, embeddings, pgvector store, retrieval, grounded Q&A |
| `src/matching/` | deterministic, evidence-aware job–CV matching and ranking |
| `src/tools/` | tool schemas, registry, implementations, saved-job stores |
| `src/agent/` | framework-free agent loop, versioned prompts, grounding check |
| `src/graph/` | LangGraph agent: state, nodes, routing, checkpointing |
| `src/api/`, `src/mcp_server/` | REST and MCP adapters |
| `src/llm/` | provider interface, OpenAI provider, fixture provider, factory |
| `src/observability/` | tracing, metrics, pricing, cache, run context, cost log |
| `src/config.py`, `src/db.py` | environment configuration, schema initialization |
| `scripts/` | CLIs, servers and evaluators |
| `data/fixtures/` | synthetic CV, jobs and labeled evaluation cases |
| `evaluation/results/`, `benchmarks/` | recorded reports |
| `infra/azure/`, `.github/workflows/` | deployment and CI |
| `docs/` | architecture, API, MCP, evaluation, operations and governance notes |

## License

[MIT License](LICENSE). Copyright © 2026 Mehran Pashaei.
