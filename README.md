# Agentic AI Job Matcher

[![CI](https://github.com/pashaeimehran/agentic-ai-job-matcher/actions/workflows/ci.yml/badge.svg)](https://github.com/pashaeimehran/agentic-ai-job-matcher/actions/workflows/ci.yml)

An evidence-grounded job-matching agent for the German job market. It extracts structured
requirements from job postings, retrieves supporting passages from a CV, scores fit with
deterministic code, and lets an LLM agent search, compare and explain matches through validated
tools. It is served through a REST API and an MCP server, evaluated with deterministic gates and
recorded live runs, and deployed on Azure Container Apps with PostgreSQL/pgvector.

**Python · LangGraph · FastAPI · MCP · PostgreSQL/pgvector · Langfuse · Docker · Bicep · Azure Container Apps**

## The problem

Job postings and CVs describe the same things differently. A useful match tells a candidate
which requirements their CV *shows* they meet, which it shows they *don't* meet, and which it
simply doesn't mention. It must never invent evidence or a score. So match judgements and scores
stay in deterministic, testable code, and LLMs are used for structured extraction, tool selection
and explanation. Evidence-backed assessments cite the CV quotation they rest on; without evidence,
a requirement stays explicitly `unknown`.

## Key capabilities

- **Structured job extraction**: postings become validated `JobPosting` objects (skills,
  languages, experience).
- **CV retrieval**: CV ingestion, chunking and embeddings in PostgreSQL/pgvector, with hybrid
  vector + keyword search and quotation-only Q&A that abstains when evidence is missing.
- **Evidence-grounded matching**: each requirement is `supported` or `unmet` with the CV quotation
  behind it, or explicitly `unknown` when the CV has no evidence, with a deterministic,
  reproducible 0–100 compatibility score.
- **Agentic tool use**: a LangGraph agent chooses among five validated tools (`search_jobs`,
  `get_job_details`, `search_cv`, `calculate_match`, `save_job`) and explains the results.
- **Interfaces**: FastAPI (`/jobs`, `/search`, `/analyze-job`, `/chat`, health and readiness) and
  an MCP server exposing the same tools.
- **Operations**: response cache, per-run token and cost budgets, cost log, Langfuse tracing,
  Docker, CI, and Azure infrastructure in Bicep.

## Architecture

```mermaid
flowchart TD
    CLIENT["REST client"] --> API["FastAPI · src/api"]
    MCPC["MCP client (its own LLM)"] --> MCP["MCP server · src/mcp_server"]
    API -- "/chat" --> GRAPH["LangGraph agent · src/graph<br/>prepare → agent ⇄ execute_tools → finalize"]
    LOOP["Manual tool-calling loop · src/agent<br/>(baseline)"]
    API -- "/jobs, /search, /analyze-job" --> REG
    MCP -- "tools/call" --> REG
    GRAPH --> REG["ToolRegistry · src/tools<br/>validation · error contract · bounded retries"]
    LOOP --> REG
    REG --> JOBS["search_jobs · get_job_details<br/>job catalog"]
    REG --> CV["search_cv<br/>hybrid vector + keyword retrieval"]
    REG --> MATCH["calculate_match<br/>deterministic scoring · src/matching"]
    REG --> SAVE["save_job"]
    MATCH --> CV
    CV --> PG[("PostgreSQL + pgvector<br/>cv_chunks · saved_jobs · cost_log")]
    SAVE --> PG

    subgraph XC["Cross-cutting · src/llm, src/observability"]
        INST["InstrumentedProvider<br/>retries · run budget · response cache"]
        PROV["LLM provider<br/>OpenAI (live) | scripted fixture"]
        OBS["usage and cost records<br/>Langfuse tracing"]
    end
    GRAPH -.-> INST
    LOOP -.-> INST
    INST -.-> PROV
    INST -.-> OBS
    OBS -.-> PG
```

The REST API, MCP server, LangGraph agent and the framework-free baseline loop all call the same
`ToolRegistry`; none of them re-implements search, matching or scoring. More detail:
[architecture overview](docs/architecture.md).

## Key engineering decisions

- **Deterministic score, LLM explanation.** `calculate_match` computes the score in Python from
  extracted requirements and CV evidence. The model chooses tools and writes the explanation; a
  grounding check flags any score in the answer that no tool produced.
- **Evidence-grounded requirement statuses.** `supported` needs a CV quotation, `unmet` needs
  contradicting evidence (for example German B1 against a C1 requirement), and a requirement
  without evidence stays `unknown`, with no quotation. Absence of evidence is not treated as proof
  of inability, so unknowns are never reported as gaps.
- **Hybrid retrieval.** pgvector similarity fused with keyword retrieval (reciprocal-rank fusion
  plus a token-coverage rerank). On the labeled matching set, recall@3 rises from 83% to 100%
  over vector-only.
- **One shared tool registry.** Schemas, validation, error codes and retries live in one place and
  serve REST, MCP and both agents. Model output is treated as untrusted input.
- **Manual loop kept as a baseline.** The framework-free loop is the reference implementation. A
  replay harness compares it with the LangGraph agent: identical on 28 of 29 scenarios, with one
  documented difference from an extra graph guard.
- **LangGraph for the served agent.** Explicit state, pure routing functions, hard limits and
  per-thread checkpoints ([why LangGraph](docs/why-langgraph.md)).
- **Fixture mode.** A scripted provider and hash embeddings run the real API, graph, tools and
  PostgreSQL path without a key, so CI, Compose and the Azure demo are reproducible and free.
- **Infrastructure separate from releases.** Bicep (`deploy.sh`) creates and changes resources;
  `deploy_app.sh` ships a new image to the Container App without touching PostgreSQL.

## Demo

Real responses from the API in fixture mode (synthetic CV and jobs, scripted model), captured in
its built-in Swagger UI.

### Evidence-grounded job analysis

![POST /analyze-job for match-05-senior-platform: deterministic score 41.26; Python supported, German C1 unmet, Kubernetes and 5 years unknown, with CV quotations](docs/images/analyze-job-breakdown.png)

Excerpts of the `POST /analyze-job` response for `match-05-senior-platform`. The request ID and the
score components are omitted; the fields shown are unchanged. The deterministic matcher separates
requirements into supported, unmet and unknown and grounds each decision in CV evidence: Python is
supported by a quotation, and German C1 is unmet because the CV states B1. Kubernetes and 5 years
have no evidence either way, so they stay unknown rather than counting as failures.

### Grounded agent execution

![POST /chat: the agent calls search_jobs then calculate_match, the run completes, and no ungrounded scores are reported](docs/images/chat-grounded-answer.png)

The request and the full `POST /chat` response (the Request URL panel is omitted). The LangGraph
agent selects validated tools, here `search_jobs` and then `calculate_match`. The match scores in
the answer come from deterministic code, and the grounding check (`ungrounded_scores: []`) confirms
that no unsupported score reached the response. Fixture-mode model turns are scripted, so this
shows orchestration and grounding, not language quality.

### Run it

Locally with Docker (fixture mode, no API key):

```bash
git clone https://github.com/pashaeimehran/agentic-ai-job-matcher.git
cd agentic-ai-job-matcher
docker compose up -d --build --wait        # API + PostgreSQL/pgvector, 1-2 minutes on first build

curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/analyze-job -H "Content-Type: application/json" \
     -d '{"job_id": "match-05-senior-platform"}'
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "Find AI Engineer jobs in Germany that fit my CV"}'
```

The interactive API is at <http://127.0.0.1:8000/docs>; on Windows PowerShell, type `curl.exe`
instead of `curl`.

The same API runs in Azure in fixture mode:
<https://jobmatch-api.blueglacier-8c9485b6.northeurope.azurecontainerapps.io/docs>. It scales to
zero, so the first request after idle takes about 25–30 s. A guided 3–5 minute walkthrough is in
[docs/demo.md](docs/demo.md).

## Results

All data is synthetic: one CV, 16 fixture jobs, 6 labeled matching jobs, 20 retrieval questions
and 18 tool-selection cases. Perfect scores mean the labeled cases pass, not general accuracy.
Live results are single runs unless marked, and Ragas is an LLM judge.

| Area | Result | Kind |
|---|---|---|
| Requirement-status accuracy | 100% (25 labeled checks, 6 jobs); scores reproducible | deterministic |
| CV retrieval Hit@3 / MRR | 95% / 0.867 offline (lexical); 100% / 0.975 live (semantic) | deterministic / single run |
| Matching retrieval @3, vector → hybrid | recall 83% → 100%, precision 28% → 33% | deterministic |
| Tool selection, 18 cases | first-tool accuracy 100% for the manual loop and LangGraph | single live run each |
| Manual loop vs LangGraph, 29 scenarios | 28 identical, 1 documented guard difference, identical scores | deterministic |
| Prompt v1 → v2 (wording of "unknown") | task success 86.7% → 96.7%; runs with deficit wording 4/30 → 1/30 (Fisher p = 0.35, not significant) | 5 repeats × 6 scenarios × 2 versions |
| Cost per query, live agent | $0.000402 cold; $0.000027 with a warm cache (91% hit rate) | single run |
| Latency, live agent | p50 2.4 s / p95 6.9 s cold; p50 6.5 ms warm | single run |
| Ragas faithfulness / answer relevancy | 0.703 / 0.698 and 0.744 / 0.672 in two recorded runs | LLM judge |
| Job extraction | 40/40 fixture variants exact | single run |
| System evaluation gates | 7 / 7 pass, in CI on every push and pull request | deterministic |
| Test suite | 308 passed with PostgreSQL; 306 passed + 2 skipped without it | deterministic |
| Azure smoke test (fixture mode) | all endpoints 200; `/analyze-job` 95.11 for `match-01-python-backend`; `/chat` `search_jobs → calculate_match`; warm latency 0.13–0.99 s | deterministic |

Methodology and every metric: [docs/evaluation.md](docs/evaluation.md). Prompt comparison:
[evaluation/results/version_comparison.md](evaluation/results/version_comparison.md).

![GitHub Actions: both CI jobs green, with test, matching, retrieval, replay, equivalence and system-gate steps passing](docs/images/ci-green.png)

CI on every push and pull request: tests with PostgreSQL, matching and retrieval evaluation,
tool-selection replay, manual-loop vs LangGraph equivalence and the system evaluation gates, plus
a second job for the container build and fixture-mode smoke tests.

## Interfaces

- **REST API**: `/health`, `/ready`, `/jobs`, `/jobs/{job_id}`, `/search` (jobs or CV),
  `/analyze-job` and `/chat` (LangGraph agent with conversation threads).
  [docs/api.md](docs/api.md)
- **MCP server**: `python -m scripts.serve_mcp` serves the five tools over stdio. Schemas come from
  the shared registry, which remains the validation authority; the server calls no LLM, the MCP
  client's model decides. [docs/mcp.md](docs/mcp.md)
- **CLI**: CV ingestion and quotation-only Q&A (`python -m scripts.cv --help`), matching
  (`python -m scripts.match_jobs --help`) and either agent (`python -m scripts.run_agent --help`).

How the two agents differ: [tool-calling agent](docs/tool-calling-agent.md) and
[why LangGraph](docs/why-langgraph.md).

## Local setup

### Fixture mode (no key)

With Docker, `docker compose up -d --build --wait` as in [Demo](#demo); stop with
`docker compose down` (`-v` also deletes the volumes). Without Docker (Python 3.14; CV chunks stay
in memory and saved jobs go to `data/private/saved_jobs.json`):

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m scripts.serve_api                # http://127.0.0.1:8000/docs
```

[What fixture mode does and doesn't simulate](docs/limitations.md#fixture-mode).

### Live mode (paid OpenAI calls)

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (`APP_MODE` becomes `live` when a key is
present). Live mode uses OpenAI for the agent and the CV embeddings. Every variable is documented
in [`.env.example`](.env.example). Guides: [CV retrieval](docs/cv-retrieval-runbook.md) and
[matching](docs/matching-runbook.md).

### Tracing

`TRACING_ENABLED=true docker compose --profile observability up -d --build --wait` adds a
self-hosted Langfuse at <http://localhost:3000>, using the local-only defaults in `compose.yaml`.
Prompts and CV text are excluded from traces unless content capture is enabled, and requests still
succeed when Langfuse is down. [docs/observability.md](docs/observability.md) ·
[docs/cost-controls.md](docs/cost-controls.md)

### Tests and evaluations

```bash
python -m pytest                                          # offline: no key, network or database
TEST_DATABASE_URL=postgresql://user:pass@127.0.0.1:5432/db python -m pytest   # + PostgreSQL tests
python -m scripts.evaluate_system                         # deterministic evaluations + gates (CI)
python -m scripts.evaluate_system --live                  # paid: live tool selection, cost, cache
python -m scripts.compare_versions --mode live --repeats 5    # paid: prompt v1 vs v2
```

On Windows, use `127.0.0.1` rather than `localhost` in connection URLs: `localhost` tries IPv6
first, and Compose publishes on IPv4 only.

## Azure deployment

One image (multi-stage, non-root, no secrets) runs locally and on **Azure Container Apps**
(Consumption, scale 0–1) with **Azure Database for PostgreSQL Flexible Server 16** and pgvector.
The image lives in **Azure Container Registry**, and the app pulls it with a user-assigned
**managed identity** (`AcrPull`); the registry admin user is disabled and the only app secret is
the database URL.

| Path | Tool | Changes |
|---|---|---|
| Infrastructure | `infra/azure/main.bicep` via `deploy.sh` | resource group, registry, identity, Log Analytics, Container Apps environment, PostgreSQL, app |
| Application release | `infra/azure/deploy_app.sh` | Docker build → push to ACR with a short-lived token → new Container App revision → health wait → smoke test; PostgreSQL untouched |

The running environment was deployed from a workstation and verified with read-only Portal checks
(revision, identity, registry, secrets, database, logs) and public smoke tests. CI (`ci.yml`) runs on every push and pull
request without secrets: tests against a pgvector service, the evaluation gates, the image build,
and container and Compose smoke tests. `deploy.yml` wraps `deploy_app.sh` behind a manual,
approval-gated trigger with GitHub OIDC; its Azure federated identity is not provisioned yet, so
that workflow has not been run. Live OpenAI mode was intentionally not enabled in Azure, because
the public API has no authentication or rate limiting. Details:
[docs/deployment.md](docs/deployment.md).

## Repository structure

| Path | Purpose |
|---|---|
| `src/extraction/` | job-text cleaning and structured extraction |
| `src/rag/` | CV parsing, chunking, embeddings, pgvector store, grounded Q&A |
| `src/matching/` | requirement comparison, hybrid retrieval, deterministic scoring and ranking |
| `src/tools/` | tool schemas, registry, implementations, saved-job stores |
| `src/agent/` | framework-free agent loop, versioned prompts, grounding check |
| `src/graph/` | LangGraph agent: state, nodes, routing, checkpointing |
| `src/api/`, `src/mcp_server/` | REST and MCP adapters |
| `src/llm/` | provider interface, OpenAI provider, fixture provider, factory |
| `src/observability/` | tracing, metrics, pricing, cache, run context, cost log |
| `src/config.py`, `src/db.py` | environment configuration, schema initialization |
| `scripts/` | CLIs, servers and evaluators |
| `data/fixtures/` | synthetic CV, jobs and labeled evaluation cases |
| `evaluation/results/`, `benchmarks/` | recorded evaluation reports |
| `infra/azure/`, `.github/workflows/` | deployment and CI |
| `docs/` | architecture, interfaces, evaluation, operations and data protection |

## Limitations

No authentication, rate limiting or multi-user isolation, so the public demo serves synthetic data
in fixture mode only. Conversation memory and the response cache are in-process, so the API runs
as one replica. The job catalog is a small synthetic fixture set. Evaluation sets are small and
synthetic, and most live results are single runs. The score is an advisory heuristic, not a
hiring decision. Full list: [docs/limitations.md](docs/limitations.md). Personal data and the EU AI
Act: [docs/data-protection.md](docs/data-protection.md).

## Future improvements

Authentication and per-user isolation, durable PostgreSQL-backed conversation state, private
networking, larger and unseen evaluation sets, and a provisioned OIDC deployment identity. Ordered
plan: [docs/future-improvements.md](docs/future-improvements.md).

## License

[MIT License](LICENSE). Copyright © 2026 Mehran Pashaei.
