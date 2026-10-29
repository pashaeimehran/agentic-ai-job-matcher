# Limitations

The current boundaries of the system, stated plainly. Planned remedies are in
`docs/future-improvements.md`.

## Security and multi-user

- **No authentication or authorization.** The REST API and MCP server trust every caller. The
  Azure deployment has a public HTTPS endpoint and is suitable only for the synthetic demo CV.
- **Single user, single CV.** The CV is chosen by server configuration (`CV_ID`). There is no
  tenant isolation for CVs, chunks or conversation threads. `saved_jobs.owner_id` exists but is
  always `default`.
- **No rate limiting** and no per-user or per-day spending cap. Cost limits apply per agent run
  only (`docs/cost-controls.md`).
- **No PII controls for real data**: no retention schedule, no export/delete API, no field
  encryption, public (TLS + firewall) database endpoint in Azure. See `docs/data-protection.md`.
- `/chat` thread IDs are bearer-style: anyone who knows a thread ID can continue that conversation
  in the same process.

## State and scaling

- **Conversation checkpoints are in memory.** LangGraph threads disappear on restart or
  scale-to-zero and are not shared between replicas.
- **The response cache is per process** (memory, TTL, LRU), so it is lost on restart and not
  shared.
- For both reasons the API runs one Uvicorn worker, and the Container App is capped at one
  replica. This is not horizontally scalable as is.
- Saved jobs without `DATABASE_URL` go to a JSON file with a process-local lock and atomic
  replace. That is safe for one process, not for several writers.
- The schema is created idempotently at startup with no migration framework. Changing an existing
  column would need one.

## Data and domain

- **Small synthetic job catalog** (16 fixture postings). The agents and API do not search live job
  boards. The optional BA Jobsuche client uses an unofficial backend whose availability and terms
  are not guaranteed, so it is not part of the served application.
- Germany-focused fixtures and requirement vocabulary.
- The match score is a deterministic weighted heuristic over extracted requirements and CV
  evidence. It is advisory, not a hiring probability, and is not calibrated against real outcomes.
- Scanned PDFs need OCR before ingestion. Multi-column PDF reading order can be wrong.
- Fixture mode uses lexical hash embeddings: retrieval there is keyword-like, not semantic.

## Evaluation

- **Small, synthetic evaluation sets**: one synthetic CV, 6 labeled matching jobs (25 status
  checks), 20 retrieval questions, 18 tool-selection cases, 6 grounding scenarios. Perfect scores
  on these show the labeled cases pass, not general accuracy.
- **LLM non-determinism.** Most live numbers are single runs on one model (`gpt-4o-mini`). The
  prompt v1 vs v2 comparison repeats each scenario 5 times per version, and its main effect
  (4/30 vs 1/30 runs with deficit wording) is not statistically significant (Fisher p = 0.35).
- **Ragas scores are LLM-judge estimates** and vary by about ±0.04 between runs. Ragas needs a
  separate Python 3.12 environment because of an OpenAI SDK conflict.
- The grounding detector is lexical and conservative. It can miss paraphrased deficit claims.
- Replay and fixture-mode metrics test orchestration and wiring with scripted model turns, not
  model quality.
- Live evaluations are manual (`live-eval.yml`) and are not run on every change.

## Fixture mode

Fixture mode runs the real API, LangGraph graph, tool registry, retrieval, matching and
persistence, but the model is a rule-based script (`src/llm/fixture_provider.py`). It recognises
a handful of request shapes, such as "find X jobs in Y that fit my CV", a job ID with "save" or
"details", or questions about "my CV". It formats tool results instead of writing prose. It does
not simulate language understanding, ambiguity handling, clarifying questions, model errors or
non-determinism, and its latency and zero cost say nothing about a real model.

## Interfaces

- The MCP server (`scripts/serve_mcp.py`) always uses the offline context: synthetic CV, hash
  embeddings, in-memory CV store, JSON saved jobs. It does not read `APP_MODE` or `DATABASE_URL`.
  It is stdio-only and calls no LLM itself.
- The CLI agent (`scripts/run_agent.py`) always uses live OpenAI. Fixture mode is available
  through the API and the evaluation scripts.

## Deployment

- The Azure deployment runs in fixture mode. A live-mode (OpenAI) run in Azure is a manual
  redeploy (`APP_MODE=live`). With no authentication, a live public endpoint lets anyone spend
  tokens, so it should stay short-lived.
- The free-account subscription expires on 2026-11-28; after that the deployment needs a
  Pay-As-You-Go upgrade or it stops.
- Scale-to-zero means a cold start of about 30 s for the first request after idle.
- Region `northeurope`: free-account subscriptions cannot create Burstable PostgreSQL in
  `germanywestcentral` or `westeurope` (`OfferRestricted`).
- Infrastructure runs (`deploy.sh`, Bicep) re-apply the PostgreSQL server. A regional
  PostgreSQL management-plane incident therefore blocks infrastructure changes, even while the app
  and database keep working (observed on 2026-10-29, Azure incident 6JYK-S00). Image updates use
  `deploy_app.sh`, which only updates the Container App, so they are not affected.
- **Live LLM mode not tested in Azure.** The optional live-mode smoke test was not executed in
  Azure: it was deferred because of the active Azure PostgreSQL management-plane incident and
  because the public API currently has no authentication. The live OpenAI path is covered by the
  local live evaluations.
- `AllowAzureServices` on PostgreSQL admits any Azure-hosted client (password + TLS still
  required).
- Langfuse is self-hosted only in Compose. In Azure, tracing needs an external Langfuse (for
  example Langfuse Cloud EU).
- Model pricing is a dated configuration table and must be reviewed when provider prices change.
