# Cost controls

This page lists the controls that are implemented and where they live. Every limit is an
environment variable (see `.env.example`) read by `src/config.py`. A limit never changes
matching behaviour: scores, requirement statuses and ranking are deterministic Python and are
identical whatever the limits are.

## Where money can be spent

| Operation | Paid in live mode | Paid in fixture mode |
|---|---|---|
| Agent model turns (`/chat`, CLI agent) | yes, OpenAI chat model | no (scripted provider) |
| CV embeddings (ingestion, `search_cv`) | yes, OpenAI embeddings | no (lexical hash embeddings) |
| Job search, job details, `calculate_match`, `/analyze-job`, saving jobs | no | no |
| MCP tool calls | no (the MCP client's own model decides; the server calls no LLM) | no |

The largest lever is architectural: search, matching, scoring and ranking never call an LLM.
The model only chooses tools and writes the explanation.

## Implemented controls

| Control | Variable (default) | Enforced in | Behaviour when reached |
|---|---|---|---|
| Max model turns per run | `AGENT_MAX_MODEL_TURNS` (8) | `src/agent/loop.py`, `src/graph/nodes.py` | run ends with status `max_model_turns_exceeded` |
| Max tool calls per run | `AGENT_MAX_TOOL_CALLS` (12) | `src/agent/loop.py`, `src/graph/nodes.py` | run ends with `max_tool_calls_exceeded`; rejected calls count too |
| Consecutive invalid tool turns (graph only) | `GraphConfig.max_invalid_tool_turns` (2) | `src/graph/nodes.py` | run ends with `invalid_tool_requests` |
| Per-run token budget | `AGENT_MAX_RUN_TOKENS` (60000) | `InstrumentedProvider` before every model call | run ends with `budget_exceeded`; `/chat` returns 422 `agent_limit_exceeded`, no partial answer |
| Per-run cost budget | `AGENT_MAX_RUN_COST_USD` (off) | same | same |
| Max output tokens per model call | `OPENAI_MAX_OUTPUT_TOKENS` (1200) | `OpenAIProvider` (`max_output_tokens`) | the provider truncates the completion |
| Provider retries | `LLM_MAX_RETRIES` (2), `LLM_RETRY_BACKOFF_SECONDS` (1.0) | `InstrumentedProvider` | only timeouts, rate limits, connection and 5xx errors are retried, with exponential backoff; SDK-internal retries are disabled so every retry is counted |
| Request timeout | `OPENAI_TIMEOUT_SECONDS` (60) | OpenAI client | call fails with category `timeout` |
| Response cache | `LLM_CACHE_ENABLED` (true) | `InstrumentedProvider` | an identical request is served from memory at zero cost |
| Model choice | `OPENAI_MODEL` (`gpt-4o-mini`) | `src/llm/factory.py` | one model for the agent; there is no automatic per-request tiering |

Budgets are checked *before* a call: a call already in flight is never cut off, so a run can
exceed its budget by at most one model call. A budget applies to one agent run (one `/chat`
request), not to a user or a day.

## Cost accounting

- `src/observability/pricing.py` holds the only pricing table (USD per million input, cached
  input and output tokens), dated by `PRICING_AS_OF`. Override or extend it with
  `LLM_PRICING_JSON` without a code change.
- Cost is computed in deterministic code from the token usage the provider reports. Unknown
  models get `null`, not a guess. Fixture and hash-embedding calls cost exactly 0.
- Every LLM and embedding call produces one `LLMCallRecord` (tokens, estimated cost, latency,
  status, cache hit, attempts, error category; no prompts or CV text). Records go to the JSON
  log and, with `DATABASE_URL` and `COST_LOG_ENABLED=true`, to the `cost_log` table.

The estimates are only as accurate as the configured prices. Provider prices change. Review
`DEFAULT_PRICING` before relying on the numbers.

Example query:

```sql
SELECT date_trunc('day', created_at) AS day, model, count(*) AS calls,
       sum(total_tokens) AS tokens, sum(estimated_cost_usd) AS usd, avg(cache_hit::int) AS hit_rate
FROM cost_log GROUP BY 1, 2 ORDER BY 1 DESC;
```

## Response cache

Key: SHA-256 over provider, model, model parameters, operation, prompt version, the full message
list (including tool observations), tool schemas and output schema, plus a namespace
(`LLM_CACHE_VERSION`) and an internal key-format version. Any difference is a miss.

- Storage is process memory, bounded (`LLM_CACHE_MAX_ENTRIES`, LRU) with a TTL
  (`LLM_CACHE_TTL_SECONDS`). Entries can contain model output derived from CV text, so they are
  never written to disk or a shared service. Keys are hashes.
- Invalidation: bump `LLM_CACHE_VERSION`, change the prompt version, or restart the process.
- Safe because a model turn is a pure function of its inputs here. Tool execution is never
  cached: `save_job` changes state, and later turns include tool output, so a changed CV or
  catalog changes the key.
- Off in fixture mode (the scripted provider is free; caching would hide the real code path)
  and off in the version comparison, where every run must be an independent model sample.

## Measured effect

From `evaluation/results/system_evaluation_live.json` (18 tool-selection cases, `gpt-4o-mini`,
LangGraph, one cold pass then one warm pass; a single live run per pass):

| Pass | LLM calls | Cache hit rate | Tokens | Estimated cost | End-to-end p50 / p95 |
|---|---:|---:|---:|---:|---:|
| Cold | 34 | 0% | 39,123 | $0.00724 | 2,362 / 6,931 ms |
| Warm | 34 | 91.2% | 3,004 | $0.00048 | 6.5 / 990 ms |

The three warm misses come from `save_job` observations that contain a timestamp, so the
following model turn is (correctly) a different request.

Prompt v2 costs more than v1 (mean $0.000733 vs $0.000632 per query in the live version
comparison) because the model calls `calculate_match` more often. The grounding improvement was
judged worth that cost. See `evaluation/results/version_comparison.md`.

## Cloud infrastructure cost

The Azure deployment (`docs/deployment.md`, `northeurope`) uses the smallest SKUs that support the
architecture. Prices are USD list prices from the Azure Retail Prices API (October 2026) for 730 hours
a month. They change; check the pricing calculator before relying on them.

| Resource | SKU | Rate | Per month | Bills while idle |
|---|---|---|---:|---|
| PostgreSQL Flexible Server compute | Burstable `Standard_B1ms` | $0.018/h | $13.14 | yes, unless the server is stopped |
| PostgreSQL storage | 32 GB, no auto-grow | $0.1265/GB-month | $4.05 | yes, even when stopped |
| PostgreSQL backups | 7 days, LRS | free up to the provisioned size | ~$0 | no |
| Container Registry | Basic (10 GB included) | $0.1666/day | $5.07 | yes |
| Container App | Consumption, 0.5 vCPU / 1 GiB, 0–1 replicas | free grant: 180,000 vCPU-s, 360,000 GiB-s, 2 M requests per month | ~$0 below ~100 active hours | no (scaled to zero) |
| Log Analytics | PerGB2018, 30-day retention | first 5 GB/month free | ~$0 | no |
| Egress | — | first 100 GB/month free | ~$0 | no |
| **Total** | | | **≈ $22/month** | ≈ $22 of it |

The Container Apps price list also has an "Environment Management" meter ($0.13/h). Microsoft's
billing documentation ties it to private endpoints, planned maintenance and Dedicated workload
profiles, none of which this environment uses. Check Cost Management after the first full day to
confirm it stays at zero.

To spend less while not demonstrating:

- Stop PostgreSQL: `az postgres flexible-server stop -g rg-job-match-agent -n <server>`. Compute
  billing stops; storage is still billed. Azure starts it again automatically after 7 days, and
  `/ready` returns 503 while it is stopped. Start it with `az postgres flexible-server start …`.
- The Container App already scales to zero. A cold start takes about 30 s.
- Delete everything: `az group delete --name rg-job-match-agent`.

The current deployment runs on an Azure free account: $200 credit until 2026-11-28 with the
spending limit on. Azure disables the subscription instead of charging the card when the credit or
the trial runs out.

## What is not implemented

- No per-user, per-day or global spending cap, and no rate limiting. The API has no user identity
  (see `docs/limitations.md`).
- No automatic model tiering or fallback model.
- No shared cache across replicas. The Container App is limited to one replica.
- No streaming cut-off when a budget is reached mid-call.
