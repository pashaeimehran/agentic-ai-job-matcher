# Demo (3–5 minutes)

A reproducible walkthrough of the full workflow in fixture mode: synthetic CV, synthetic job
catalog, deterministic scripted model, no API key and no paid calls. Every number below is
deterministic and the same locally and in Azure.

## Setup

Pick one target:

```bash
# Local: API + PostgreSQL/pgvector (about 1-2 minutes on the first build)
docker compose up -d --build --wait
BASE=http://127.0.0.1:8000

# Or the Azure deployment (scales to zero: the first request can take about 30 s)
BASE=https://jobmatch-api.blueglacier-8c9485b6.northeurope.azurecontainerapps.io
```

The commands use bash. In Windows PowerShell, set `$BASE = "http://127.0.0.1:8000"` and type
`curl.exe` instead of `curl`; JSON bodies are easiest to send from Git Bash or the OpenAPI UI.
Piping into `python -m json.tool` is optional pretty-printing. The interactive OpenAPI UI is at
`$BASE/docs`.

## 1. Health and readiness

```bash
curl -s $BASE/health | python -m json.tool
curl -s $BASE/ready  | python -m json.tool
```

`/health` shows `"mode": "fixture"`, the agent (`langgraph`) and the five registered tools.
`/ready` checks the database schema (`"schema ready"`), the CV index (13 chunks), the job catalog
(16 jobs) and the chat provider. It never calls an LLM.

## 2. List and search jobs

```bash
curl -s "$BASE/jobs?limit=5" | python -m json.tool
curl -s -X POST $BASE/search -H "Content-Type: application/json" \
     -d '{"target": "jobs", "query": "AI Engineer", "location": "Germany"}' | python -m json.tool
```

Both go through the same `search_jobs` tool that the agent and the MCP server use.

## 3. Retrieve CV evidence

```bash
curl -s -X POST $BASE/search -H "Content-Type: application/json" \
     -d '{"target": "cv", "query": "Python FastAPI PostgreSQL"}' | python -m json.tool
```

Each evidence chunk carries its CV section and the hybrid retrieval signals: `vector_cosine`,
`token_coverage` and the fused rank score (`fusion_score`). In fixture mode the vectors are
lexical hash embeddings; live mode uses OpenAI embeddings in the same pgvector table.

## 4. Analyze one job: deterministic score and requirement statuses

```bash
curl -s -X POST $BASE/analyze-job -H "Content-Type: application/json" \
     -d '{"job_id": "match-05-senior-platform"}' | python -m json.tool
```

What to point out:

- `"score_source": "deterministic"` and `match.score.value` = **41.26**. Repeat the call: the score
  never changes. `match.score.components` shows each weighted term.
- Three requirement statuses, never merged:
  - `supported`: `Python`, with the CV quotation *"Python for data processing and model training."*
    in `comparisons[].evidence`;
  - `unmet`: `German C1`, *"Explicit proficiency below C1"*, backed by the quotation *"English C1 and
    German B1."*: an evidenced gap, not a guess;
  - `unknown`: `Kubernetes` (*"Not evidenced in CV; absence is not proof of inability"*) and
    `5 years` (*"No precise, evidenced employment boundaries"*).

For contrast, `match-01-python-backend` scores **95.11** with all four requirements supported.
Expected output: [screenshot](images/analyze-job-breakdown.png).

## 5. Ask the agent

```bash
curl -s -X POST $BASE/chat -H "Content-Type: application/json" \
     -d '{"message": "Find AI Engineer jobs in Germany that fit my CV"}' | python -m json.tool
```

What to point out:

- `tool_calls`: `search_jobs` → `calculate_match`. The agent chose the tools; the registry validated
  and executed them.
- `answer`: every score in it comes from `calculate_match`, and unknown requirements are labelled
  "no evidence in the CV, not a confirmed gap".
- `ungrounded_scores: []`: the grounding check found no score in the answer that a tool did not
  produce.
- `thread_id`: send it back with a follow-up message to continue the conversation.

In fixture mode the model is a rule-based script, so this shows orchestration and grounding, not
language quality. Live-mode quality is measured separately ([evaluation.md](evaluation.md)).
Expected output: [screenshot](images/chat-grounded-answer.png).

## 6. Azure architecture (one minute)

```
Internet ─HTTPS─► Container App (Consumption, scale 0–1)
                    ├─ image pull: user-assigned managed identity + AcrPull ─► ACR (admin user disabled)
                    ├─ DATABASE_URL secretRef ─► PostgreSQL Flexible Server 16 (B1ms) + pgvector
                    └─ logs ─► Log Analytics
```

- Infrastructure: `infra/azure/main.bicep` through `deploy.sh`.
- Releases: `deploy_app.sh` builds the image, pushes it to ACR with a short-lived token, updates
  only the Container App and waits for a healthy revision. PostgreSQL is not touched. A manual,
  approval-gated GitHub workflow (`deploy.yml`, OIDC login) wraps the same script; its federated
  identity is not provisioned yet.
- Details: [deployment.md](deployment.md). Costs: [cost-controls.md](cost-controls.md).

Clean up locally with `docker compose down` (add `-v` to delete the database volume).

## What to highlight in an interview

1. **The score is code, not the model.** `calculate_match` computes it; the LLM only chooses tools
   and explains. `ungrounded_scores` and the replay evaluations check that the answer never
   invents one.
2. **"Unknown" is not "unmet".** A requirement the CV doesn't mention is reported as unknown, not
   as a gap. Prompt v2 was introduced to reduce deficit wording for unknowns, and the comparison is
   reported with its limits (4/30 → 1/30 runs, not significant).
3. **One tool layer, four entry points.** REST, MCP, the LangGraph agent and the framework-free
   baseline loop all call the same `ToolRegistry`, which owns validation, error codes and retries.
4. **Measured before claimed.** Seven deterministic system gates run in CI (requirement-status
   accuracy, score reproducibility, retrieval Hit@3, replay statuses, orchestration equivalence
   and score consistency, prompt-version labels), plus recorded live runs for tool selection,
   cost, latency and cache effect. The data sets are small and synthetic, and the reports say so.
5. **Fixture mode makes the whole system testable without keys**: the same API, graph, tools,
   retrieval and PostgreSQL path run in CI, in Compose and in Azure with a scripted provider.
6. **Deployment separates infrastructure from releases**: Bicep for infrastructure, an app-only
   path for images, managed-identity image pulls, no registry admin user and no secrets in Git.
