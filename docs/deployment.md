# Deployment

Two targets share one image: Docker Compose for local use and Azure Container Apps with Azure
Database for PostgreSQL Flexible Server (pgvector) for the cloud. Configuration is environment
variables only (`.env.example`, `src/config.py`). Nothing secret is baked into the image.

## Container image

`Dockerfile`, multi-stage:

- builder: installs runtime requirements (`requirements-retrieval.txt`) into a virtualenv,
  pinned by `constraints.txt` (every transitive version);
- runtime: `python:3.14-slim`, only `src/`, the API/MCP/CV scripts and `data/fixtures/`
  (synthetic data), non-root user `app` (UID 10001), `HEALTHCHECK` on `/health`,
  `APP_MODE=fixture` by default;
- `.dockerignore` excludes `.env*`, `data/private`, `venv`, tests, docs and Git metadata.

```bash
docker build -t job-match-agent .
docker run --rm -p 8000:8000 job-match-agent      # fixture mode, no keys, in-memory storage
```

Validated locally (2026-10-29): clean `--no-cache` build in 44 s, image 422 MB, healthy, runs as
UID 10001, and `infra/azure/smoke_test.sh --agentic` passes with no credentials.

## Docker Compose

| Service | Profile | Purpose |
|---|---|---|
| `api` | default | FastAPI app; waits for a healthy database; health check on `/ready` |
| `postgres` | default | `pgvector/pgvector:pg16`; CV chunks, saved jobs, cost log |
| `langfuse-web`, `langfuse-worker`, `langfuse-postgres`, `clickhouse`, `redis`, `minio` | `observability` | self-hosted Langfuse v4 |

```bash
docker compose up -d --build --wait                                    # API + PostgreSQL
TRACING_ENABLED=true docker compose --profile observability up -d --build --wait   # + Langfuse
docker compose down          # stop; add -v to delete the volumes (CV index, saved jobs, traces)
```

All ports bind to `127.0.0.1`. Default passwords in `compose.yaml` are local-only placeholders
marked `CHANGEME`. The Postgres volume keeps the password it was created with, so if you change
`POSTGRES_PASSWORD` later, the API fails startup with `SchemaInitializationError` until you reset
the volume.

Validated locally from fresh volumes, without `.env`: the full stack was healthy in 58 s,
`/ready` reported `schema ready` and 13 CV chunks, the smoke test passed, saved jobs and
`cost_log` rows persisted, traces appeared in Langfuse, and the API kept serving and shut down
cleanly with Langfuse stopped.

## Schema initialization

`src/db.py` applies three idempotent SQL files in one transaction at startup
(`DB_INIT_ON_STARTUP=true`): `src/rag/schema.sql` (pgvector extension, `cv_documents`,
`cv_chunks`), `src/tools/saved_jobs.sql` and `src/observability/schema.sql` (`cost_log`). If this
fails, startup stops with `SchemaInitializationError` (no connection string in the message), and
`/ready` reports any missing table. There is no migration framework: the schema only adds tables
(`CREATE ... IF NOT EXISTS`). Changing existing columns would need one.

## Health, readiness, shutdown

- `/health`: liveness only, used as the container `HEALTHCHECK` and the Container Apps startup and
  liveness probes.
- `/ready`: database reachable, required tables present, CV indexed, catalog loaded. It never
  calls an LLM. Used as the Compose health check and the Container Apps readiness probe.
- Shutdown: FastAPI lifespan flushes and stops the tracer (bounded to 3 s per step). Uvicorn has a
  20 s graceful timeout. Compose gives the API 30 s (`stop_grace_period`); Container Apps defaults
  to 30 s. Database connections are opened per operation and use one transaction each, so a stop
  cannot leave a half-written CV replacement.
- One worker and at most one replica: conversation threads and the response cache are
  process-local (see `docs/limitations.md`).

## Azure

Deployed and verified on 2026-10-29: **https://jobmatch-api.blueglacier-8c9485b6.northeurope.azurecontainerapps.io**
(fixture mode, synthetic data only). The app scales to zero, so the first request after idle takes
about 30 s.

### Architecture

```
Internet ── HTTPS ──► Container App  jobmatch-api        (Consumption, 0.5 vCPU / 1 GiB, 0–1 replicas)
                        │  image pull: user-assigned managed identity + AcrPull (no registry password)
                        ├─► Container Registry  jobmatchacr<suffix>   (Basic, admin user disabled)
                        ├─► PostgreSQL Flexible Server 16  jobmatch-pg-<suffix>   (B1ms, 32 GB, pgvector)
                        │      database careermatch: cv_documents, cv_chunks (vector), saved_jobs, cost_log
                        └─► Log Analytics  jobmatch-logs-<suffix>   (console + system logs, 30 days)
```

`<suffix>` is `uniqueString(resourceGroup().id)`, so names stay the same across re-runs. Resources
(`infra/azure/main.bicep`):

| Resource | Setting |
|---|---|
| Log Analytics workspace | PerGB2018, 30-day retention |
| Container Registry | Basic, admin user disabled |
| User-assigned managed identity | `AcrPull` on the registry; the app pulls images with it |
| Container Apps environment | Consumption only (no workload profiles, no private endpoint) |
| Container App `jobmatch-api` | 0.5 vCPU / 1 GiB, scale 0–1, external HTTPS-only ingress on port 8000; startup and liveness probes on `/health`, readiness on `/ready` |
| PostgreSQL Flexible Server | PostgreSQL 16, Burstable `Standard_B1ms`, 32 GB, no auto-grow, 7-day backups, no HA or geo backup, password auth, `azure.extensions=VECTOR` |
| Database `careermatch` | created by Bicep; the app creates the tables and runs `CREATE EXTENSION vector` at startup (`src/db.py`) |
| Firewall rule `AllowAzureServices` | public endpoint, TLS required (`sslmode=require`) |

Secrets live only in the Container App secret store and are referenced by `secretRef`:
`database-url`, plus `openai-api-key` and `langfuse-secret-key` when supplied. They are `@secure()`
Bicep parameters, so they never appear in deployment outputs, `what-if` or logs.

Networking caveat: `AllowAzureServices` admits traffic from any Azure-hosted source, not only this
subscription. That is acceptable for a synthetic-data demo. Real CV data needs VNet integration
with private access (see `docs/future-improvements.md`).

### Region

The deployment uses `northeurope` (Ireland). On a free-account subscription, PostgreSQL Flexible
Server reports `OfferRestricted` and offers no Burstable SKUs in `germanywestcentral` or
`westeurope`. `northeurope` allows B1ms and is also the cheapest EU region for this template (see
`docs/cost-controls.md`). On a Pay-As-You-Go subscription any region works: set `AZ_LOCATION`. To
check a region, run `az postgres flexible-server list-skus --location <region>` and look at the
`OfferRestricted` feature and the Burstable SKUs.

### Deploying

Requirements: Azure CLI (logged in, subscription selected), Docker, `python3` and `curl`. The image
is built **locally from the current working tree** and pushed with a short-lived Entra token from
`az acr login --expose-token` (valid for about 3 hours). ACR Tasks (`az acr build`) are not used:
Microsoft pauses them on free-credit subscriptions, and a normal Docker build works everywhere.

There are two workflows, and they touch different resources:

| Change | Command | Touches |
|---|---|---|
| New environment, or an infrastructure change (SKUs, secrets, `APP_MODE`, resources) | `infra/azure/deploy.sh` (Bicep) | every resource, including PostgreSQL |
| New application image (the normal case) | `infra/azure/deploy_app.sh` or the `deploy.yml` workflow | the Container App only |

```bash
az login
az account set --subscription "<subscription name>"

# Infrastructure (Bicep is the source of truth)
DRY_RUN=1 bash infra/azure/deploy.sh   # resource group only; then Bicep build + lint, validate, what-if
APP_MODE=fixture bash infra/azure/deploy.sh

# Application image
bash infra/azure/deploy_app.sh
```

What `deploy.sh` does:

1. creates the resource group and deploys the platform (`deployApp=false`);
2. runs `docker build`, logs in with the token and pushes to `<registry>.azurecr.io/job-match-agent:<tag>`;
3. deploys the Container App (`deployApp=true`) with that image;
4. runs `infra/azure/smoke_test.sh <url> --agentic`.

What `deploy_app.sh` does:

1. reads the registry from the existing Container App;
2. runs `docker build`, token login and `docker push`;
3. runs `az containerapp update --image …`, which creates a new revision, then waits until that
   revision reports Healthy. In single-revision mode the previous revision keeps serving until then;
4. runs the agentic smoke test against the public URL.

A Bicep run re-applies the PostgreSQL server even when nothing about it changes. During
validation, a North Europe incident in the PostgreSQL management plane (Azure Service Health
`6JYK-S00`, 29 October 2026) made every Bicep run fail on that resource, while the running database
and app stayed healthy. App-only deployments therefore never go through Bicep. If you update the
Container App directly, keep its settings in line with `main.bicep` so the next infrastructure
run doesn't revert anything unexpectedly.

Configuration and secrets:

- **PostgreSQL admin password:** read from `POSTGRES_ADMIN_PASSWORD`, or generated once (32 random
  letters and digits) into `~/.config/job-match-agent/postgres-admin-password`. That file has mode
  600 and is outside the repository. Re-runs must use the same password.
- **Secure parameters:** passed to Azure through a temporary mode-600 parameters file, never on the
  command line.
- **Image tag:** the UTC time plus the commit, with `-dirty` for uncommitted changes (for example
  `20261029-123702-8dd4848-dirty`). No Git tag is created.

Live mode: `APP_MODE=live OPENAI_API_KEY=… bash infra/azure/deploy.sh` stores the key as the
`openai-api-key` secret. The API has no authentication, so in live mode anyone with the URL can
spend OpenAI tokens; `AGENT_MAX_RUN_TOKENS` caps each run. Redeploy with `APP_MODE=fixture` to
remove the secret. **Status:** not executed in Azure. The optional live-mode smoke test was
deferred because of the active Azure PostgreSQL management-plane incident and because the public
API has no authentication. The live OpenAI path is covered by the local live evaluations
(`docs/evaluation.md`).

The smoke test prints status codes, latency, mode and the deterministic score, never response
bodies: `bash infra/azure/smoke_test.sh https://<fqdn> [--agentic]`.

### Verification (2026-10-29, fixture mode)

| Check | Result |
|---|---|
| Before deploying | Bicep build and lint clean; ARM validation passed for both passes; `what-if` showed exactly the 10 resources above |
| Image | pushed with a token; pulled by the managed identity in 4.4 s; revision Healthy with 100 % of traffic |
| Probes | one startup probe failed 0.7 s after the container started (before Uvicorn was listening), then all probes passed |
| PostgreSQL | state Ready, version 16.15, `azure.extensions = VECTOR`, database `careermatch` present |
| pgvector (SQL) | `CREATE EXTENSION vector` ok, extension 0.8.7; `'[1,2,3]' <-> '[1,2,4]'` = 1 and `'[1,0]' <=> '[0,1]'` = 1; 13 CV chunks of 512 dimensions; nearest-neighbour `ORDER BY embedding <=> …` works |
| Schema | `cost_log`, `cv_chunks`, `cv_documents` and `saved_jobs` created at startup; `/ready` reports "schema ready" |
| `/health`, `/ready`, `/jobs`, `/search` (jobs and CV) | 200 in 120–310 ms (warm) |
| `/analyze-job` for `match-01-python-backend` | 200, score 95.11 on three repeated calls, `score_source = deterministic` |
| `/chat` "Find AI Engineer jobs in Germany that fit my CV" | 200 in about 1.1 s; tools `search_jobs → calculate_match`; no ungrounded scores |
| `/chat` save request | 200 with `save_job`; an unknown job on `/analyze-job` returns 404 `job_not_found` |
| Langfuse unreachable (temporary revision with tracing pointed at an invalid host) | `/chat` still 200 in about 1 s; export failures appear only as background warnings |
| Cold start from zero replicas | about 26–31 s for the first request |
| Logs | no database, image-pull, configuration or pgvector errors; startup takes under 1 s once the container runs |

For the SQL check, a temporary firewall rule let one client IP reach the database. It was deleted
straight afterwards, and `AllowAzureServices` is the only remaining rule.

### Operating and cost control

```bash
RG=rg-job-match-agent
az resource list -g $RG -o table                                      # everything in the group
az containerapp show -g $RG -n jobmatch-api --query "{fqdn:properties.configuration.ingress.fqdn,state:properties.runningStatus}"
az containerapp revision list -g $RG -n jobmatch-api -o table         # revisions, health, traffic
az containerapp logs show -g $RG -n jobmatch-api --type console --tail 100
az acr repository show-tags -n <registry> --repository job-match-agent -o table
PG=$(az postgres flexible-server list -g $RG --query "[0].name" -o tsv)
az postgres flexible-server stop  -g $RG -n $PG   # stops compute billing; Azure restarts it after 7 days
az postgres flexible-server start -g $RG -n $PG   # /ready returns 503 while the server is stopped
az group delete --name $RG                        # deletes everything, including the database
```

### Continuous delivery

| Workflow | Trigger | Does |
|---|---|---|
| `.github/workflows/ci.yml` | pull requests; pushes to `main`, `feat/**`, `fix/**`; manual | installs pinned dependencies, byte-compiles, runs the full pytest suite against a pgvector service (integration tests included) and the deterministic evaluation gates, builds the image, smoke-tests the container and checks `/ready` on the Compose stack. No secrets, no paid calls. |
| `.github/workflows/deploy.yml` | manual (`workflow_dispatch`) only, GitHub environment `production` (add a required reviewer) | OIDC login, then `infra/azure/deploy_app.sh`: `docker build`, token login and `docker push` to ACR, `az containerapp update` to the new tag, wait for a healthy revision, agentic smoke test. Container App only. |
| `.github/workflows/live-eval.yml` | manual only, environment `live-evaluation` | paid live system evaluation and version comparison; uploads the reports |

Deployment is never automatic on push. Merging to `main` runs CI; deploying is a separate, approved
action. `deploy.yml` deploys application images only; infrastructure is created and changed with
`deploy.sh`.

GitHub configuration (names only):

| Name | Kind | Purpose |
|---|---|---|
| `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` | secrets | OIDC federated credential (a user-assigned managed identity or Entra app; no client secret stored). Roles: `AcrPush` on the registry, `Contributor` on the Container App. |
| `AZURE_RESOURCE_GROUP`, `AZURE_CONTAINER_APP` | variables | deployment target; the registry is read from the Container App |
| `OPENAI_API_KEY` | secret (environment `live-evaluation`) | live evaluations only |

Runtime secrets (OpenAI key, database URL, Langfuse keys) are not stored in GitHub. They live in
the Container App secret store, set by `deploy.sh`.
