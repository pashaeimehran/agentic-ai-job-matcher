# Future improvements

Ordered by what each change unlocks. Each item maps to a limitation in `docs/limitations.md`.

## 1. Required before real users

1. **Authentication and per-user isolation.** Entra ID (or another OIDC provider) in front of the
   API. Derive `owner_id` from the token, and scope `cv_documents`, `cv_chunks`, `saved_jobs`,
   checkpoints and the cache key by owner. Add cross-tenant access tests.
2. **Durable, owner-scoped conversation state.** Replace the in-memory LangGraph checkpointer
   with the PostgreSQL checkpointer (`langgraph-checkpoint-postgres`) plus a retention job. Then
   raise `maxReplicas` above 1.
3. **Private networking.** VNet-integrated Container Apps environment, PostgreSQL with private
   access, managed-identity (Entra) authentication to PostgreSQL instead of a password in
   `DATABASE_URL`, Key Vault references for the remaining secrets.
4. **Data-subject tooling.** Export and delete endpoints for CV, saved jobs and threads,
   scheduled retention, and a documented backup-expiry window.
5. **Rate limiting and spending caps** per user and per day, backed by the existing `cost_log`.

## 2. Evaluation depth

- Larger labeled sets: more synthetic CVs (different seniorities and fields), more jobs per
  requirement type, and adversarial grounding scenarios.
- Repeat the prompt comparison with more samples (power analysis suggests roughly 100+ runs per
  arm for the observed effect) and on a second model.
- Replace or complement the lexical deficit detector with a calibrated LLM judge, validated
  against a manually labeled set of answers.
- Schedule `live-eval.yml` weekly, archive its reports, and alert on regressions against stored
  baselines.
- Resolve the OpenAI SDK conflict (or move to the Ragas version that supports it) so Ragas can run
  in the main environment and in CI.

## 3. Cost and performance

- Model tiering. Route simple single-tool requests to a cheaper model, and use a stronger model
  only for multi-job explanations. Measure with the existing version-comparison harness.
- Shared response cache (e.g. Azure Cache for Redis), with entries encrypted and scoped by owner,
  once several replicas run.
- Use provider prompt caching for the stable system prompt and tool schemas. Cached input tokens
  are already priced separately.
- Streaming responses for `/chat`.
- Approximate vector indexes (HNSW) once a CV store grows beyond exact-search sizes.

## 4. Product scope

- A real job source with a documented API and terms, replacing the fixture catalog. Keep fixtures
  for tests.
- OCR for scanned PDFs and better multi-column PDF parsing.
- Optional MCP server configuration from `Settings` (PostgreSQL, live embeddings), and an
  HTTP transport with authentication.
- Saved-job management endpoints (list, delete, edit note).

## 5. Operations

- Run the optional live-mode smoke test in Azure once the API has authentication (or behind a
  temporary IP restriction on ingress), then return to fixture mode.
- Split `main.bicep` into a platform module and an app module, so infrastructure changes to the
  Container App also skip PostgreSQL. Image updates already do (`deploy_app.sh`).
- Keep the Azure PostgreSQL password out of a connection string: Entra authentication for
  PostgreSQL with the Container App's managed identity.
- A GitHub OIDC managed identity and federated credential, provisioned by Bicep, so `deploy.yml`
  works without manual setup.
- Migrations (e.g. Alembic or versioned SQL files with a schema-version table) before the first
  breaking schema change.
- Azure Monitor alerts on `/ready` failures, error-rate and cost-log spend. Dashboards from
  `cost_log`.
- Image signing and an SBOM in the deploy workflow, and Dependabot for pinned dependencies.
- Blue/green revisions in Container Apps (`activeRevisionsMode: Multiple`) with traffic shifting
  after the smoke test.
