# Data protection and responsible use

Engineering notes for this specific system: what personal data it touches, where that data
goes, and what must change before real CVs are processed at scale. This is not legal advice.
A real deployment needs its own GDPR assessment (lawful basis, records of processing, DPIA where
required) by someone qualified to make it.

## Personal data in this system

A CV is personal data. It can contain special-category data, such as health information, a photo,
or details that imply ethnicity or religion. This system treats everything derived from a CV as
personal data too:

| Data | Where | Contents | Personal data |
|---|---|---|---|
| CV chunks | `cv_chunks` (PostgreSQL) or process memory | verbatim CV text in ~900-character chunks, section names, offsets | yes |
| Embeddings | `cv_chunks.embedding` | vectors computed from CV text | yes, derived. Similarity search can partly reveal content, so they are protected and deleted like the text. |
| CV metadata | `cv_documents` | CV ID, source filename, content hash, embedding model | yes (filename and hash identify a document) |
| Saved jobs | `saved_jobs` or `data/private/saved_jobs.json` | job IDs, optional note (≤ 500 chars), timestamp | potentially (a note can contain anything) |
| Conversation state | LangGraph in-memory checkpointer | messages, tool observations incl. CV excerpts | yes. Process memory only, gone on restart. |
| LLM response cache | process memory | model outputs, possibly paraphrasing CV content | yes. Never on disk, TTL 1 h by default. |
| Cost log | `cost_log` | IDs, model, token counts, cost, latency, status | no CV content (by schema) |
| Application logs | stdout / Log Analytics | events, IDs, status codes, durations, error categories | designed to contain none |
| Traces (Langfuse) | Langfuse | span names, IDs, metadata, usage | none by default. With `LANGFUSE_CAPTURE_CONTENT=true`, prompts and tool outputs, i.e. CV text. |

## Processors and data flow

- **Live mode** sends CV chunks to OpenAI for embeddings and, during chat, sends retrieved CV
  excerpts and job text as prompt content. OpenAI is then a processor. Region, retention and
  training settings depend on the OpenAI account and contract, not on this code.
- **Fixture mode** sends nothing to any external service. The scripted provider and hash
  embeddings run locally.
- **Langfuse**: self-hosted in Compose (data stays on the machine). Langfuse Cloud is optional in
  Azure. Choose its EU region and keep content capture off.
- **Azure**: PostgreSQL, Container Apps and Log Analytics live in the deployment region
  (`northeurope` for the current deployment; see `docs/deployment.md`).

## Controls in place

- Logs and metrics carry no CV text by construction. `LLMCallRecord` has no content fields; the
  API logs neither request bodies, query strings nor validation-error input values. Tests
  (`test_no_raw_cv_text_or_secrets_in_operational_logs`, the agent, graph and tool log tests,
  and the provider-failure test) assert this.
- Trace content capture is opt-in and off by default.
- The cache is memory-only with a TTL and keys are hashes, so CV-derived output is never written
  to a shared store.
- Secrets are `SecretStr` (masked in `repr`, logs and error messages). They come from environment
  variables or the Container App secret store and are never baked into the image (`.dockerignore`
  excludes `.env*` and `data/private`). The smoke test prints status codes, never bodies.
- Personal files belong in git-ignored `data/private/`. Demos, screenshots, CI and all committed
  evaluations use only the synthetic CV (`data/fixtures/sample_cv.md`) and fixture jobs.
- The API selects the CV by server configuration. A request cannot name another CV (`cv_id` in a
  body is rejected).

## Retention and deletion

- Deleting a CV: `python -m scripts.cv delete --cv-id <id>` removes the document row, and
  `ON DELETE CASCADE` removes its chunks and embeddings in the same transaction. Re-ingesting the
  same ID replaces content transactionally.
- Saved jobs: delete rows from `saved_jobs` or the JSON file. No API endpoint does this yet.
- Conversations and cache: cleared on process restart. Cache entries also expire after
  `LLM_CACHE_TTL_SECONDS`.
- Cost log: contains no CV content. Retention is up to the operator, for example
  `DELETE FROM cost_log WHERE created_at < now() - interval '90 days'`.
- Backups: Azure PostgreSQL keeps 7 days of point-in-time backups, so a deleted CV remains in
  backups for up to 7 days. Log Analytics retention is 30 days.
- No automatic retention schedule exists. See below.

## Human oversight and the EU AI Act

AI systems used for recruitment or selection, including analysing applications or evaluating
candidates, are listed as high-risk in Annex III of the EU AI Act. This project is designed as a
job seeker's own tool for assessing their CV against postings, not as an employer's screening
system. Design choices that keep it that way:

- The match score is advisory. It is a deterministic, documented weighted heuristic
  (`src/matching/`), not a probability of being hired, and the API says so in `scoring_note`.
- Scores and requirement statuses come only from deterministic code. The LLM cannot invent or
  change a score; ungrounded scores in answers are detected and reported (`ungrounded_scores`).
- "Unknown" (no evidence in the CV) is kept separate from "unmet", and prompt v2 forbids
  presenting missing evidence as a deficit of the person (measured in
  `evaluation/results/version_comparison.md`).
- Every claim cites CV evidence the user can inspect. The system takes no action beyond saving a
  job the user explicitly asked to save.

Repurposing this system to rank or filter other people's applications would move it into the
high-risk category, with obligations (risk management, data governance, logging, human oversight,
accuracy testing on representative data) that the current project does not meet.

## Current limitations relevant to data protection

- No authentication or authorization. Anyone who can reach the API can query the configured CV.
  The Azure deployment exposes a public HTTPS endpoint and must only ever serve the synthetic CV.
- Single-user design. `saved_jobs.owner_id` exists, but there is no tenant isolation for CVs.
- No data-subject request tooling (export, rectification) beyond the CLI delete.
- No encryption beyond what the platform provides (TLS to Azure PostgreSQL, storage encryption at
  rest), and no field-level encryption of CV text.
- PostgreSQL is reachable from Azure services over a public endpoint (TLS, firewall), not a
  private network.

## Required before processing real CVs at scale

1. Authentication (e.g. Entra ID) and per-user authorization. CV, chunks, saved jobs and threads
   scoped by owner, with tests for cross-user access.
2. Private networking (VNet-integrated Container Apps, private PostgreSQL access) and managed
   identity for database authentication instead of a password in a connection string.
3. A data-processing agreement with the LLM provider, an EU data-residency choice, and a
   documented decision on whether CV text may leave the EU.
4. Retention jobs (CV and saved-job expiry, cost-log pruning), deletion that also covers backups
   and traces, and self-service export and delete endpoints.
5. A DPIA, a privacy notice that names the processors, and consent or another documented lawful
   basis.
6. Rate limiting and audit logging of access to personal data.
7. Durable, owner-scoped conversation storage with retention, or none at all.
