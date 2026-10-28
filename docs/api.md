# REST API

A typed FastAPI backend over the shared tool layer and the LangGraph agent.

```
HTTP request → route (src/api/app.py) → ApiServices.call_tool() / LangGraphAgent.run()
             → ToolRegistry.execute() → src.matching / src.rag / job catalog / saved-job store
```

Routes contain no search, matching or scoring logic. Every endpoint except `/health` and the
catalog listing is a call to one of the five shared tools, or to the LangGraph agent.

## Running

```bash
python -m pip install -r requirements.txt
python -m scripts.serve_api            # http://127.0.0.1:8000, OpenAPI UI at /docs
```

Default services (`build_default_services`): the synthetic fixture CV with offline hash
embeddings, the fixture job catalog, and saved jobs in git-ignored `data/private/saved_jobs.json`.
`/chat` is enabled only when `OPENAI_API_KEY` is set (`OPENAI_MODEL` defaults to `gpt-4o-mini`);
those calls are paid. All other endpoints are deterministic and make no LLM calls.

The CV is selected by server configuration, never by the request. A request that includes `cv_id`
is rejected as a validation error.

## Endpoints

| Method | Path | Backed by | Purpose |
|---|---|---|---|
| GET | `/health` | registry, catalog | process health; no LLM call |
| GET | `/jobs` | `JobCatalog.all()` | browse the catalog (paged, catalog order) |
| GET | `/jobs/{job_id}` | `get_job_details` | full structured posting |
| POST | `/search` | `search_jobs` or `search_cv` | ranked job search or CV evidence retrieval |
| POST | `/analyze-job` | `calculate_match` | deterministic CV compatibility for one job |
| POST | `/chat` | `LangGraphAgent` | natural-language request through the agent |

Every response includes `request_id`. A client may send `X-Request-ID` (`[A-Za-z0-9_-]{1,64}`);
otherwise one is generated. It is echoed in the `X-Request-ID` response header.

### GET /health

```json
{"status": "ok", "version": "0.1.0", "agent": "langgraph", "chat_available": true,
 "tools": ["search_jobs", "get_job_details", "search_cv", "calculate_match", "save_job"],
 "catalog_jobs": 16}
```

### GET /jobs?offset=0&limit=20

`offset ≥ 0`, `1 ≤ limit ≤ 100`. This is a listing, not a search: no ranking, no filtering.

```json
{"request_id": "req-1", "total": 16, "offset": 0, "limit": 1,
 "jobs": [{"job_id": "fixture-01", "role": "AI Engineer", "location": "Berlin", "seniority": null,
           "skills": ["Python", "LangGraph", "Azure"], "source": "extraction_fixture"}]}
```

### POST /search

The `target` field selects the tool. The remaining fields are exactly the tool's arguments
(the request models subclass `SearchJobsArgs` / `SearchCVArgs`), with the same limits and strict
typing (`"5"` is not an integer).

Jobs (`search_jobs`: `query` 1–200 chars, `location?`, `limit` 1–20, default 5):

```json
{"target": "jobs", "query": "AI Engineer", "location": "Germany", "limit": 1}
```
```json
{"request_id": "req-1", "target": "jobs", "status": "ok", "message": null, "total_matches": 16,
 "jobs": [{"job_id": "fixture-01", "role": "AI Engineer", "location": "Berlin", "seniority": null,
           "skills": ["Python", "LangGraph", "Azure"], "relevance": 1.0}]}
```

`relevance` is deterministic keyword relevance, not a CV match score.

CV (`search_cv`: `query` 1–500 chars, `k` 1–10, default 3, `section?`):

```json
{"target": "cv", "query": "Python experience", "k": 1}
```
```json
{"request_id": "req-1", "target": "cv", "status": "ok", "message": null,
 "evidence": [{"chunk_id": "fe5e5a993fbd4be232df41bd", "section": "Programming",
               "text": "## Programming\nPython for data processing and model training. C++ for embedded software. SQL for relational queries.",
               "fusion_score": 0.032522, "vector_cosine": 0.150756, "token_coverage": 0.5}]}
```

(Synthetic fixture CV with offline hash embeddings; values differ with OpenAI embeddings.)

**Empty results are not errors.** They return `200` with `"status": "empty"`, an empty list and
a `message`, which is the tool layer's semantics. A frontend should render "no results" rather
than an error.

### POST /analyze-job

```json
{"job_id": "fixture-01"}
```
```json
{"request_id": "req-1", "score_source": "deterministic",
 "match": {"rank": 1, "job_id": "fixture-01", "role": "AI Engineer",
           "score": {"value": 69.59,
                     "components": [{"name": "skills", "raw": 0.666667, "weight": 0.5, "weighted_contribution": 33.333333},
                                    {"name": "experience", "raw": 0.5, "weight": 0.2, "weighted_contribution": 10.0},
                                    {"name": "languages", "raw": 1.0, "weight": 0.2, "weighted_contribution": 20.0},
                                    {"name": "retrieval", "raw": 0.625988, "weight": 0.1, "weighted_contribution": 6.259882}],
                     "evidenced_requirements": 4, "total_requirements": 6, "retrieval_cosine": 0.252},
           "supported": ["Python", "Azure", "English", "German"], "unmet": [], "unknown": ["LangGraph", "3 years"],
           "comparisons": [{"kind": "skill", "requirement": "Python", "status": "supported",
                            "reason": "Explicitly evidenced in CV",
                            "evidence": [{"chunk_id": "…", "quote": "Python for data processing and model training."}]}],
           "retrieved_chunk_ids": ["…"]},
 "scoring_note": "score.value is calculated by deterministic Python code (0-100). Quote it exactly; never estimate or adjust it."}
```

No LLM is involved. The score is `src.matching.scoring.score_match`, reached through
`calculate_match`. `unknown` means the CV contains no evidence, not that the candidate lacks the skill.

### POST /chat

```json
{"message": "How well do I fit fixture-01?", "thread_id": null}
```
```json
{"request_id": "req-1", "run_id": "3f2a9c1d7b4e", "thread_id": "9d0c…", "status": "completed",
 "answer": "fixture-01 scores 69.59/100. …", "model_turns": 2,
 "tool_calls": [{"step": 1, "tool": "calculate_match", "status": "ok", "error_code": null, "attempts": 1}],
 "ungrounded_scores": []}
```

- `message`: 1–4000 characters. `thread_id` (optional, `[A-Za-z0-9_-]{1,64}`) continues a conversation
  held in the process-local LangGraph checkpointer. Omit it to start a new thread; the response
  returns the new `thread_id`. Request IDs never select a thread.
- `tool_calls` summarizes the run. Tool payloads (job data, CV excerpts) are not returned.
- `ungrounded_scores` lists score-like numbers in the answer that no `calculate_match` call in the
  run produced. A non-empty list means the answer should not be trusted for scores.
- The production path is LangGraph. The manual loop is not reachable from the API; it remains
  available as the CLI baseline (`scripts.run_agent --agent manual`).

## Error contract

All non-2xx responses share one body (`ErrorResponse`):

```json
{"request_id": "req-1", "run_id": null,
 "error": {"code": "job_not_found", "message": "Unknown job_id(s): nope. Use search_jobs first.",
           "retryable": false, "details": [{"tool_error_code": "not_found"}]}}
```

| HTTP | `error.code` | Cause | `retryable` |
|---|---|---|---|
| 422 | `validation_error` | body or query does not match the schema; `details` holds `loc`, `msg` and `type` (input values omitted) | false |
| 404 | `job_not_found` | unknown `job_id` (tool `not_found`) | false |
| 409 | `cv_not_available` | configured CV not ingested (tool `precondition_failed`) | false |
| 503 | `dependency_unavailable` | transient dependency failure after bounded retries (tool `transient`) | true |
| 500 | `tool_failure` | unexpected tool exception (tool `internal`); generic message | false |
| 502 | `llm_provider_error` | the LLM provider call failed during `/chat`; `run_id` set | true |
| 422 | `agent_limit_exceeded` | the agent hit a model-turn, tool-call or invalid-request limit; `details[0].agent_status` names it; `run_id` set | false |
| 503 | `chat_unavailable` | no LLM provider configured | false |
| 404 / 405 | `not_found` / `method_not_allowed` | unknown route or method | false |
| 500 | `internal_error` | any other exception; generic message | false |

Validation example (strict typing inherited from the tool schema):

```json
{"request_id": "req-1", "run_id": null,
 "error": {"code": "validation_error", "message": "Request does not match the API schema.", "retryable": false,
           "details": [{"loc": ["body", "jobs", "limit"], "msg": "Input should be a valid integer", "type": "int_type"}]}}
```

Responses never contain stack traces, exception text, provider payloads or credentials.
Unexpected exceptions are logged by type only (`src.api` logger), with the request ID.

**Empty search** is deliberately not in this table: it is a successful `200` with `status: "empty"`.

## Dependency injection

`create_app(services)` stores an `ApiServices` object (tool context, registry, LangGraph agent,
retry policy) on `app.state`. Routes receive dependencies through `Depends(get_services)`,
`get_catalog`, `get_agent` and `get_request_id`. Tests build an app per test with their own
context, registry and scripted provider; there is no module-level mutable state. Without injected
services, `build_default_services()` runs in the app's startup lifespan.

## Tests

`tests/unit/test_api.py` (TestClient, scripted provider, no network) covers every endpoint with
happy paths, validation failures and each domain error above. It also checks that routes reach the
shared registry (a spy registry records the executed tool), that `/analyze-job` scores equal a direct
`analyze_jobs()` computation, request-ID handling, thread isolation, and that exception text never
appears in a response.

## Limitations

- Single-user and unauthenticated. Anyone who can reach the port can use the configured CV and
  any known `thread_id`. Bind to localhost (the default) until authentication exists.
- Conversation threads live in process memory: they are lost on restart and not shared between workers.
- The default services use the synthetic CV. Serving an ingested PostgreSQL CV requires building
  `ApiServices` with a live `ToolContext`; no configuration switch exists yet.
