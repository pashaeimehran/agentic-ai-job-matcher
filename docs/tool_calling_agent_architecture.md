# Tool Calling and Manual Agent Loop

This layer lets an LLM decide *which* application capability to use, while
Python decides *whether* the request is valid and *how* it is executed. It is
deliberately built without an agent framework so the mechanics are explicit.

## Components

```
src/llm/tool_calling.py      provider-neutral ToolSpec, ToolCall, AssistantTurn, ToolResultMessage
src/llm/openai_provider.py   OpenAIProvider.complete(): Responses API adapter (only SDK-specific code)
src/tools/schemas.py         strict argument models, result models, ToolOutcome / ToolError envelope
src/tools/errors.py          failure codes, exception classification, retryable vs. non-retryable
src/tools/registry.py        Tool + ToolRegistry: name -> argument schema -> handler; validate/execute
src/tools/jobs.py            JobCatalog, search_jobs, get_job_details
src/tools/cv.py              search_cv
src/tools/matching.py        calculate_match
src/tools/saved_jobs.py      SavedJobStore (JSON file / memory), save_job
src/tools/context.py         ToolContext: application-owned scope (CV, catalog, store)
src/agent/loop.py            ManualAgent: bounded model/tool loop
src/agent/schemas.py         AgentConfig, ToolCallRecord, AgentRunResult
src/agent/prompts.py         system prompt
```

## What a tool call is

A tool call is a *request* emitted by the model: a tool name plus a JSON string
of arguments. It is not code execution. The model only sees each tool's name,
description and JSON schema (generated from the Pydantic argument model). The
application decides whether and how to act on the request.

## Tools and the code they reuse

| Tool | Arguments | Reuses | Result |
|---|---|---|---|
| `search_jobs` | `query`, `location?`, `limit` | `load_fixture_jobs`, `matching_jobs.json`, `JobPosting` | ranked `JobSummary` list, explicit empty result |
| `get_job_details` | `job_id` | `JobPosting`, language levels, `prepare_job` | full structured posting |
| `search_cv` | `query`, `k`, `section?` | `matching.retrieval.hybrid_search` over `CVService` | chunk ID, section, text, retrieval scores |
| `calculate_match` | `job_ids` (1–10) | `matching.service.analyze_jobs` → `score_match`, `rank_analyses` | ranked deterministic `MatchScore`, comparisons, evidence |
| `save_job` | `job_id`, `note?` | new minimal `SavedJobStore` | `saved` / `already_saved` |

The job catalog is the committed structured fixture set. `jobs.json` contains
four phrasing variants per posting for extraction benchmarking; identical
structured postings are collapsed to the first ID. All fixture postings are
German, so `location="Germany"` matches the whole catalog. BA Jobsuche is not
wired in: its results are raw text that would need per-posting LLM extraction.
A future catalog only needs to implement `JobCatalog`.

`cv_id` is never a tool argument. CV scope comes from `ToolContext`, so a model
cannot request another user's CV (extra keys such as `cv_id` are rejected).

No saved-job persistence existed before. `JsonFileSavedJobStore` writes
`data/private/saved_jobs.json` (git-ignored) atomically; saving is idempotent
and only catalog job IDs are accepted.

## Why arguments are validated

Model output is untrusted input. Every argument model uses
`extra="forbid"` and `strict=True`:

- unknown keys (`cv_id`, `score`) are rejected, not ignored;
- types are not coerced (`"5"` is not an integer);
- ranges and lengths are bounded (`limit` 1–20, `k` 1–10, `job_ids` 1–10, unique).

Validation failures return an `invalid_arguments` observation with field
locations and messages (never input values). The handler is not called.

## How the manual loop works

```
messages = [system, user]
for step in 1..max_model_turns:
    turn = provider.complete(messages, registry.specs())
    if no tool calls: return completed(turn.text)
    for each call:
        if tool budget used up: return max_tool_calls_exceeded
        outcome = registry.execute(name, raw_args, context, retry_policy)   # never raises
        messages += ToolResultMessage(call_id, outcome JSON)
return max_model_turns_exceeded
```

Every outcome, including errors, is returned to the model as an observation,
so it can correct an argument, choose another tool, or explain the failure.

## Limits

| Setting | Default | Meaning |
|---|---|---|
| `max_model_turns` | 8 | provider calls per run |
| `max_tool_calls` | 12 | requested calls per run, including rejected/failed ones |
| `tool_retry.max_retries` | 1 | re-executions after a *retryable* failure |
| `tool_retry.backoff_seconds` | 0 | linear backoff (`backoff * attempt`) |

Reaching a limit returns `AgentRunResult` with a non-`completed` status and an
error message; the loop never continues past it. A provider exception returns
`provider_error`.

## Failure handling and retry policy

| Code | Cause | Retried |
|---|---|---|
| `unknown_tool` | name not registered | no (not executed) |
| `invalid_arguments` | JSON/schema failure | no (not executed) |
| `not_found` | unknown `job_id` | no |
| `precondition_failed` | e.g. CV not ingested | no |
| `transient` | `TransientToolError`, `TimeoutError`, `ConnectionError`, `BAJobsucheUnavailable`, `psycopg.OperationalError` | yes, bounded |
| `internal` | any other exception | no |

Only `transient` failures are retried, and only by the executor. Unknown
exceptions get a generic message (`Tool failed unexpectedly (KeyError).`)
because exception text may contain CV text, SQL or credentials. The model
itself may choose to call a tool again; that counts toward `max_tool_calls`.

Empty results are not errors: `search_jobs` and `search_cv` return status
`empty` with an explicit message, and the prompt forbids fabricating results.

## Deterministic vs. LLM boundary

| LLM decides | Python decides |
|---|---|
| whether a tool is needed, which one, with which arguments | whether arguments are valid |
| the order of steps | job search ranking, CV retrieval, scoring, ranking |
| wording of the final answer | which errors are retryable, when to stop |

`calculate_match` delegates to `analyze_jobs`; no score logic exists in the
tool layer or prompt. As a post-hoc guard, `ungrounded_scores()` lists every
score-like number in the final answer (`NN/100`, `score ... NN`) that no
`calculate_match` call in that run produced. It is a detector, not a rewriter.

## Logging

Loggers `src.agent` and `src.tools` emit one JSON object per event:
`agent_run_started`, `model_turn`, `tool_succeeded`, `tool_rejected`,
`tool_failed`, `model_turn_failed`, `agent_run_finished`. Fields include
`run_id`, `step`, `tool`, `arguments_valid`, `attempt(s)`, `error_code`,
`retryable`, `will_retry`, `duration_ms`. The user request, argument values,
CV text and exception messages are not logged.

## Tool-selection evaluation

`data/fixtures/tool_selection_cases.json` holds 18 requests with acceptable
first tools, optional expected sequences (ordered subsequence), and reference
calls. `scripts/evaluate_tool_selection.py` computes from actual runs:
first-tool accuracy, sequence accuracy, argument-validity rate, task
completion (completed, non-empty answer, no ungrounded scores, no tool for
no-tool cases, sequence satisfied) and tool-call counts.

- `--mode replay` (deterministic, committed result): replays reference calls
  through the real loop. It checks the dataset and harness, not the model.
- `--mode live` (paid, git-ignored result): the configured OpenAI model on the
  synthetic CV and fixture catalog. One run per case; results vary between runs.

## Why LangGraph is not used yet

The loop above *is* what an agent framework automates: a state of messages, a
model node, a tool node, a conditional edge on "has tool calls", and limits.
Building it by hand makes these mechanics, failure modes and limits explicit,
and gives a behavioural baseline (tests and evaluation) for the LangGraph
version to match.

## Reuse

The tool layer has no dependency on the loop. `ToolRegistry.specs()` gives
JSON schemas, and `ToolRegistry.execute(name, args, context)` returns a
serialisable `ToolOutcome` and never raises for tool problems. A LangGraph
tool node, a FastAPI route or an MCP server can call the same two methods with
their own `ToolContext`.

## Running

```bash
python -m pytest tests/unit/test_tools.py tests/unit/test_agent_loop.py
python -m scripts.evaluate_tool_selection --mode replay
python -m scripts.evaluate_tool_selection --mode live          # paid
python -m scripts.run_agent "Find AI Engineer jobs in Germany that fit my CV and explain the best matches."
```

## Limitations

- The job catalog is a small set of synthetic fixtures; keyword search is not semantic.
- Observations are not truncated; bounds on `limit`, `k` and `job_ids` keep them small.
- The score guard uses patterns and can miss unusual phrasings or flag unrelated numbers.
- Provider errors are not retried by the loop.
- Live evaluation is a single run of 18 cases and is not statistically robust.
