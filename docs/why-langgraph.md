# LangGraph Orchestration and the Manual Loop

The project has two orchestrators over one tool layer:

| | Manual loop (`src/agent/loop.py`) | LangGraph agent (`src/graph/`) |
|---|---|---|
| Role | Baseline, framework-free reference | Production path (`POST /chat`) |
| Control flow | A `for` loop with early `return`s | Explicit nodes, edges and conditional routing |
| Tools | `ToolRegistry.execute()` | `ToolRegistry.execute()` (same call, same registry) |
| Scores | `calculate_match` → `src.matching` | identical |
| Memory | none (one request per run) | optional per-thread checkpointing |

The graph adds no matching intelligence. It makes the orchestration that the manual loop
already performed explicit, testable per step, and resumable per conversation thread.

## Graph

```
START → prepare → agent ──route_after_agent──┬─▶ execute_tools ──route_after_tools──┬─▶ agent
                                             ├─▶ finalize → END                     └─▶ fail → END
                                             └─▶ fail → END
```

| Node | Responsibility | Business logic it calls |
|---|---|---|
| `prepare` | Understand Request: start a run on a new or continued thread, add the user message, reset per-run counters | none (no LLM) |
| `agent` | One model turn via `ToolCallingLLMProvider.complete()`; record requested calls or the answer | provider adapter only |
| `execute_tools` | Run each requested call within the tool budget; append observations; track invalid-request turns | `ToolRegistry.execute()` |
| `finalize` | Produce the final recommendation; flag score-like numbers not returned by `calculate_match` | `ungrounded_scores()` from the manual loop |
| `fail` | Terminate with a structured failure status and safe message | none |

Nodes are methods of `GraphNodes` and take the state dict directly, so each is unit-tested
without compiling the graph (`tests/unit/test_graph_nodes.py`).

### Why there are no per-tool nodes

A conceptual flow such as *search jobs → job details → CV evidence → calculate match → rank →
recommend* is a *possible* path, not a fixed pipeline. The model chooses which tools to call;
`execute_tools` dispatches any of the five through the registry. Separate `search_jobs` /
`calculate_match` nodes would each contain the same `registry.execute()` call and would force
a fixed order. Short requests therefore take short paths:

| Request | Path |
|---|---|
| "What is a CV summary?" | `prepare → agent → finalize` |
| "Search my CV for ROS2" | `prepare → agent → execute_tools(search_cv) → agent → finalize` |
| "Save job fixture-13" | `prepare → agent → execute_tools(save_job) → agent → finalize` |
| Reference scenario | `prepare → agent → execute_tools(search_jobs) → agent → execute_tools(calculate_match) → agent → finalize` |

Ranking is not a graph step: `calculate_match` returns jobs already ordered by
`src.matching.scoring.rank_analyses()`.

## State

`AgentState` (`src/graph/state.py`) is a `TypedDict`:

| Field | Scope | Purpose |
|---|---|---|
| `request`, `run_id` | input | current request and run identifier |
| `messages` | thread | provider-neutral conversation (`ChatMessage`, `AssistantTurn`, `ToolResultMessage`) |
| `pending_calls` | turn | calls requested by the last model turn |
| `tool_calls` | run | `ToolCallRecord`s (the same record type as the manual loop) |
| `model_turns`, `invalid_tool_turns` | run | execution counters for the limits |
| `path` | run | nodes visited; used by tests and the comparison report |
| `status`, `final_answer`, `error`, `ungrounded_scores` | run | outcome |

Fields are overwritten by nodes rather than merged by reducers. Every run-scoped field is
reset by `prepare`, so a continued thread carries the conversation but not old counters.

The state never contains raw CV documents. It contains the CV excerpts the model retrieved
through `search_cv` (the model needs them as observations). Nothing in the state is logged:
the `src.graph` logger records event names, IDs, counters, tool names and statuses only.

## Conditional routing

Both routing functions are pure functions of `(state, limits)` and are tested directly:

| Function | Condition | Route |
|---|---|---|
| `route_after_agent` | provider raised | `fail` (`provider_error`) |
| | model requested tool calls | `execute_tools` |
| | model answered | `finalize` |
| `route_after_tools` | tool-call budget exhausted | `fail` (`max_tool_calls_exceeded`) |
| | `max_invalid_tool_turns` consecutive turns with only rejected calls | `fail` (`invalid_tool_requests`) |
| | `model_turns >= max_model_turns` | `fail` (`max_model_turns_exceeded`) |
| | otherwise | `agent` |

Execution failures (`not_found`, `precondition_failed`, `transient` after retries,
`internal`) are *not* routing failures. They return to the model as observations, exactly
as in the manual loop, so it can explain or recover. A single invalid request is also
returned as an observation; only repeated all-invalid turns terminate.

## Bounded cycles

The `agent ⇄ execute_tools` cycle is bounded three ways:

1. `max_model_turns` (default 8): checked by `route_after_tools`.
2. `max_tool_calls` (default 12): checked per call in `execute_tools`; counts rejected and failed calls.
3. LangGraph `recursion_limit = 2 × max_model_turns + 4`: a backstop set on every
   `invoke`. Routing ends runs before it; if it ever fires, the run returns
   `graph_recursion_limit` instead of raising.

Tool retries remain the registry's bounded `RetryPolicy` (default one retry, transient errors only).

## Checkpointing

`LangGraphAgent(..., checkpointer=memory_checkpointer())` enables per-thread conversation
state via LangGraph's `InMemorySaver`:

- `run(request, thread_id="t1")` twice continues the same conversation; for example,
  "save the first one" can refer to jobs found in the previous turn.
- Without a checkpointer every run is independent. Passing a `thread_id` without one raises.
- Deserialization is restricted to an explicit allowlist (`CHECKPOINT_TYPES`). A test fails
  if any application type is restored without being on it.
- If a previous run stopped mid-turn (for example, at the tool limit), `prepare` adds a
  `not_executed` observation for each unanswered call so the history remains valid for the provider.

Scope: process-local and lost on restart. It is not durable, multi-user memory and is
unrelated to CV storage (PostgreSQL) or saved jobs (`SavedJobStore`).

## What is reused and what stays deterministic

| Reused unchanged | Location |
|---|---|
| Tool schemas, validation, execution, error classification, retries | `src/tools/` |
| Provider abstraction and OpenAI adapter | `src/llm/` |
| `ToolCallRecord`, `AgentConfig` limits, system prompt | `src/agent/` |
| Ungrounded-score detector | `src/agent/loop.py::ungrounded_scores` |
| Matching, scoring, ranking | `src/matching/` via `calculate_match` |

The LLM never computes, estimates, modifies or overrides a match score. Scores are produced
only by `src.matching.scoring.score_match`. Graph tests assert that the scores recorded in a
run equal a direct `analyze_jobs()` computation, including when the model's answer misquotes
them (the misquote is flagged in `ungrounded_scores`).

## Comparison with the manual loop

`python -m scripts.compare_agents` replays fixed model turns through both orchestrators with
the same registry and fixtures (29 scenarios: the 18 tool-selection cases plus 11 failure,
limit and grounding scenarios). It compares status, tool sequence, tool-call count, model
turns, per-call outcome/attempts, match scores and ungrounded-score flags. It is deterministic
and measures orchestration equivalence, not model quality.

Recorded result (`evaluation/results/agent_comparison_offline.json`):

| Metric | Manual | LangGraph |
|---|---|---|
| Scenarios functionally equivalent | 28 / 29 | 28 / 29 |
| Task completion | 26 / 29 | 25 / 29 |
| Total tool calls | 35 | 34 |
| Match scores identical across engines | yes (6 scenarios with scores) | |
| Termination statuses observed | completed, provider_error, max_model_turns_exceeded, max_tool_calls_exceeded | + invalid_tool_requests |

The 3 manual non-completions are the provider-error, model-turn-limit and tool-call-limit
scenarios, which are designed to fail; LangGraph additionally stops the repeated-invalid scenario.

### Concrete differences observed

1. **Repeated invalid requests.** Scenario `repeated-invalid-requests` (unknown tool, then
   invalid arguments, then malformed JSON): the manual loop executes all three rejected calls and
   completes on turn 4. LangGraph stops after two consecutive all-invalid turns with
   `invalid_tool_requests`. This is the only behavioural difference and is intentional: the
   manual loop has no guard against a model that cannot form a valid call except the overall
   turn budget.
2. **Conversation continuation.** The manual loop is single-shot. The graph can continue a
   thread, and it repairs histories left with unanswered tool calls.
3. **Observability of control flow.** Graph results include `path` (visited nodes), so a test or
   report can assert the route taken, for example that a CV question never visits job search.
4. **Additional statuses.** `GraphRunResult.status` extends the manual statuses with
   `invalid_tool_requests` and `graph_recursion_limit`.

Identical in every compared scenario: tool selection replay results (all 18 cases, field by
field), retry counts, error codes, the provider-error and limit statuses and messages, and the
deterministic scores.

### Live check (single run, not statistical evidence)

One live gpt-4o-mini run of the reference request through `scripts.run_agent --agent langgraph`
on the synthetic CV completed in 3 model turns (`search_jobs → calculate_match`). The five quoted
scores and their order matched a direct deterministic recomputation, and no ungrounded scores were
flagged. In one place the model described an *unknown* requirement as "lacking"; the prompt says
unknown means "not evidenced". Prose quality was not evaluated.

## When to use which

LangGraph is useful here when:
- the route depends on model decisions and must still be bounded and testable per step;
- conversations continue across HTTP requests (`thread_id`);
- new guarded branches are expected later (human approval before `save_job`, interrupts), where
  explicit nodes are easier to extend than nested loop conditions.

A deterministic pipeline is preferable when the steps are known in advance. `POST /analyze-job`
is such a case: it calls `calculate_match` directly, with no LLM and no graph, because there is
nothing for a model to decide. Batch evaluations and scoring should also stay plain functions.

The manual loop remains the right reference for debugging provider behaviour: it has no
framework layer between the provider call and the tool call.
