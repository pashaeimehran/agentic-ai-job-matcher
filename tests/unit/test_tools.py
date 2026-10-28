"""Tool layer tests: run offline against the synthetic fixture CV and job catalog."""
import json

import pytest

from src.matching.fixtures import fixture_jobs
from src.matching.scoring import score_match
from src.matching.service import analyze_jobs
from src.tools import default_registry
from src.tools.context import build_offline_context
from src.tools.errors import TransientToolError, classify_exception
from src.tools.registry import RetryPolicy, Tool, ToolRegistry
from src.tools.saved_jobs import JsonFileSavedJobStore
from src.tools.schemas import SearchJobsArgs, ToolResult


@pytest.fixture(scope="module")
def ctx():
    return build_offline_context()


@pytest.fixture(scope="module")
def registry():
    return default_registry()


def run(registry, ctx, name, args):
    return registry.execute(name, json.dumps(args) if isinstance(args, dict) else args, ctx)


def test_registry_exposes_exactly_the_five_tools_with_strict_schemas(registry):
    assert registry.names() == ["search_jobs", "get_job_details", "search_cv", "calculate_match", "save_job"]
    for spec in registry.specs():
        assert spec.parameters["type"] == "object"
        assert spec.parameters["additionalProperties"] is False
    # CV scoping belongs to the application, not to model arguments.
    assert all("cv_id" not in s.parameters["properties"] for s in registry.specs())


def test_search_jobs_filters_location_and_ranks_deterministically(registry, ctx):
    out = run(registry, ctx, "search_jobs", {"query": "AI Engineer", "location": "Germany", "limit": 5})
    assert out.status == "ok"
    jobs = out.data["jobs"]
    assert len(jobs) == 5 and jobs[0]["job_id"] == "fixture-01"
    assert all("AI" in j["role"] for j in jobs)
    assert jobs == run(registry, ctx, "search_jobs", {"query": "AI Engineer", "location": "Germany",
                                                      "limit": 5}).data["jobs"]
    berlin = run(registry, ctx, "search_jobs", {"query": "Engineer", "location": "berlin", "limit": 20})
    assert {j["location"] for j in berlin.data["jobs"]} == {"Berlin"}


def test_search_jobs_catalog_collapses_duplicate_phrasing_variants(ctx):
    ids = [job.job_id for job in ctx.catalog.all()]
    assert len(ids) == len(set(ids)) == 16
    assert "fixture-02" not in ids  # identical structured posting to fixture-01


def test_search_jobs_empty_result_is_explicit(registry, ctx):
    out = run(registry, ctx, "search_jobs", {"query": "Underwater Basket Weaver"})
    assert out.status == "empty" and out.error is None
    assert out.data["jobs"] == [] and "No jobs" in out.data["message"]
    out = run(registry, ctx, "search_jobs", {"query": "AI Engineer", "location": "Paris"})
    assert out.status == "empty"


def test_get_job_details_returns_existing_structured_schema(registry, ctx):
    out = run(registry, ctx, "get_job_details", {"job_id": "match-03-robotics"})
    assert out.status == "ok"
    assert set(out.data["posting"]) == {"role", "skills", "experience_years", "language", "seniority", "location"}
    assert out.data["language_levels"] == {"German": "B2"}
    missing = run(registry, ctx, "get_job_details", {"job_id": "does-not-exist"})
    assert missing.status == "error" and missing.error.code == "not_found" and not missing.error.retryable


def test_search_cv_wraps_scoped_hybrid_retrieval(registry, ctx):
    out = run(registry, ctx, "search_cv", {"query": "Python experience", "k": 2})
    assert out.status == "ok" and len(out.data["evidence"]) == 2
    chunk_ids = {c.chunk_id for c in ctx.cv_service.store.list_chunks(ctx.cv_id)}
    for item in out.data["evidence"]:
        assert item["chunk_id"] in chunk_ids and item["text"]
    assert "Python" in out.data["evidence"][0]["text"]


def test_search_cv_empty_when_section_has_no_chunks(registry, ctx):
    out = run(registry, ctx, "search_cv", {"query": "Python", "section": "No Such Section"})
    assert out.status == "empty" and out.data["evidence"] == []


def test_search_cv_cannot_escape_cv_scope(registry, ctx):
    out = run(registry, ctx, "search_cv", {"query": "Python", "cv_id": "someone-else"})
    assert out.status == "error" and out.error.code == "invalid_arguments"
    assert out.error.details[0]["type"] == "extra_forbidden"


def test_calculate_match_scores_equal_existing_deterministic_matching(registry, ctx):
    ids = ["match-03-robotics", "match-01-python-backend", "fixture-01"]
    out = run(registry, ctx, "calculate_match", {"job_ids": ids})
    assert out.status == "ok"
    expected = analyze_jobs([ctx.catalog.get(i).requirements() for i in ids], ctx.cv_id,
                            ctx.cv_service, profile=ctx.profile())
    assert [m["job_id"] for m in out.data["matches"]] == [a.job_id for a in expected]
    for match, analysis in zip(out.data["matches"], expected):
        assert match["score"] == analysis.score.model_dump(mode="json")
    robotics = next(m for m in out.data["matches"] if m["job_id"] == "match-03-robotics")
    assert robotics["unmet"] == ["German B2"] and robotics["supported"] == ["ROS 2", "C++"]
    # Same job definitions as the matching layer's fixtures.
    job = next(j for j in fixture_jobs() if j.job_id == "match-03-robotics")
    assert ctx.catalog.get("match-03-robotics").requirements() == job
    assert score_match(job, ctx.profile())[1] == analysis_comparisons(robotics)


def analysis_comparisons(match):
    from src.matching.schemas import RequirementResult
    return [RequirementResult.model_validate(c) for c in match["comparisons"]]


def test_calculate_match_is_repeatable(registry, ctx):
    args = {"job_ids": ["fixture-01", "fixture-09", "match-02-ml-engineer"]}
    assert run(registry, ctx, "calculate_match", args).data == run(registry, ctx, "calculate_match", args).data


@pytest.mark.parametrize("args, error_type", [
    ({"job_ids": []}, "too_short"),
    ({"job_ids": ["a"] * 11}, "too_long"),
    ({"job_ids": ["fixture-01", "fixture-01"]}, "value_error"),
    ({"job_ids": "fixture-01"}, "list_type"),
    ({"job_ids": ["fixture-01"], "score": 99}, "extra_forbidden"),
])
def test_calculate_match_rejects_malformed_arguments(registry, ctx, args, error_type):
    out = run(registry, ctx, "calculate_match", args)
    assert out.status == "error" and out.error.code == "invalid_arguments" and out.attempts == 0
    assert error_type in {d["type"] for d in out.error.details}


def test_calculate_match_unknown_job_and_missing_cv(registry, ctx):
    out = run(registry, ctx, "calculate_match", {"job_ids": ["nope"]})
    assert out.error.code == "not_found"
    empty_ctx = build_offline_context(cv_id="other-cv")
    empty_ctx.cv_service.store.delete("other-cv")
    out = run(registry, empty_ctx, "calculate_match", {"job_ids": ["fixture-01"]})
    assert out.error.code == "precondition_failed" and not out.error.retryable


def test_save_job_is_idempotent_and_validates_job(registry, tmp_path):
    ctx = build_offline_context(saved_jobs=JsonFileSavedJobStore(tmp_path / "saved.json"))
    first = run(registry, ctx, "save_job", {"job_id": "fixture-01", "note": "Apply this week"})
    second = run(registry, ctx, "save_job", {"job_id": "fixture-01"})
    assert first.data["status"] == "saved" and second.data["status"] == "already_saved"
    stored = json.loads((tmp_path / "saved.json").read_text(encoding="utf-8"))
    assert stored == [{"job_id": "fixture-01", "note": "Apply this week", "saved_at": first.data["saved_at"]}]
    assert run(registry, ctx, "save_job", {"job_id": "ghost"}).error.code == "not_found"
    assert not (tmp_path / "saved.json.tmp").exists()


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"query": 5}', '{"query": ""}',
                                 '{"query": "AI", "limit": 0}', '{"query": "AI", "limit": "5"}',
                                 '{"query": "AI", "limit": 5.5}', '{"location": "Berlin"}',
                                 '{"query": "AI", "__class__": "x"}'])
def test_malformed_arguments_never_reach_the_implementation(raw):
    calls = []
    registry = ToolRegistry([Tool("search_jobs", "spy", SearchJobsArgs,
                                  lambda args, ctx: calls.append(args) or ToolResult())])
    out = registry.execute("search_jobs", raw, context=None)
    assert out.status == "error" and out.error.code == "invalid_arguments"
    assert calls == []


def test_valid_arguments_reach_implementation_as_typed_model():
    seen = []
    registry = ToolRegistry([Tool("search_jobs", "spy", SearchJobsArgs,
                                  lambda args, ctx: seen.append(args) or ToolResult())])
    assert registry.execute("search_jobs", {"query": "  AI  "}, context=None).status == "ok"
    assert isinstance(seen[0], SearchJobsArgs) and seen[0].query == "AI" and seen[0].limit == 5


def test_unknown_tool_is_not_executed():
    out = default_registry().execute("delete_all_jobs", "{}", context=None)
    assert out.error.code == "unknown_tool" and out.attempts == 0


def test_retry_only_for_retryable_failures_and_bounded():
    attempts = {"transient": 0, "value": 0}

    def transient(args, ctx):
        attempts["transient"] += 1
        raise TransientToolError("backend busy")

    def value(args, ctx):
        attempts["value"] += 1
        raise ValueError("secret CV text inside exception")

    registry = ToolRegistry([Tool("t", "t", SearchJobsArgs, transient), Tool("v", "v", SearchJobsArgs, value)])
    out = registry.execute("t", {"query": "x"}, None, RetryPolicy(max_retries=2))
    assert out.attempts == 3 == attempts["transient"] and out.error.code == "transient" and out.error.retryable
    out = registry.execute("v", {"query": "x"}, None, RetryPolicy(max_retries=2))
    assert out.attempts == 1 == attempts["value"] and out.error.code == "internal"
    assert "secret" not in out.error.message


def test_exception_classification():
    assert classify_exception(TimeoutError())[:2] == ("transient", True)
    assert classify_exception(ConnectionError())[:2] == ("transient", True)
    from src.data_sources.ba_jobsuche import BAJobsucheUnavailable
    assert classify_exception(BAJobsucheUnavailable("x"))[:2] == ("transient", True)
    assert classify_exception(KeyError("x"))[:2] == ("internal", False)


def test_tool_logs_are_structured_and_omit_arguments(registry, ctx, caplog):
    caplog.set_level("INFO", logger="src.tools")
    registry.execute("search_cv", {"query": "very private question"}, ctx, run_id="r1", step=2)
    registry.execute("search_cv", {"query": 3}, ctx, run_id="r1", step=3)
    events = [json.loads(r.getMessage()) for r in caplog.records if r.name == "src.tools"]
    assert [e["event"] for e in events] == ["tool_succeeded", "tool_rejected"]
    assert events[0]["run_id"] == "r1" and events[0]["step"] == 2 and "duration_ms" in events[0]
    assert events[1]["arguments_valid"] is False
    assert "private" not in caplog.text and "Python" not in caplog.text
