"""MCP smoke tests: an MCP client calls the server, which must reach the shared tool registry.

In-process tests use mcp.Client(server); one test launches the real stdio entry point.
No external MCP host (desktop app, IDE) is needed.
"""
import asyncio
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from src.mcp_server import DEFAULT_EXPOSED, build_mcp_server
from src.tools import default_registry
from src.tools.context import build_offline_context

ROOT = Path(__file__).resolve().parents[2]

CALLS = {
    "search_jobs": {"query": "AI Engineer", "location": "Germany", "limit": 3},
    "get_job_details": {"job_id": "match-03-robotics"},
    "search_cv": {"query": "ROS2", "k": 2},
    "calculate_match": {"job_ids": ["fixture-01", "fixture-21"]},
    "save_job": {"job_id": "fixture-13", "note": "apply next week"},
}


class SpyRegistry:
    def __init__(self):
        self.inner, self.executed = default_registry(), []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def execute(self, name, raw_arguments, *args, **kwargs):
        self.executed.append((name, raw_arguments))
        return self.inner.execute(name, raw_arguments, *args, **kwargs)


def run(coro_factory, server):
    async def main():
        async with Client(server) as client:
            return await coro_factory(client)
    return asyncio.run(main())


@pytest.fixture
def ctx():
    return build_offline_context()


def test_lists_exactly_the_registry_schemas(ctx):
    registry = default_registry()
    tools = run(lambda c: c.list_tools(), build_mcp_server(registry, ctx)).tools
    assert [t.name for t in tools] == list(DEFAULT_EXPOSED) == registry.names()
    specs = {s.name: s for s in registry.specs()}
    for tool in tools:
        assert tool.input_schema == specs[tool.name].parameters
        assert tool.description == specs[tool.name].description
    hints = {t.name: t.annotations.read_only_hint for t in tools}
    assert hints == {"search_jobs": True, "get_job_details": True, "search_cv": True,
                     "calculate_match": True, "save_job": False}


@pytest.mark.parametrize("name", list(CALLS))
def test_every_tool_reaches_the_shared_registry_and_matches_direct_execution(name):
    spy = SpyRegistry()
    mcp_ctx, direct_ctx = build_offline_context(), build_offline_context()
    result = run(lambda c: c.call_tool(name, CALLS[name]), build_mcp_server(spy, mcp_ctx))
    assert spy.executed == [(name, CALLS[name])]
    assert result.is_error is False and result.structured_content["status"] == "ok"
    direct = default_registry().execute(name, CALLS[name], direct_ctx).model_dump(mode="json")
    data, expected = result.structured_content["data"], direct["data"]
    if name == "save_job":  # timestamps differ between the two stores
        data, expected = {**data, "saved_at": None}, {**expected, "saved_at": None}
    assert data == expected


def test_calculate_match_scores_are_the_deterministic_scores(ctx):
    from src.matching.service import analyze_jobs
    result = run(lambda c: c.call_tool("calculate_match", CALLS["calculate_match"]),
                 build_mcp_server(default_registry(), ctx))
    expected = analyze_jobs([ctx.catalog.get(i).requirements() for i in CALLS["calculate_match"]["job_ids"]],
                            ctx.cv_id, ctx.cv_service, profile=ctx.profile())
    assert [(m["job_id"], m["score"]["value"]) for m in result.structured_content["data"]["matches"]] == \
        [(a.job_id, a.score.value) for a in expected]


def test_save_job_persists_in_the_shared_store(ctx):
    run(lambda c: c.call_tool("save_job", {"job_id": "fixture-13"}), build_mcp_server(default_registry(), ctx))
    assert [j.job_id for j in ctx.saved_jobs.list()] == ["fixture-13"]


@pytest.mark.parametrize("arguments, code", [
    ({"query": "AI", "limit": "2"}, "invalid_arguments"),   # strict: no string-to-int coercion
    ({"query": "AI", "cv_id": "someone-else"}, "invalid_arguments"),
    ({}, "invalid_arguments"),
])
def test_application_validation_is_enforced(ctx, arguments, code):
    result = run(lambda c: c.call_tool("search_jobs", arguments), build_mcp_server(default_registry(), ctx))
    assert result.is_error is True and result.structured_content["error"]["code"] == code
    assert result.structured_content["attempts"] == 0  # handler never ran


def test_domain_errors_are_structured(ctx):
    result = run(lambda c: c.call_tool("get_job_details", {"job_id": "no-such-job"}),
                 build_mcp_server(default_registry(), ctx))
    assert result.is_error is True and result.structured_content["error"]["code"] == "not_found"


def test_unexposed_and_unknown_tools_are_rejected(ctx):
    server = build_mcp_server(default_registry(), ctx, exposed=("search_jobs",))

    async def calls(client):
        listed = [t.name for t in (await client.list_tools()).tools]
        return listed, await client.call_tool("save_job", {"job_id": "fixture-13"}), \
            await client.call_tool("drop_tables", {})

    listed, hidden, unknown = run(calls, server)
    assert listed == ["search_jobs"]
    for result in (hidden, unknown):
        assert result.structured_content["error"]["code"] == "unknown_tool"
        assert result.structured_content["error"]["message"].endswith("Available: search_jobs.")
    assert ctx.saved_jobs.list() == []


def test_exposing_an_unregistered_tool_fails_fast(ctx):
    with pytest.raises(ValueError):
        build_mcp_server(default_registry(), ctx, exposed=("search_jobs", "delete_cv"))


def test_stdio_entry_point_serves_the_tools():
    params = StdioServerParameters(command=sys.executable, args=["-m", "scripts.serve_mcp", "--memory-saved-jobs"],
                                   cwd=str(ROOT))

    async def main():
        async with Client(params) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            result = await client.call_tool("get_job_details", {"job_id": "fixture-01"})
            return names, result

    names, result = asyncio.run(asyncio.wait_for(main(), timeout=60))
    assert names == list(DEFAULT_EXPOSED)
    assert result.structured_content["data"]["job_id"] == "fixture-01"
