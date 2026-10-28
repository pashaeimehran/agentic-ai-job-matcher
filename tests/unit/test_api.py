"""FastAPI integration tests with TestClient, fixture data and scripted models (no live API calls)."""
import pytest
from fastapi.testclient import TestClient

from src.api import build_services, create_app
from src.graph import GraphConfig
from src.matching.service import analyze_jobs
from src.rag.embeddings import HashEmbeddings
from src.rag.service import CVService
from src.rag.store import MemoryVectorStore
from src.tools import default_registry
from src.tools.context import ToolContext, build_offline_context
from src.tools.jobs import FixtureJobCatalog
from src.tools.registry import Tool
from src.tools.saved_jobs import MemorySavedJobStore
from src.tools.schemas import SearchJobsArgs
from tests.unit.agent_fakes import ScriptedProvider, answer, call, observations, tools


class SpyRegistry:
    """Wraps the default registry and records which shared tools the API invoked."""

    def __init__(self, registry=None):
        self.inner, self.executed = registry or default_registry(), []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def execute(self, name, *args, **kwargs):
        self.executed.append(name)
        return self.inner.execute(name, *args, **kwargs)


@pytest.fixture
def ctx():
    return build_offline_context()


def client_for(ctx=None, provider=None, registry=None, **kwargs):
    services = build_services(context=ctx or build_offline_context(), provider=provider, registry=registry,
                              config=GraphConfig(**kwargs))
    return TestClient(create_app(services), raise_server_exceptions=False), services


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code and body["request_id"] == response.headers["X-Request-ID"]
    assert "Traceback" not in response.text
    return body


def registry_with_search(handler):
    registry = default_registry()
    registry._tools["search_jobs"] = Tool("search_jobs", "injected", SearchJobsArgs, handler)
    return registry


def empty_cv_context():
    def no_profile():
        raise AssertionError("profile must not be loaded without a CV")
    return ToolContext(catalog=FixtureJobCatalog(), cv_service=CVService(HashEmbeddings(), MemoryVectorStore()),
                       cv_id="missing", profile_loader=no_profile, saved_jobs=MemorySavedJobStore())


# --- /health --------------------------------------------------------------------

def test_health_without_llm(ctx):
    client, _ = client_for(ctx)
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chat_available"] is False and body["agent"] == "langgraph"
    assert body["tools"] == default_registry().names() and body["catalog_jobs"] == len(ctx.catalog.all())


def test_health_reports_chat_available_with_provider(ctx):
    client, _ = client_for(ctx, provider=ScriptedProvider([]))
    assert client.get("/health").json()["chat_available"] is True


def test_request_id_is_echoed_or_generated(ctx):
    client, _ = client_for(ctx)
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"
    generated = client.get("/health", headers={"X-Request-ID": "bad id with spaces"}).headers["X-Request-ID"]
    assert generated != "bad id with spaces" and len(generated) == 12


def test_unknown_route_uses_error_contract(ctx):
    client, _ = client_for(ctx)
    assert_error(client.get("/nope"), 404, "not_found")
    assert_error(client.delete("/health"), 405, "method_not_allowed")


# --- /jobs ----------------------------------------------------------------------

def test_list_jobs_paginates_catalog(ctx):
    client, _ = client_for(ctx)
    body = client.get("/jobs", params={"offset": 2, "limit": 3}).json()
    assert body["total"] == len(ctx.catalog.all()) and len(body["jobs"]) == 3
    assert [j["job_id"] for j in body["jobs"]] == [j.job_id for j in ctx.catalog.all()[2:5]]


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "many"}])
def test_list_jobs_validation(ctx, params):
    client, _ = client_for(ctx)
    body = assert_error(client.get("/jobs", params=params), 422, "validation_error")
    assert body["error"]["details"][0]["loc"][0] == "query"


def test_job_details_via_shared_tool(ctx):
    spy = SpyRegistry()
    client, _ = client_for(ctx, registry=spy)
    body = client.get("/jobs/match-03-robotics").json()
    assert body["job"]["job_id"] == "match-03-robotics" and spy.executed == ["get_job_details"]


def test_job_details_unknown_job(ctx):
    client, _ = client_for(ctx)
    body = assert_error(client.get("/jobs/no-such-job"), 404, "job_not_found")
    assert body["error"]["details"] == [{"tool_error_code": "not_found"}]
    assert_error(client.get("/jobs/" + "x" * 121), 422, "validation_error")


# --- /search --------------------------------------------------------------------

def test_search_jobs_matches_tool_output(ctx):
    spy = SpyRegistry()
    client, _ = client_for(ctx, registry=spy)
    body = client.post("/search", json={"target": "jobs", "query": "AI Engineer", "location": "Germany"}).json()
    direct = default_registry().execute("search_jobs", {"query": "AI Engineer", "location": "Germany"}, ctx)
    assert body["status"] == "ok" and body["jobs"] == direct.data["jobs"] and spy.executed == ["search_jobs"]


def test_search_cv(ctx):
    client, _ = client_for(ctx)
    body = client.post("/search", json={"target": "cv", "query": "ROS2", "k": 2}).json()
    assert body["target"] == "cv" and body["status"] == "ok" and 1 <= len(body["evidence"]) <= 2


def test_empty_search_is_a_successful_empty_result(ctx):
    client, _ = client_for(ctx)
    response = client.post("/search", json={"target": "jobs", "query": "Underwater Welder", "location": "Munich"})
    assert response.status_code == 200
    assert response.json()["status"] == "empty" and response.json()["jobs"] == [] and response.json()["message"]


@pytest.mark.parametrize("payload", [
    {"target": "jobs"},                                   # missing query
    {"target": "jobs", "query": "AI", "limit": 50},       # limit above tool maximum
    {"target": "jobs", "query": "AI", "limit": "5"},      # no coercion (tool schema is strict)
    {"target": "jobs", "query": "AI", "cv_id": "other"},  # CV scope is not client-selectable
    {"target": "cv", "query": "Python", "k": 11},
    {"target": "people", "query": "AI"},                  # unknown target
])
def test_search_validation(ctx, payload):
    client, _ = client_for(ctx)
    assert_error(client.post("/search", json=payload), 422, "validation_error")


def test_search_transient_failure_maps_to_503(ctx):
    def down(args, ctx):
        raise TimeoutError()

    client, _ = client_for(ctx, registry=registry_with_search(down))
    body = assert_error(client.post("/search", json={"target": "jobs", "query": "AI"}), 503, "dependency_unavailable")
    assert body["error"]["retryable"] is True


def test_search_tool_failure_hides_exception_text(ctx):
    def explode(args, ctx):
        raise RuntimeError("password=hunter2")

    client, _ = client_for(ctx, registry=registry_with_search(explode))
    response = client.post("/search", json={"target": "jobs", "query": "AI"})
    assert_error(response, 500, "tool_failure")
    assert "hunter2" not in response.text and "RuntimeError" not in response.text


# --- /analyze-job ---------------------------------------------------------------

def test_analyze_job_returns_deterministic_score(ctx):
    spy = SpyRegistry()
    client, _ = client_for(ctx, registry=spy)
    body = client.post("/analyze-job", json={"job_id": "fixture-01"}).json()
    expected = analyze_jobs([ctx.catalog.get("fixture-01").requirements()], ctx.cv_id, ctx.cv_service,
                            profile=ctx.profile())[0]
    assert body["score_source"] == "deterministic" and spy.executed == ["calculate_match"]
    assert body["match"]["score"] == expected.score.model_dump(mode="json")


@pytest.mark.parametrize("payload", [{}, {"job_id": ""}, {"job_id": "fixture-01", "score": 100},
                                     {"job_ids": ["fixture-01"]}])
def test_analyze_job_validation(ctx, payload):
    client, _ = client_for(ctx)
    assert_error(client.post("/analyze-job", json=payload), 422, "validation_error")


def test_analyze_job_unknown_job(ctx):
    client, _ = client_for(ctx)
    assert_error(client.post("/analyze-job", json={"job_id": "no-such-job"}), 404, "job_not_found")


def test_analyze_job_missing_cv():
    client, _ = client_for(empty_cv_context())
    body = assert_error(client.post("/analyze-job", json={"job_id": "fixture-01"}), 409, "cv_not_available")
    assert body["error"]["retryable"] is False


# --- /chat ----------------------------------------------------------------------

def test_chat_runs_langgraph_agent(ctx):
    def explain(messages):
        match = observations(messages)[-1]["data"]["matches"][0]
        return answer(f"{match['job_id']} scores {match['score']['value']}/100.")

    provider = ScriptedProvider([tools(call("calculate_match", {"job_ids": ["fixture-01"]})), explain])
    client, _ = client_for(ctx, provider=provider)
    body = client.post("/chat", json={"message": "How well do I fit fixture-01?"}).json()
    assert body["status"] == "completed" and body["run_id"] and body["thread_id"]
    assert [t["tool"] for t in body["tool_calls"]] == ["calculate_match"] and body["ungrounded_scores"] == []
    assert "data" not in body["tool_calls"][0]  # tool payloads (CV excerpts) are not returned


def test_chat_thread_continuation(ctx):
    provider = ScriptedProvider([answer("Hello."), answer("Still here.")])
    client, _ = client_for(ctx, provider=provider)
    first = client.post("/chat", json={"message": "hi"}).json()
    second = client.post("/chat", json={"message": "again", "thread_id": first["thread_id"]}).json()
    assert second["thread_id"] == first["thread_id"] and len(provider.calls[1]) == 4


def test_reused_request_id_never_joins_an_existing_thread(ctx):
    provider = ScriptedProvider([answer("one"), answer("two")])
    client, _ = client_for(ctx, provider=provider)
    headers = {"X-Request-ID": "same-id"}
    first = client.post("/chat", json={"message": "private first message"}, headers=headers).json()
    second = client.post("/chat", json={"message": "hi"}, headers=headers).json()
    assert first["thread_id"] != second["thread_id"] and first["run_id"] != second["run_id"]
    assert len(provider.calls[1]) == 2  # system + user only: no history from the first request


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "x" * 4001},
                                     {"message": "hi", "thread_id": "bad id"}, {"message": "hi", "cv_id": "x"}])
def test_chat_validation(ctx, payload):
    client, _ = client_for(ctx, provider=ScriptedProvider([]))
    assert_error(client.post("/chat", json=payload), 422, "validation_error")


def test_chat_unavailable_without_provider(ctx):
    client, _ = client_for(ctx)
    assert_error(client.post("/chat", json={"message": "hi"}), 503, "chat_unavailable")


def test_chat_provider_failure(ctx):
    class Broken:
        def complete(self, messages, tools):
            raise ConnectionError("sk-secret")

    client, _ = client_for(ctx, provider=Broken())
    response = client.post("/chat", json={"message": "hi"})
    body = assert_error(response, 502, "llm_provider_error")
    assert body["run_id"] and body["error"]["retryable"] is True and "sk-secret" not in response.text


def test_chat_graph_limit(ctx):
    looping = [tools(call("search_jobs", {"query": f"AI {i}"})) for i in range(5)]
    client, _ = client_for(ctx, provider=ScriptedProvider(looping), max_model_turns=2)
    body = assert_error(client.post("/chat", json={"message": "loop"}), 422, "agent_limit_exceeded")
    assert body["error"]["details"] == [{"agent_status": "max_model_turns_exceeded"}]


def test_unexpected_exception_is_internal_error(ctx):
    class Exploding(SpyRegistry):
        def execute(self, *args, **kwargs):
            raise KeyError("cv text should not leak")

    client, _ = client_for(ctx, registry=Exploding())
    response = client.post("/analyze-job", json={"job_id": "fixture-01"})
    assert_error(response, 500, "internal_error")
    assert "cv text" not in response.text


def test_app_builds_services_at_startup_when_not_injected(ctx):
    built = []

    def factory():
        built.append(1)
        return build_services(context=ctx)

    with TestClient(create_app(services_factory=factory)) as client:
        assert client.get("/health").status_code == 200
    assert built == [1]


def test_openapi_documents_error_contract(ctx):
    client, _ = client_for(ctx)
    spec = client.get("/openapi.json").json()
    assert {"/health", "/jobs", "/jobs/{job_id}", "/search", "/analyze-job", "/chat"} <= set(spec["paths"])
    assert "ErrorResponse" in spec["components"]["schemas"]
