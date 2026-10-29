"""Zero-key fixture mode: the real API, LangGraph agent and tools without an LLM provider or network."""
import json

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.dependencies import build_default_services
from src.config import Settings
from src.graph import GraphConfig, LangGraphAgent
from src.llm.fixture_provider import FixtureToolCallingProvider
from src.matching.service import analyze_jobs
from src.tools import default_registry
from src.tools.context import build_offline_context


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("fixture mode must not construct an OpenAI client")
    monkeypatch.setattr("openai.OpenAI", no_network)
    settings = Settings.from_env({"APP_MODE": "fixture", "SAVED_JOBS_PATH": str(tmp_path / "saved.json")})
    with TestClient(create_app(services_factory=lambda: build_default_services(settings))) as test_client:
        yield test_client


def test_api_starts_and_is_ready_without_credentials(client):
    health = client.get("/health").json()
    assert health["mode"] == "fixture" and health["chat_available"] is True
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    assert ready.json()["checks"]["chat_provider"]["detail"] == "deterministic fixture provider"


def test_job_search_and_deterministic_matching(client):
    search = client.post("/search", json={"target": "jobs", "query": "AI Engineer", "location": "Germany"}).json()
    assert search["status"] == "ok" and search["jobs"]
    match = client.post("/analyze-job", json={"job_id": "match-03-robotics"}).json()
    expected = analyze_jobs([build_offline_context().catalog.get("match-03-robotics").requirements()],
                            "synthetic-demo", build_offline_context().cv_service,
                            profile=build_offline_context().profile())[0]
    assert match["match"]["score"]["value"] == expected.score.value and match["match"]["unmet"] == ["German B2"]


def test_chat_runs_the_real_graph_and_tools(client):
    body = client.post("/chat", json={"message": "Find AI Engineer jobs in Germany that fit my CV"}).json()
    assert [t["tool"] for t in body["tool_calls"]] == ["search_jobs", "calculate_match"]
    assert body["ungrounded_scores"] == [] and "Fixture mode" in body["answer"]
    assert "not a confirmed gap" in body["answer"]


def test_saved_job_uses_configured_path(client, tmp_path):
    body = client.post("/chat", json={"message": "Save fixture-13"}).json()
    assert body["tool_calls"][0]["tool"] == "save_job"
    assert json.loads((tmp_path / "saved.json").read_text())[0]["job_id"] == "fixture-13"


@pytest.mark.parametrize("request_text, sequence", [
    ("Find AI Engineer jobs in Germany that fit my CV", ["search_jobs", "calculate_match"]),
    ("Search for Machine Learning Engineer jobs in Hamburg", ["search_jobs"]),
    ("Show details for match-03-robotics", ["get_job_details"]),
    ("How well do I fit match-03-robotics?", ["calculate_match"]),
    ("Compare fixture-21 and fixture-05", ["calculate_match"]),
    ("What is my German level in my CV?", ["search_cv"]),
    ("Save fixture-13 for later", ["save_job"]),
    ("Hello", []),
])
def test_fixture_provider_routes_requests(request_text, sequence):
    agent = LangGraphAgent(FixtureToolCallingProvider(), default_registry(), build_offline_context(), GraphConfig())
    result = agent.run(request_text)
    assert result.status == "completed" and result.tool_sequence == sequence


def test_fixture_provider_reports_empty_results_and_errors():
    agent = LangGraphAgent(FixtureToolCallingProvider(), default_registry(), build_offline_context(), GraphConfig())
    assert "No jobs" in agent.run("Find Underwater Welder jobs in Munich").final_answer
    assert "error" in agent.run("How well do I fit fixture-99?").final_answer
