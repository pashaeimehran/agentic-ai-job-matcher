"""Environment configuration: defaults, mode selection, validation and secret hygiene."""
import pytest

from src.config import ConfigurationError, Settings


def test_defaults_are_fixture_mode_without_keys():
    settings = Settings.from_env({})
    assert settings.app_mode == "fixture" and settings.embeddings == "hash"
    assert settings.database_url is None and not settings.tracing_active and settings.llm_cache_enabled
    assert settings.agent_max_model_turns == 8 and settings.agent_max_tool_calls == 12


def test_auto_mode_selects_live_when_a_key_is_present():
    settings = Settings.from_env({"OPENAI_API_KEY": "sk-test-123"})
    assert settings.app_mode == "live" and settings.embeddings == "openai"
    assert Settings.from_env({"OPENAI_API_KEY": "sk-test-123", "APP_MODE": "fixture"}).app_mode == "fixture"


def test_live_mode_requires_a_key():
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        Settings.from_env({"APP_MODE": "live"})


def test_values_are_parsed_and_validated():
    settings = Settings.from_env({"AGENT_MAX_MODEL_TURNS": "4", "LLM_CACHE_ENABLED": "false",
                                  "AGENT_MAX_RUN_TOKENS": "none", "LLM_CACHE_OPERATIONS": "generate",
                                  "LANGFUSE_BASE_URL": "http://langfuse:3000", "EMBEDDING_PROVIDER": "hash"})
    assert settings.agent_max_model_turns == 4 and not settings.llm_cache_enabled
    assert settings.agent_max_run_tokens is None and settings.llm_cache_operations == {"generate"}
    assert settings.langfuse_host == "http://langfuse:3000"
    for bad in ({"AGENT_MAX_MODEL_TURNS": "0"}, {"LLM_CACHE_ENABLED": "maybe"}, {"APP_MODE": "prod"},
                {"LLM_CACHE_OPERATIONS": "embed"}, {"TRACING_ENABLED": "true"}):
        with pytest.raises(ConfigurationError):
            Settings.from_env(bad)


def test_tracing_turns_on_with_both_langfuse_keys():
    settings = Settings.from_env({"LANGFUSE_PUBLIC_KEY": "pk-lf-1", "LANGFUSE_SECRET_KEY": "sk-lf-1"})
    assert settings.tracing_active and not settings.langfuse_capture_content


def test_secrets_never_appear_in_repr_summary_or_errors():
    env = {"OPENAI_API_KEY": "sk-live-SECRET1", "DATABASE_URL": "postgresql://u:PASSWORD2@db/x",
           "LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk-lf-SECRET3"}
    settings = Settings.from_env(env)
    text = repr(settings) + str(settings.public_summary()) + settings.model_dump_json()
    for secret in ("SECRET1", "PASSWORD2", "SECRET3"):
        assert secret not in text
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({**env, "AGENT_MAX_TOOL_CALLS": "-1"})
    assert all(secret not in str(error.value) for secret in ("SECRET1", "PASSWORD2", "SECRET3"))
    assert "agent_max_tool_calls" in str(error.value)


def test_env_example_lists_only_placeholders():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
    names = {line.split("=", 1)[0] for line in text.splitlines() if "=" in line and not line.startswith("#")}
    assert {"APP_MODE", "OPENAI_API_KEY", "DATABASE_URL", "LANGFUSE_SECRET_KEY", "LLM_CACHE_ENABLED"} <= names
    assert "sk-" not in text.replace("sk-lf-replace_me", "").replace("pk-lf-replace_me", "")
    Settings.from_env({k: v for k, v in (line.split("=", 1) for line in text.splitlines()
                                         if "=" in line and not line.startswith("#"))
                       if k not in {"OPENAI_API_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "APP_MODE"}})
