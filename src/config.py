"""Runtime configuration from environment variables.

Every setting has a safe default for local fixture mode. Secrets are SecretStr,
so they never appear in repr(), logs or error messages. See .env.example for
the full list with placeholders.
"""
from __future__ import annotations

import os
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

AppMode = Literal["fixture", "live"]


class ConfigurationError(ValueError):
    """Invalid or incomplete runtime configuration; messages name variables, never values."""


def _bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    app_env: str = "development"
    app_mode: AppMode = "fixture"

    # --- LLM provider -------------------------------------------------------------
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_max_output_tokens: int = Field(default=1200, ge=16, le=32_000)
    openai_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_retry_backoff_seconds: float = Field(default=1.0, ge=0, le=30)
    llm_pricing_json: str | None = None
    embedding_provider: Literal["hash", "openai"] | None = None   # default: hash (fixture) / openai (live)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=1, le=2000)

    # --- Agent cost controls ------------------------------------------------------
    agent_prompt_version: str = "v2"
    agent_max_model_turns: int = Field(default=8, ge=1, le=50)
    agent_max_tool_calls: int = Field(default=12, ge=0, le=100)
    agent_max_run_tokens: int | None = Field(default=60_000, ge=1)
    agent_max_run_cost_usd: float | None = Field(default=None, gt=0)

    # --- Response cache -----------------------------------------------------------
    llm_cache_enabled: bool = True
    llm_cache_ttl_seconds: float = Field(default=3600, gt=0)
    llm_cache_max_entries: int = Field(default=512, ge=1)
    llm_cache_version: str = "v1"
    llm_cache_operations: frozenset[str] = frozenset({"generate", "complete"})

    # --- Persistence --------------------------------------------------------------
    database_url: SecretStr | None = None
    db_init_on_startup: bool = True
    cost_log_enabled: bool = True
    cv_id: str = Field(default="synthetic-demo", pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    saved_jobs_path: str | None = None   # JSON file used when no DATABASE_URL is set

    # --- Tracing (Langfuse) -------------------------------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "http://localhost:3000"
    langfuse_capture_content: bool = False
    tracing_enabled: bool | None = None   # default: on when both Langfuse keys are set

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.app_mode == "live" and self.openai_api_key is None:
            raise ValueError("APP_MODE=live requires OPENAI_API_KEY")
        if self.tracing_enabled and not (self.langfuse_public_key and self.langfuse_secret_key):
            raise ValueError("TRACING_ENABLED=true requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        unknown = self.llm_cache_operations - {"generate", "complete"}
        if unknown:
            raise ValueError("LLM_CACHE_OPERATIONS accepts only 'generate' and 'complete'")
        return self

    @property
    def tracing_active(self) -> bool:
        if self.tracing_enabled is not None:
            return self.tracing_enabled
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def embeddings(self) -> Literal["hash", "openai"]:
        return self.embedding_provider or ("openai" if self.app_mode == "live" else "hash")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ

        def get(name: str) -> str | None:
            value = env.get(name)
            return value.strip() if value is not None and value.strip() != "" else None

        raw: dict[str, object] = {}
        plain = {"APP_ENV": "app_env", "OPENAI_MODEL": "openai_model", "LLM_PRICING_JSON": "llm_pricing_json",
                 "EMBEDDING_PROVIDER": "embedding_provider", "EMBEDDING_MODEL": "embedding_model",
                 "AGENT_PROMPT_VERSION": "agent_prompt_version", "LLM_CACHE_VERSION": "llm_cache_version",
                 "CV_ID": "cv_id", "SAVED_JOBS_PATH": "saved_jobs_path", "LANGFUSE_PUBLIC_KEY": "langfuse_public_key", "LOG_LEVEL": "log_level"}
        numbers = {"OPENAI_MAX_OUTPUT_TOKENS": "openai_max_output_tokens",
                   "OPENAI_TIMEOUT_SECONDS": "openai_timeout_seconds", "LLM_MAX_RETRIES": "llm_max_retries",
                   "LLM_RETRY_BACKOFF_SECONDS": "llm_retry_backoff_seconds",
                   "EMBEDDING_DIMENSIONS": "embedding_dimensions", "AGENT_MAX_MODEL_TURNS": "agent_max_model_turns",
                   "AGENT_MAX_TOOL_CALLS": "agent_max_tool_calls",
                   "LLM_CACHE_TTL_SECONDS": "llm_cache_ttl_seconds",
                   "LLM_CACHE_MAX_ENTRIES": "llm_cache_max_entries"}
        optional_numbers = {"AGENT_MAX_RUN_TOKENS": "agent_max_run_tokens",
                            "AGENT_MAX_RUN_COST_USD": "agent_max_run_cost_usd"}
        booleans = {"LLM_CACHE_ENABLED": "llm_cache_enabled", "DB_INIT_ON_STARTUP": "db_init_on_startup",
                    "COST_LOG_ENABLED": "cost_log_enabled", "LANGFUSE_CAPTURE_CONTENT": "langfuse_capture_content",
                    "TRACING_ENABLED": "tracing_enabled"}
        secrets = {"OPENAI_API_KEY": "openai_api_key", "DATABASE_URL": "database_url",
                   "LANGFUSE_SECRET_KEY": "langfuse_secret_key"}
        try:
            for name, field in {**plain, **numbers, **secrets}.items():
                if get(name) is not None:
                    raw[field] = get(name)
            for name, field in optional_numbers.items():
                value = get(name)
                if value is not None:
                    raw[field] = None if value.lower() in {"none", "off", "0"} else value
            for name, field in booleans.items():
                if get(name) is not None:
                    raw[field] = _bool(get(name))
            host = get("LANGFUSE_HOST") or get("LANGFUSE_BASE_URL")
            if host:
                raw["langfuse_host"] = host
            if get("LLM_CACHE_OPERATIONS") is not None:
                raw["llm_cache_operations"] = frozenset(
                    op.strip() for op in get("LLM_CACHE_OPERATIONS").split(",") if op.strip())
            mode = (get("APP_MODE") or "auto").lower()
            if mode not in {"auto", "fixture", "live"}:
                raise ValueError("APP_MODE must be auto, fixture or live")
            raw["app_mode"] = ("live" if get("OPENAI_API_KEY") else "fixture") if mode == "auto" else mode
            return cls.model_validate(raw)
        except ValidationError as exc:
            # Report variable locations and messages only; input values may be secrets.
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'settings'}: {e['msg']}"
                                 for e in exc.errors(include_input=False, include_url=False))
            raise ConfigurationError(f"Invalid configuration: {problems}") from None
        except ValueError as exc:
            raise ConfigurationError(f"Invalid configuration: {exc}") from None

    def public_summary(self) -> dict[str, object]:
        """Non-secret configuration facts for /ready and startup logs."""
        return {"app_env": self.app_env, "app_mode": self.app_mode, "model": self.openai_model
                if self.app_mode == "live" else "fixture-scripted", "embeddings": self.embeddings,
                "database": self.database_url is not None, "tracing": self.tracing_active,
                "cache": self.llm_cache_enabled, "prompt_version": self.agent_prompt_version}
