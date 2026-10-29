from __future__ import annotations

from typing import Literal, Protocol, Sequence, TypeVar, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class LLMMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


T = TypeVar("T", bound=BaseModel)


class StructuredLLMProvider(Protocol):
    """Provider-neutral interface for schema-constrained LLM generation."""

    def generate(
        self,
        messages: Sequence[LLMMessage],
        schema: type[T],
    ) -> T:
        ...


class LLMUsage(BaseModel):
    """Token usage reported by a provider for one call (counts only, never content)."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    # Subset of input_tokens served from the provider's prompt cache (billed at a lower rate).
    cached_input_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot return a parsed structured result."""
