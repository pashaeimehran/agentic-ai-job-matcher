"""Request/run scope carried through contextvars.

The API middleware opens a request scope; agents open a run scope inside it.
Instrumentation reads the current scope to label metrics and to enforce the
per-run token/cost budget, so business code never passes these values around.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


class BudgetExceededError(RuntimeError):
    """Raised before an LLM call when the current run has spent its configured budget."""


@dataclass
class RunBudget:
    """Per-run spending limit. Checked before each LLM call; a call in flight is never cut off."""
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0

    def check(self) -> None:
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            raise BudgetExceededError(f"Run token budget of {self.max_tokens} reached.")
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(f"Run cost budget of ${self.max_cost_usd} reached.")

    def add(self, tokens: int, cost_usd: float | None) -> None:
        self.llm_calls += 1
        self.tokens += tokens
        self.cost_usd += cost_usd or 0.0


@dataclass(frozen=True)
class RunContext:
    run_id: str | None = None
    request_id: str | None = None
    budget: RunBudget = field(default_factory=RunBudget)


_CURRENT: contextvars.ContextVar[RunContext] = contextvars.ContextVar("job_match_run", default=RunContext())


def current_run() -> RunContext:
    return _CURRENT.get()


@contextmanager
def run_scope(run_id: str | None = None, request_id: str | None = None,
              budget: RunBudget | None = None) -> Iterator[RunContext]:
    """Open a nested scope; unspecified fields are inherited from the enclosing scope."""
    parent = _CURRENT.get()
    scope = RunContext(run_id=run_id or parent.run_id, request_id=request_id or parent.request_id,
                       budget=budget if budget is not None else parent.budget)
    token = _CURRENT.set(scope)
    try:
        yield scope
    finally:
        _CURRENT.reset(token)
