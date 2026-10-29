"""Model pricing metadata and deterministic cost estimation.

Prices are USD per one million tokens. They are configuration, not permanent
facts: providers change prices. PRICING_AS_OF records when the table was last
reviewed. Override or extend it with the LLM_PRICING_JSON environment variable,
for example {"gpt-4o-mini": {"input": 0.15, "output": 0.6, "cached_input": 0.075}}.
Unknown models get no estimate (None) instead of a guessed one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from src.llm.provider import LLMUsage

PRICING_AS_OF = "2026-10-28"


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None


DEFAULT_PRICING: dict[str, ModelPrice] = {
    "gpt-4o-mini": ModelPrice(0.15, 0.60, 0.075),
    "gpt-4o": ModelPrice(2.50, 10.00, 1.25),
    "gpt-4.1-nano": ModelPrice(0.10, 0.40, 0.025),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, 0.10),
    "gpt-4.1": ModelPrice(2.00, 8.00, 0.50),
    "text-embedding-3-small": ModelPrice(0.02, 0.0),
    "text-embedding-3-large": ModelPrice(0.13, 0.0),
    # Offline stand-ins: no provider is billed.
    "fixture-scripted": ModelPrice(0.0, 0.0, 0.0),
    "lexical-hash-v1": ModelPrice(0.0, 0.0),
}


class PricingTable:
    def __init__(self, prices: dict[str, ModelPrice] | None = None) -> None:
        self._prices = dict(DEFAULT_PRICING if prices is None else prices)

    @classmethod
    def from_json(cls, raw: str | None) -> "PricingTable":
        """Defaults plus overrides from a JSON object; invalid JSON raises ValueError at startup."""
        table = cls()
        if raw:
            for model, price in json.loads(raw).items():
                cached = price.get("cached_input")
                table._prices[model] = ModelPrice(float(price["input"]), float(price["output"]),
                                                  None if cached is None else float(cached))
        return table

    def price_for(self, model: str) -> ModelPrice | None:
        """Exact match, else the longest known prefix (dated snapshots such as gpt-4o-mini-2024-07-18)."""
        if model in self._prices:
            return self._prices[model]
        prefixes = [name for name in self._prices if model.startswith(name + "-")]
        return self._prices[max(prefixes, key=len)] if prefixes else None

    def estimate(self, model: str, usage: LLMUsage | None) -> float | None:
        price = self.price_for(model)
        if price is None or usage is None:
            return None
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        cached_rate = price.cached_input_per_1m if price.cached_input_per_1m is not None else price.input_per_1m
        cost = ((usage.input_tokens - cached) * price.input_per_1m + cached * cached_rate
                + usage.output_tokens * price.output_per_1m) / 1_000_000
        return round(cost, 8)
