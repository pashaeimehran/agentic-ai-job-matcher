from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def validate_vectors(vectors, count: int, dimensions: int) -> list[list[float]]:
    if len(vectors) != count:
        raise ValueError("Embedding count does not match input count.")
    result = []
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError("Embedding dimension mismatch.")
        values = [float(x) for x in vector]
        if not all(math.isfinite(x) for x in values) or not any(values):
            raise ValueError("Embeddings must be finite, nonzero vectors.")
        result.append(values)
    return result


class OpenAIEmbeddings:
    def __init__(self, client, model="text-embedding-3-small", dimensions=1536):
        if not 1 <= dimensions <= 2000:
            raise ValueError("Use 1..2000 embedding dimensions.")
        self.client, self.model, self.dimensions = client, model, dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_with_usage(texts)[0]

    def embed_with_usage(self, texts: Sequence[str]):
        """Vectors plus summed token usage (src.llm.provider.LLMUsage) across batches."""
        from src.llm.provider import LLMUsage
        if any(not text.strip() or len(text) > 2000 for text in texts):
            raise ValueError("Embedding inputs must contain 1..2000 characters.")
        output, tokens = [], 0
        for offset in range(0, len(texts), 32):
            batch = list(texts[offset:offset + 32])
            response = self.client.embeddings.create(model=self.model, input=batch,
                         dimensions=self.dimensions, encoding_format="float")
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(batch))):
                raise ValueError("Unexpected embedding response indices.")
            output.extend(validate_vectors([item.embedding for item in items], len(batch), self.dimensions))
            tokens += getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        return output, LLMUsage(input_tokens=tokens)


class HashEmbeddings:
    """Offline lexical test baseline, NOT a semantic embedding model."""
    model = "lexical-hash-v1"
    dimensions = 512

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"\w+", text.casefold()):
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % self.dimensions
                vector[index] += 1
            vectors.append(vector)
        return validate_vectors(vectors, len(texts), self.dimensions)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimension mismatch.")
    score = sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return max(-1.0, min(1.0, score))
