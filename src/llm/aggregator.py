"""Cross-model agreement: mean pairwise cosine similarity of response embeddings."""
from itertools import combinations
from typing import Optional

import numpy as np

LOW_AGREEMENT = 0.5


class EmbeddingsAdapter:
    """Reuses the retrieval embedding model (LangChain Embeddings) for agreement scoring."""

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(self._embeddings.embed_documents(texts), dtype=float)


def agreement_score(responses: list[str], embedder) -> Optional[float]:
    """Mean pairwise cosine similarity. None when fewer than two responses exist,
    because agreement with yourself carries no information."""
    if len(responses) < 2:
        return None
    vectors = embedder.encode(responses)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1.0, norms)
    sims = [float(np.dot(vectors[i], vectors[j])) for i, j in combinations(range(len(vectors)), 2)]
    return float(np.mean(sims))


def combine_responses(responses: dict[str, str], confidence: float) -> str:
    combined = "\n\n".join(f"**{name}:**\n{text}" for name, text in responses.items())
    return f"Note: Models had low agreement (confidence: {confidence:.0%}). All perspectives shown.\n\n{combined}"
