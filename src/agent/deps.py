from dataclasses import dataclass
from typing import Any

from src.config import AppConfig
from src.llm.providers import ModelRegistry


@dataclass
class Deps:
    """Everything the graph nodes need. Tests pass fakes; production uses build_default_deps."""
    registry: ModelRegistry
    vector_store: Any
    ctgov: Any        # provides fetch_trials(params, max_results) and get_trial(nct_id)
    embedder: Any     # provides encode(list[str]) -> array

    @property
    def config(self) -> AppConfig:
        return self.registry.config


def build_default_deps(config: AppConfig) -> Deps:
    from src.llm.aggregator import EmbeddingsAdapter
    from src.retrieval import ctgov_client
    from src.retrieval.vector_store import get_embeddings, get_vector_store

    embeddings = get_embeddings(config.embedding.model, config.embedding.device)
    return Deps(
        registry=ModelRegistry(config),
        vector_store=get_vector_store(embeddings),
        ctgov=ctgov_client,
        embedder=EmbeddingsAdapter(embeddings),
    )
