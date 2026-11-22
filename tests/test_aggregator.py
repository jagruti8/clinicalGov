import pytest

from src.llm.aggregator import EmbeddingsAdapter, agreement_score, combine_responses
from tests.conftest import BowEmbedder


def test_single_response_has_no_agreement_score():
    assert agreement_score(["only one"], BowEmbedder()) is None
    assert agreement_score([], BowEmbedder()) is None


def test_identical_responses_agree_fully():
    assert agreement_score(["metformin study", "metformin study"], BowEmbedder()) == pytest.approx(1.0)


def test_unrelated_responses_disagree():
    assert agreement_score(["metformin diabetes", "melanoma pembrolizumab"], BowEmbedder()) < 0.5


def test_works_with_three_models():
    score = agreement_score(["a b c", "a b c", "a b d"], BowEmbedder())
    assert 0.5 < score < 1.0


def test_combine_lists_every_model():
    text = combine_responses({"x": "one", "y": "two"}, 0.3)
    assert "low agreement" in text and "**x:**" in text and "**y:**" in text and "30%" in text


def test_embeddings_adapter_wraps_langchain_embeddings():
    class Fake:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]
    assert EmbeddingsAdapter(Fake()).encode(["a", "b"]).shape == (2, 2)
