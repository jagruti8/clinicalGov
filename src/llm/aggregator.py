import numpy as np
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

_embed_model: SentenceTransformer = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def compute_agreement(response1: str, response2: str) -> float:
    model = _get_embed_model()
    embeddings = model.encode([response1, response2], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def build_synthesis_chain(groq_llm, gemini_llm, prompt: ChatPromptTemplate) -> RunnableParallel:
    return RunnableParallel(
        groq=prompt | groq_llm,
        gemini=prompt | gemini_llm,
    )


def aggregate_responses(groq_response: str, gemma_response: str = "", gemini_response: str = "") -> tuple[str, float]:
    available = [(name, r) for name, r in [
        ("Groq (Llama)", groq_response),
        ("Groq (Gemma)", gemma_response),
        ("Gemini", gemini_response),
    ] if r]

    if len(available) == 1:
        return available[0][1], 1.0

    responses = [r for _, r in available]
    pairs = [(responses[i], responses[j]) for i in range(len(responses)) for j in range(i + 1, len(responses))]
    confidence = float(np.mean([compute_agreement(a, b) for a, b in pairs]))

    if confidence >= 0.5:
        return groq_response, confidence

    combined = "\n\n".join(f"**{name}:**\n{r}" for name, r in available)
    return (
        f"Note: Models had low agreement (confidence: {confidence:.0%}). All perspectives shown.\n\n{combined}"
    ), confidence
