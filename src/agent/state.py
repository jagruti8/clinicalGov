from typing import Optional, TypedDict


class RAGState(TypedDict):
    question: str
    query_params: dict
    retrieved_trials: list[dict]
    relevant_chunks: list[str]
    chunk_scores: list[float]
    is_relevant: bool
    reformulation_count: int
    groq_response: str
    gemma_response: str
    gemini_response: str
    aggregated_answer: str
    confidence_score: float
    citations: list[str]
    eval_scores: Optional[dict]
