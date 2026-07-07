from functools import partial

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    MAX_REFORMULATIONS,
    agreement_scorer_node,
    multi_llm_synthesizer_node,
    query_constructor_node,
    reformulator_node,
    relevance_checker_node,
    retriever_node,
)
from src.agent.state import RAGState
from src.llm.providers import get_gemini_llm, get_groq_llm, get_secondary_groq_llm
from src.retrieval.vector_store import get_embeddings, get_vector_store


def _route_after_relevance_check(state: RAGState) -> str:
    if state.get("is_relevant"):
        return "synthesize"
    if state.get("reformulation_count", 0) >= MAX_REFORMULATIONS:
        return "synthesize"
    return "reformulate"


def build_graph():
    groq_llm = get_groq_llm()
    gemma_llm = get_secondary_groq_llm()
    gemini_llm = get_gemini_llm()
    vector_store = get_vector_store(get_embeddings())

    graph = StateGraph(RAGState)

    graph.add_node("query_constructor", partial(query_constructor_node, groq_llm=groq_llm))
    graph.add_node("retriever", partial(retriever_node, vector_store=vector_store))
    graph.add_node("relevance_checker", relevance_checker_node)
    graph.add_node("reformulator", partial(reformulator_node, groq_llm=groq_llm))
    graph.add_node("multi_llm_synthesizer", partial(multi_llm_synthesizer_node, groq_llm=groq_llm, gemma_llm=gemma_llm, gemini_llm=gemini_llm))
    graph.add_node("agreement_scorer", agreement_scorer_node)

    graph.add_edge(START, "query_constructor")
    graph.add_edge("query_constructor", "retriever")
    graph.add_edge("retriever", "relevance_checker")
    graph.add_conditional_edges(
        "relevance_checker",
        _route_after_relevance_check,
        {"synthesize": "multi_llm_synthesizer", "reformulate": "reformulator"},
    )
    graph.add_edge("reformulator", "retriever")
    graph.add_edge("multi_llm_synthesizer", "agreement_scorer")
    graph.add_edge("agreement_scorer", END)

    return graph.compile()


def run_query(question: str) -> RAGState:
    graph = build_graph()
    return graph.invoke(RAGState(
        question=question,
        query_params={},
        retrieved_trials=[],
        relevant_chunks=[],
        chunk_scores=[],
        is_relevant=False,
        reformulation_count=0,
        groq_response="",
        gemma_response="",
        gemini_response="",
        aggregated_answer="",
        confidence_score=0.0,
        citations=[],
        eval_scores=None,
    ))
