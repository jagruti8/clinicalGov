import json
import re
import time

from langchain_core.output_parsers import StrOutputParser

from src.agent.state import RAGState
from src.agent.prompts import QUERY_CONSTRUCTOR_PROMPT, REFORMULATION_PROMPT, SYNTHESIS_PROMPT
from src.retrieval.ctgov_client import build_query_params, fetch_trials
from src.retrieval.vector_store import store_trials, retrieve_chunks
from src.llm.aggregator import build_synthesis_chain, aggregate_responses

RELEVANCE_THRESHOLD = 0.4
MAX_REFORMULATIONS = 2


def _parse_json(raw: str, fallback: dict) -> dict:
    cleaned = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback


def query_constructor_node(state: RAGState, groq_llm) -> dict:
    chain = QUERY_CONSTRUCTOR_PROMPT | groq_llm | StrOutputParser()
    raw = chain.invoke({"question": state["question"]})
    params = _parse_json(raw, {"condition": state["question"], "intervention": None, "phase": None, "status": None})
    return {
        "query_params": build_query_params(
            condition=params.get("condition"),
            intervention=params.get("intervention"),
            phase=params.get("phase"),
            status=params.get("status"),
        )
    }


def retriever_node(state: RAGState, vector_store) -> dict:
    try:
        trials = fetch_trials(state["query_params"])
    except Exception as exc:
        print(f"[Warning] Trial fetch failed ({exc}), treating as no results.")
        return {"retrieved_trials": [], "relevant_chunks": [], "chunk_scores": []}
    if not trials:
        return {"retrieved_trials": [], "relevant_chunks": [], "chunk_scores": []}

    store_trials(trials, vector_store)
    chunks, scores = retrieve_chunks(state["question"], vector_store)
    return {"retrieved_trials": trials, "relevant_chunks": chunks, "chunk_scores": scores}


def relevance_checker_node(state: RAGState) -> dict:
    scores = state.get("chunk_scores", [])
    is_relevant = bool(scores) and max(scores) >= RELEVANCE_THRESHOLD
    return {"is_relevant": is_relevant}


def reformulator_node(state: RAGState, groq_llm) -> dict:
    chain = REFORMULATION_PROMPT | groq_llm | StrOutputParser()
    raw = chain.invoke({
        "question": state["question"],
        "query_params": json.dumps(state["query_params"]),
        "reformulation_count": state["reformulation_count"],
    })
    params = _parse_json(raw, {"condition": state["question"], "intervention": None, "phase": None, "status": None})
    return {
        "query_params": build_query_params(
            condition=params.get("condition"),
            intervention=params.get("intervention"),
            phase=params.get("phase"),
            status=params.get("status"),
        ),
        "reformulation_count": state["reformulation_count"] + 1,
    }


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg


def _call_llm_with_retry(chain, inputs: dict, label: str) -> str:
    for wait in [0, 15, 30]:
        try:
            if wait:
                time.sleep(wait)
            return chain.invoke(inputs).content
        except Exception as exc:
            if not _is_rate_limit(exc):
                print(f"[Warning] {label} unavailable ({type(exc).__name__}), skipping.")
                return ""
            if wait == 30:
                print(f"[Warning] {label} unavailable after retries, skipping.")
            else:
                print(f"[Warning] {label} rate-limited, retrying in {wait + 15}s...")
    return ""


def multi_llm_synthesizer_node(state: RAGState, groq_llm, gemma_llm, gemini_llm) -> dict:
    context = "\n\n---\n\n".join(state.get("relevant_chunks", [])) or "No relevant clinical trial data found."
    inputs = {"question": state["question"], "context": context}

    gemma_inputs = {"question": inputs["question"], "context": inputs["context"][:6000]}

    groq_response = (SYNTHESIS_PROMPT | groq_llm).invoke(inputs).content
    gemma_response = _call_llm_with_retry(SYNTHESIS_PROMPT | gemma_llm, gemma_inputs, "Llama-8B")
    gemini_response = _call_llm_with_retry(SYNTHESIS_PROMPT | gemini_llm, inputs, "Gemini")

    return {"groq_response": groq_response, "gemma_response": gemma_response, "gemini_response": gemini_response}


def agreement_scorer_node(state: RAGState) -> dict:
    aggregated, confidence = aggregate_responses(
        state.get("groq_response", ""),
        state.get("gemma_response", ""),
        state.get("gemini_response", ""),
    )
    citations = list(dict.fromkeys(re.findall(r"\[?(NCT\d{8})\]?", aggregated)))
    return {
        "aggregated_answer": aggregated,
        "confidence_score": confidence,
        "citations": citations,
    }
