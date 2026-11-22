"""ClinicalPipeline: the one object CLI, REST API, and evals all use.

The graph (models, embeddings, vector store) is built once and reused, so running
thousands of queries does not reload anything per query.
"""
import threading
from typing import Optional

from src.agent.deps import Deps, build_default_deps
from src.agent.graph import build_graph
from src.agent.state import RAGState
from src.agent.tools import execute_tool, validate_call
from src.config import AppConfig, load_config
from src.evaluation.judge import run_judge
from src.retrieval.vector_store import retrieve_documents, store_trials

_default: Optional["ClinicalPipeline"] = None
_default_lock = threading.Lock()


def result_from_state(state: RAGState) -> dict:
    """JSON-serializable result of a full run."""
    return {
        "question": state.get("question", ""),
        "answer": state.get("aggregated_answer", ""),
        "intent": state.get("intent"),
        "confidence": state.get("confidence_score"),
        "agreement_models": state.get("agreement_models", 0),
        "citations": state.get("citations", []),
        "verified_citations": state.get("verified_citations", []),
        "unverified_citations": state.get("unverified_citations", []),
        "evidence_support_rate": state.get("evidence_support_rate"),
        "evidence": state.get("evidence", []),
        "tool_calls_llm": state.get("tool_calls_llm", []),
        "tool_runs": state.get("tool_runs", []),
        "plan_source": state.get("plan_source"),
        "models_used": state.get("models_used", {}),
        "responses": state.get("responses", {}),
        "critique": state.get("critique", {}),
        "revision_count": state.get("revision_count", 0),
        "reformulation_count": state.get("reformulation_count", 0),
        "context_nct_ids": state.get("context_nct_ids", []),
        "context_chunks": state.get("relevant_chunks", []),
        "timings": state.get("timings", {}),
        "total_latency_s": state.get("total_latency_s"),
        "warnings": state.get("warnings", []),
        "errors": state.get("errors", []),
    }


class ClinicalPipeline:
    def __init__(self, config: Optional[AppConfig] = None, deps: Optional[Deps] = None):
        self.config = config or load_config()
        self.deps = deps or build_default_deps(self.config)
        self._graphs: dict = {}
        self._lock = threading.Lock()

    def _graph(self, mode: str = "full"):
        with self._lock:
            if mode not in self._graphs:
                self._graphs[mode] = build_graph(self.deps, mode)
            return self._graphs[mode]

    def run_state(self, question: str) -> RAGState:
        return self._graph("full").invoke({"question": question})

    def ask(self, question: str) -> dict:
        return result_from_state(self.run_state(question))

    def plan(self, question: str) -> dict:
        """Tool planning only (no fetch, no synthesis). Used for tool-selection evals."""
        state = self._graph("plan_only").invoke({"question": question})
        return {
            "question": question,
            "intent": state.get("intent"),
            "tool_calls_llm": state.get("tool_calls_llm", []),
            "tool_plan": state.get("tool_plan", []),
            "plan_source": state.get("plan_source"),
            "plan_json_ok": state.get("plan_json_ok", False),
            "models_used": state.get("models_used", {}),
            "timings": state.get("timings", {}),
            "errors": state.get("errors", []),
        }

    def retrieve(self, question: str) -> dict:
        """Plan, fetch, and retrieve; stops before synthesis. Used for retrieval evals."""
        state = self._graph("retrieval_only").invoke({"question": question})
        return {
            "question": question,
            "intent": state.get("intent"),
            "tool_runs": state.get("tool_runs", []),
            "fetched_nct_ids": [t["nct_id"] for t in state.get("retrieved_trials", [])],
            "context_nct_ids": state.get("context_nct_ids", []),
            "chunk_scores": state.get("chunk_scores", []),
            "reformulation_count": state.get("reformulation_count", 0),
            "errors": state.get("errors", []),
        }

    def search(self, args: dict, question: str = "") -> dict:
        """Structured search plus semantic ranking, with no LLM call."""
        call = validate_call("search_trials", args)
        if call is None:
            raise ValueError("invalid search arguments")
        settings = self.config.pipeline
        trials = execute_tool("search_trials", call["args"], self.deps.ctgov, settings.fetch_page_size)
        docs = []
        if trials:
            store_trials(trials, self.deps.vector_store, settings.eligibility_chars)
            docs = retrieve_documents(
                question or " ".join(str(v) for v in call["args"].values()),
                self.deps.vector_store, k=min(settings.retrieval_k, len(trials)),
                restrict_ids=[t["nct_id"] for t in trials],
            )
        return {"n_fetched": len(trials), "results": docs}

    def call_tool(self, name: str, args: dict) -> list[dict]:
        call = validate_call(name, args)
        if call is None:
            raise ValueError(f"invalid arguments for tool {name!r}")
        return execute_tool(name, call["args"], self.deps.ctgov, self.config.pipeline.fetch_page_size)

    def judge(self, question: str, answer: str, contexts: list[str]) -> dict:
        context = "\n\n---\n\n".join(contexts)[:self.config.pipeline.max_context_chars]
        return run_judge(question, answer, context, self.deps.registry).model_dump()

    def describe(self) -> dict:
        return self.config.describe()


def get_default_pipeline() -> ClinicalPipeline:
    global _default
    with _default_lock:
        if _default is None:
            _default = ClinicalPipeline()
        return _default
