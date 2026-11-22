"""Orchestrator agent: initializes the run, routes by intent, and assembles the result."""
import re
import time

from src.agent.deps import Deps
from src.agent.state import RAGState
from src.agent.tools import INTENT_BY_TOOL, find_nct_ids


def start_node(state: RAGState, deps: Deps) -> dict:
    return {
        "started_at": time.perf_counter(),
        "timings": {},
        "errors": [],
        "warnings": [],
        "models_used": {},
        "responses": {},
        "reformulation_count": 0,
        "revision_count": 0,
        "critique": {},
    }


def route_node(state: RAGState, deps: Deps) -> dict:
    """Intent follows the first planned tool. Lookups by NCT ID get a cheaper critique
    pass (code checks only) unless critique_mode is off, because the context is exact."""
    plan = state.get("tool_plan") or [{"name": "search_trials"}]
    intent = INTENT_BY_TOOL.get(plan[0]["name"], "search")
    mode = deps.config.pipeline.critique_mode
    if intent in {"detail", "eligibility"} and mode == "full":
        mode = "code_only"
    return {"intent": intent, "critique_mode": mode}


def finalize_node(state: RAGState, deps: Deps) -> dict:
    """Authoritative citation check on the text the user will see, plus latency."""
    warnings = list(state.get("warnings") or [])
    answer = state.get("aggregated_answer") or ""
    if not answer:
        answer = "No answer could be generated: every configured model failed. See `errors` for details."
        warnings.append("no model produced an answer")

    context_ids = set(state.get("context_nct_ids") or [])
    cited = find_nct_ids(answer)
    verified = [i for i in cited if i in context_ids]
    unverified = [i for i in cited if i not in context_ids]
    for nct_id in unverified:
        answer = re.sub(rf"\[?{nct_id}\]?", f"[{nct_id} unverified]", answer)
    if unverified:
        warnings.append(f"{len(unverified)} citation(s) not found in the retrieved trial data: {', '.join(unverified)}")

    started = state.get("started_at")
    return {
        "aggregated_answer": answer,
        "citations": cited,
        "verified_citations": verified,
        "unverified_citations": unverified,
        "warnings": warnings,
        "total_latency_s": round(time.perf_counter() - started, 4) if started else None,
    }
