"""Retrieval agent nodes: tool planning, execution, retrieval, reformulation,
multi-model synthesis, and agreement scoring.

Every node is `fn(state, deps) -> dict` (a partial state update).
"""
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.deps import Deps
from src.agent.prompts import (
    PLAN_REPAIR, REFORMULATION_PROMPT, REVISION_FEEDBACK, SYNTHESIS_PROMPT, TOOL_ROUTER_PROMPT,
)
from src.agent.state import RAGState
from src.agent.tools import (
    TOOLS, SearchTrialsArgs, calls_from_json, calls_from_native, execute_tool, find_nct_ids,
    heuristic_search_args, nct_fix_call, parse_json_loose, plan_covers_ids, tool_descriptions,
    tool_schemas, validate_call,
)
from src.llm.aggregator import LOW_AGREEMENT, agreement_score, combine_responses
from src.llm.fallback import AllModelsFailed
from src.retrieval.ctgov_client import build_query_params, format_trial_text, trial_id_from_text
from src.retrieval.vector_store import retrieve_documents, store_trials

NATIVE_SYSTEM = (
    "You answer questions about clinical trials by calling tools. Use get_trial, "
    "get_eligibility, or compare_trials only when the question names NCT IDs; "
    "otherwise call search_trials."
)


def _with_model(state: RAGState, step: str, model: str) -> dict:
    return {**(state.get("models_used") or {}), step: model}


def _plan_native(deps: Deps, planner: str, question: str) -> list[dict]:
    llm = deps.registry.get_llm(planner).bind_tools(tool_schemas())
    response = llm.invoke([SystemMessage(content=NATIVE_SYSTEM), HumanMessage(content=question)])
    return calls_from_native(getattr(response, "tool_calls", []))


def plan_tools_node(state: RAGState, deps: Deps) -> dict:
    """Ask the planner model which tool(s) to call. `tool_calls_llm` keeps the model's
    own valid calls untouched; `tool_plan` is what will run (after the NCT override).

    If the reply has no usable call, the planner gets `plan_repair_retries` more tries,
    each told what was wrong. After that a heuristic search runs instead."""
    question = state["question"]
    settings = deps.config.pipeline
    planner = deps.config.roles.planner
    errors = list(state.get("errors") or [])
    models_used = dict(state.get("models_used") or {})

    calls: list[dict] = []
    json_ok = False
    raw = ""
    attempts = 0
    failed_calls: list[str] = []
    if deps.registry.spec(planner).tool_calling == "native":
        attempts += 1
        try:
            calls = _plan_native(deps, planner, question)
            json_ok = bool(calls)
            models_used["planner"] = planner
        except Exception as exc:
            errors.append(f"native tool calling failed on {planner}: {type(exc).__name__}: {str(exc)[:150]}")

    if not calls:
        messages = TOOL_ROUTER_PROMPT.format_messages(question=question, tools=tool_descriptions())
        for _ in range(1 + settings.plan_repair_retries):
            try:
                result = deps.registry.call(planner, messages)
            except AllModelsFailed as exc:
                errors.append(f"tool planning failed: {exc}")
                break
            attempts += 1
            raw = result.text
            models_used["planner"] = result.model
            calls, json_ok = calls_from_json(raw)
            if calls:
                break
            failed_calls.append(f"planner gave no usable tool call (call {attempts}): {raw.strip()[:120]!r}")
            problem = "it had no valid tool call" if json_ok else "it was not valid JSON"
            repair = PLAN_REPAIR.format(problem=problem, tools=", ".join(TOOLS))
            messages = messages + [AIMessage(content=raw), HumanMessage(content=repair)]

    if not calls:                       # a repaired reply needs no error; `plan_attempts` shows it
        errors.extend(failed_calls)

    plan = [dict(call, args=dict(call["args"])) for call in calls]
    source = "llm" if plan else "default"

    nct_ids = find_nct_ids(question)
    if settings.nct_routing_override and nct_ids and not plan_covers_ids(plan, nct_ids):
        plan = [nct_fix_call(question, nct_ids)]
        source += "+nct_fix"

    if not plan:
        plan = [{"name": "search_trials", "args": heuristic_search_args(question)}]

    return {
        "tool_calls_llm": calls,
        "plan_source": source,
        "plan_json_ok": json_ok,
        "plan_raw": raw.strip()[:500],
        "plan_attempts": attempts,
        "tool_plan": plan,
        "errors": errors,
        "models_used": models_used,
    }


def execute_tools_node(state: RAGState, deps: Deps) -> dict:
    """Run the planned tool calls against the live API and merge the trials found."""
    settings = deps.config.pipeline
    errors = list(state.get("errors") or [])
    merged: dict[str, dict] = {}
    runs: list[dict] = []
    query_params = {}

    for call in state.get("tool_plan", []):
        args = dict(call["args"])
        if call["name"] == "search_trials":
            args["max_results"] = min(args.get("max_results", 25), settings.fetch_max_results)
        try:
            trials = execute_tool(call["name"], args, deps.ctgov, page_size=settings.fetch_page_size)
        except Exception as exc:
            errors.append(f"tool {call['name']} failed: {type(exc).__name__}: {str(exc)[:150]}")
            trials = []
        runs.append({"name": call["name"], "args": args, "n_trials": len(trials)})
        if call["name"] == "search_trials" and not query_params:
            search = SearchTrialsArgs(**args)
            query_params = build_query_params(
                condition=search.condition, intervention=search.intervention,
                phase=search.phase, status=search.status, term=search.term,
            )
        for trial in trials:
            merged.setdefault(trial["nct_id"], trial)

    return {
        "retrieved_trials": list(merged.values()),
        "tool_runs": runs,
        "query_params": query_params,
        "errors": errors,
    }


def retrieve_node(state: RAGState, deps: Deps) -> dict:
    """Build the model context. Search results go through the vector store (semantic
    ranking restricted to the trials just fetched). Lookups by NCT ID use the trials
    directly: the question already named them, so there is nothing to rank."""
    settings = deps.config.pipeline
    trials = state.get("retrieved_trials", [])
    empty = {"relevant_chunks": [], "chunk_scores": [], "context_nct_ids": []}
    if not trials:
        return empty

    if state.get("intent") != "search":
        chunks = [format_trial_text(t, settings.eligibility_chars) for t in trials]
        return {"relevant_chunks": chunks, "chunk_scores": [1.0] * len(chunks),
                "context_nct_ids": [t["nct_id"] for t in trials]}

    errors = list(state.get("errors") or [])
    try:
        store_trials(trials, deps.vector_store, settings.eligibility_chars)
        docs = retrieve_documents(
            state["question"], deps.vector_store, k=settings.retrieval_k,
            restrict_ids=[t["nct_id"] for t in trials], query_params=state.get("query_params"),
        )
    except Exception as exc:
        errors.append(f"retrieval failed: {type(exc).__name__}: {str(exc)[:150]}")
        return {**empty, "errors": errors}

    return {
        "relevant_chunks": [d["text"] for d in docs],
        "chunk_scores": [d["score"] for d in docs],
        "context_nct_ids": [d["nct_id"] or trial_id_from_text(d["text"]) or "" for d in docs],
    }


def relevance_check_node(state: RAGState, deps: Deps) -> dict:
    if state.get("intent") != "search":
        return {"is_relevant": bool(state.get("retrieved_trials"))}
    scores = state.get("chunk_scores", [])
    threshold = deps.config.pipeline.relevance_threshold
    return {"is_relevant": bool(scores) and max(scores) >= threshold}


def reformulate_node(state: RAGState, deps: Deps) -> dict:
    """Weak retrieval: ask the planner for better search parameters and re-plan."""
    settings = deps.config.pipeline
    count = state.get("reformulation_count", 0)
    errors = list(state.get("errors") or [])
    params = None
    try:
        messages = REFORMULATION_PROMPT.format_messages(
            question=state["question"],
            query_params=str({k: v for k, v in (state.get("query_params") or {}).items() if k != "fields"}),
            reformulation_count=count,
        )
        result = deps.registry.call(deps.config.roles.planner, messages)
        params = parse_json_loose(result.text)
    except AllModelsFailed as exc:
        errors.append(f"reformulation failed: {exc}")

    call = None
    if params:
        call = validate_call("search_trials", {
            **{k: v for k, v in params.items() if k in {"condition", "intervention", "phase", "status"} and v},
            "max_results": settings.fetch_max_results,
        })
        if call is None:  # an invalid phase/status should not discard the whole reformulation
            call = validate_call("search_trials", {
                **{k: v for k, v in params.items() if k in {"condition", "intervention"} and v},
                "max_results": settings.fetch_max_results,
            })
    if call is None:
        call = {"name": "search_trials", "args": {"term": state["question"], "max_results": settings.fetch_max_results}}

    return {"tool_plan": [call], "reformulation_count": count + 1, "errors": errors}


def route_after_relevance(state: RAGState, deps: Deps) -> str:
    if state.get("is_relevant"):
        return "synthesize"
    if state.get("intent") != "search":
        return "synthesize"
    if state.get("reformulation_count", 0) >= deps.config.pipeline.max_reformulations:
        return "synthesize"
    return "reformulate"


def synthesize_node(state: RAGState, deps: Deps) -> dict:
    """Every configured synthesizer answers in parallel, each with retry and fallback."""
    settings = deps.config.pipeline
    context = "\n\n---\n\n".join(state.get("relevant_chunks", []))[:settings.max_context_chars]
    context = context or "No relevant clinical trial data found."

    critique = state.get("critique") or {}
    issues = critique.get("issues", []) if critique.get("verdict") == "revise" and state.get("revision_count", 0) else []
    feedback = REVISION_FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues)) if issues else ""

    names = deps.config.roles.synthesizers

    def run(name: str):
        messages = SYNTHESIS_PROMPT.format_messages(
            question=state["question"], context=context, feedback=feedback)
        try:
            return name, deps.registry.call(name, messages), None
        except AllModelsFailed as exc:
            return name, None, str(exc)

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        outcomes = list(pool.map(run, names))

    responses = {}
    models_used = dict(state.get("models_used") or {})
    errors = list(state.get("errors") or [])
    for name, result, error in outcomes:
        if result:
            responses[name] = result.text
            models_used[f"synthesizer:{name}"] = result.model
        else:
            errors.append(f"synthesizer {name} failed: {error}")

    return {"responses": responses, "models_used": models_used, "errors": errors}


def score_agreement_node(state: RAGState, deps: Deps) -> dict:
    responses = state.get("responses", {})
    ordered = {name: responses[name] for name in deps.config.roles.synthesizers if name in responses}
    if not ordered:
        return {"draft_answer": "", "aggregated_answer": "", "confidence_score": None, "agreement_models": 0}

    draft = next(iter(ordered.values()))
    confidence = agreement_score(list(ordered.values()), deps.embedder)
    answer = draft
    if confidence is not None and confidence < LOW_AGREEMENT:
        answer = combine_responses(ordered, confidence)
    return {
        "draft_answer": draft,
        "aggregated_answer": answer,
        "confidence_score": confidence,
        "agreement_models": len(ordered),
    }
