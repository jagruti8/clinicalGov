from typing import Optional, TypedDict


class RAGState(TypedDict, total=False):
    # input
    question: str

    # orchestrator
    intent: str                      # search | detail | compare | eligibility
    critique_mode: str               # effective mode for this question
    started_at: float
    total_latency_s: float
    timings: dict                    # seconds per node, e.g. {"retrieval.synthesize": 4.2}
    warnings: list[str]
    errors: list[str]

    # retrieval agent: planning and tools
    tool_calls_llm: list[dict]       # valid calls the planner model produced (what evals score)
    plan_source: str                 # llm | llm+nct_fix | default
    plan_json_ok: bool
    tool_plan: list[dict]            # calls that will actually run
    tool_runs: list[dict]            # {"name", "args", "n_trials"} per executed call
    query_params: dict
    reformulation_count: int

    # retrieval agent: retrieval
    retrieved_trials: list[dict]
    relevant_chunks: list[str]
    chunk_scores: list[float]
    context_nct_ids: list[str]       # trials whose text the models actually saw
    is_relevant: bool

    # retrieval agent: synthesis
    responses: dict                  # model name -> draft text
    models_used: dict                # step -> model that answered (after fallback)
    draft_answer: str
    aggregated_answer: str           # final answer shown to the user
    confidence_score: Optional[float]
    agreement_models: int

    # evidence agent
    evidence: list[dict]
    evidence_support_rate: Optional[float]
    citations: list[str]
    verified_citations: list[str]
    unverified_citations: list[str]

    # critique agent
    critique: dict                   # {"verdict": pass|revise, "issues": [...], "source": ...}
    revision_count: int

    # evaluation
    eval_scores: Optional[dict]
