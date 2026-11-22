"""Shared pieces for the Inspect tasks: solvers, scorers, and a metric.

How Inspect fits together (see evals/README.md for the full walkthrough):
  Dataset -> Solver -> Scorer
  - the dataset gives samples (input question, optional target)
  - the solver runs OUR pipeline on the input and stores the result on the sample
  - scorers turn that result into numbers; metrics aggregate numbers over samples

Our pipeline calls Ollama itself (models come from config/models.yaml), so Inspect's own
`--model` is not used. Each task defaults to Inspect's built-in `mockllm/model` placeholder.
A score of NaN means "unscored": Inspect leaves it out of the averages.
"""
import asyncio
import math
import statistics
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspect_ai.dataset import json_dataset  # noqa: E402
from inspect_ai.scorer import (  # noqa: E402
    CORRECT, INCORRECT, Metric, SampleScore, Score, Target, mean, metric, scorer, stderr, accuracy,
)
from inspect_ai.solver import TaskState, solver  # noqa: E402

NAN = float("nan")
DATASET_DIR = REPO_ROOT / "evals" / "datasets"

_pipeline = None


def get_pipeline():
    """One shared pipeline per process (models and embeddings load once)."""
    global _pipeline
    if _pipeline is None:
        from src.pipeline import ClinicalPipeline
        _pipeline = ClinicalPipeline()
    return _pipeline


def set_pipeline(pipeline) -> None:
    """Inject a pipeline (used by tests)."""
    global _pipeline
    _pipeline = pipeline


def load_dataset(path: str):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset not found: {resolved}\n"
            "Generate it with: python -m evals.generate_queries --out-dir evals/datasets\n"
            "or use a sample file from evals/datasets/sample_*.jsonl"
        )
    return json_dataset(str(resolved))


def pipeline_info() -> dict:
    from src.config import load_config
    return load_config().describe()


# ---------------------------------------------------------------- metrics

@metric
def median_value() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        values = [float(s.score.value) for s in scores]
        return statistics.median(values) if values else NAN
    return compute


@metric
def p95_value() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        values = sorted(float(s.score.value) for s in scores)
        return values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)] if values else NAN
    return compute


# ---------------------------------------------------------------- solvers

@solver
def ask_solver():
    """Run the full pipeline. Pipeline calls are blocking, so run them in a thread."""
    async def solve(state: TaskState, generate) -> TaskState:
        result = await asyncio.to_thread(get_pipeline().ask, state.input_text)
        state.output.completion = result["answer"]
        state.metadata["result"] = result
        return state
    return solve


@solver
def plan_solver():
    """Tool planning only: which tool(s) does the planner model choose?"""
    async def solve(state: TaskState, generate) -> TaskState:
        plan = await asyncio.to_thread(get_pipeline().plan, state.input_text)
        state.output.completion = ",".join(c["name"] for c in plan["tool_calls_llm"]) or "none"
        state.metadata["plan"] = plan
        return state
    return solve


@solver
def retrieve_solver():
    """Plan, fetch, and retrieve; stop before synthesis."""
    async def solve(state: TaskState, generate) -> TaskState:
        retrieval = await asyncio.to_thread(get_pipeline().retrieve, state.input_text)
        state.output.completion = ",".join(retrieval["context_nct_ids"]) or "none"
        state.metadata["retrieval"] = retrieval
        return state
    return solve


# ---------------------------------------------------------------- scorers

@scorer(metrics={"faithfulness": [mean(), stderr()], "citation_accuracy": [mean(), stderr()]})
def judge_scorer():
    """LLM-as-a-judge (judge model from config). Judge failures are NaN, not zero."""
    async def score(state: TaskState, target: Target) -> Score:
        result = state.metadata["result"]
        scores = await asyncio.to_thread(
            get_pipeline().judge, state.input_text, result["answer"], result["context_chunks"])
        faithfulness = scores["faithfulness"] if scores["faithfulness"] is not None else NAN
        citation = scores["citation_accuracy"] if scores["citation_accuracy"] is not None else NAN
        explanation = scores["error"] or f"{scores['faithfulness_reasoning']} | {scores['citation_reasoning']}"
        return Score(value={"faithfulness": faithfulness, "citation_accuracy": citation},
                     answer=result["answer"][:500], explanation=explanation)
    return score


@scorer(metrics={
    "verified_citation_rate": [mean(), stderr()],
    "evidence_support_rate": [mean(), stderr()],
    "answers_with_citation": [mean()],
})
def citation_scorer():
    """Deterministic checks computed by the pipeline (no LLM involved in the verdict)."""
    async def score(state: TaskState, target: Target) -> Score:
        result = state.metadata["result"]
        cited = len(result["verified_citations"]) + len(result["unverified_citations"])
        verified_rate = len(result["verified_citations"]) / cited if cited else NAN
        support = result["evidence_support_rate"]
        return Score(value={
            "verified_citation_rate": verified_rate,
            "evidence_support_rate": support if support is not None else NAN,
            "answers_with_citation": 1.0 if cited else 0.0,
        })
    return score


@scorer(metrics={"seconds": [median_value(), p95_value(), mean()]})
def latency_scorer():
    """End-to-end seconds per question (median and p95)."""
    async def score(state: TaskState, target: Target) -> Score:
        result = state.metadata["result"]
        seconds = result["total_latency_s"]
        timings = " ".join(f"{k}={v:.1f}s" for k, v in result["timings"].items())
        return Score(value={"seconds": seconds if seconds is not None else NAN}, explanation=timings)
    return score


@scorer(metrics=[accuracy(), stderr()])
def tool_selection_scorer():
    """Correct when the planner's first tool equals the target tool name."""
    async def score(state: TaskState, target: Target) -> Score:
        calls = state.metadata["plan"]["tool_calls_llm"]
        chosen = calls[0]["name"] if calls else "none"
        return Score(value=CORRECT if chosen == target.text else INCORRECT, answer=chosen,
                     explanation=f"expected {target.text}, planner chose {chosen}")
    return score


@scorer(metrics=[accuracy(), stderr()])
def valid_plan_scorer():
    """Correct when the planner produced at least one schema-valid tool call."""
    async def score(state: TaskState, target: Target) -> Score:
        plan = state.metadata["plan"]
        ok = bool(plan["tool_calls_llm"])
        return Score(value=CORRECT if ok else INCORRECT,
                     explanation=f"json parsed: {plan['plan_json_ok']}, valid calls: {len(plan['tool_calls_llm'])}")
    return score


@scorer(metrics={"fetched_recall": [mean(), stderr()], "context_recall": [mean(), stderr()]})
def retrieval_recall_scorer():
    """Did the expected trial surface? fetched = returned by the API query;
    context = also survived semantic ranking into the model's context."""
    async def score(state: TaskState, target: Target) -> Score:
        retrieval = state.metadata["retrieval"]
        expected = str(state.metadata.get("expected_nct", target.text)).upper()
        return Score(value={
            "fetched_recall": 1.0 if expected in retrieval["fetched_nct_ids"] else 0.0,
            "context_recall": 1.0 if expected in retrieval["context_nct_ids"] else 0.0,
        }, explanation=f"expected {expected}; fetched {len(retrieval['fetched_nct_ids'])} trials")
    return score
