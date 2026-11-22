"""Quick evaluation harness for the CLI (`python main.py --eval`).

For large, resumable, logged runs use the Inspect tasks in evals/ instead.
"""
import os
import time
from typing import Optional

DEFAULT_TEST_CASES = [
    {"question": "What phase 3 trials are recruiting for type 2 diabetes treatments?"},
    {"question": "Are there completed trials studying immunotherapy for lung cancer?"},
    {"question": "What trials study metformin in combination with other drugs?"},
]


def run_evaluation_suite(test_cases: Optional[list[dict]] = None, pipeline=None,
                         delay_s: Optional[float] = None) -> list[dict]:
    if pipeline is None:
        from src.pipeline import get_default_pipeline
        pipeline = get_default_pipeline()
    if test_cases is None:
        test_cases = DEFAULT_TEST_CASES
    if delay_s is None:
        delay_s = float(os.environ.get("EVAL_DELAY_SECONDS", "0"))

    results = []
    for index, case in enumerate(test_cases):
        question = case["question"]
        print(f"Evaluating: {question}")
        result = pipeline.ask(question)
        scores = pipeline.judge(question, result["answer"], result["context_chunks"])
        results.append({
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "citations": result["citations"],
            "verified_citations": result["verified_citations"],
            "unverified_citations": result["unverified_citations"],
            "total_latency_s": result["total_latency_s"],
            "eval_scores": scores,
        })
        if delay_s and index < len(test_cases) - 1:
            time.sleep(delay_s)
    return results
