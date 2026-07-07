import time

from src.evaluation.judge import JudgeScores, run_judge
from src.llm.providers import get_gemini_llm

DEFAULT_TEST_CASES = [
    {"question": "What phase 3 trials are recruiting for type 2 diabetes treatments?"},
    {"question": "Are there completed trials studying immunotherapy for lung cancer?"},
    {"question": "What trials study metformin in combination with other drugs?"},
]


def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    gemini_llm = get_gemini_llm()
    context_str = "\n\n---\n\n".join(contexts)

    judge: JudgeScores = run_judge(question, answer, context_str, gemini_llm)
    scores = {
        "faithfulness_judge": judge.faithfulness,
        "citation_accuracy": judge.citation_accuracy,
        "faithfulness_reasoning": judge.faithfulness_reasoning,
        "citation_reasoning": judge.citation_reasoning,
    }


    return scores


def run_evaluation_suite(test_cases: list[dict] = None) -> list[dict]:
    from src.agent.graph import run_query

    if test_cases is None:
        test_cases = DEFAULT_TEST_CASES

    results = []
    for case in test_cases:
        question = case["question"]
        print(f"Evaluating: {question}")

        state = run_query(question)
        eval_scores = evaluate_response(
            question=question,
            answer=state["aggregated_answer"],
            contexts=state["relevant_chunks"],
        )

        results.append({
            "question": question,
            "answer": state["aggregated_answer"],
            "confidence": state["confidence_score"],
            "citations": state["citations"],
            "eval_scores": eval_scores,
        })

        if case is not test_cases[-1]:
            time.sleep(20)

    return results
