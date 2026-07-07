import argparse
import json

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="ClinicalTrials.gov RAG Pipeline")
    parser.add_argument("question", nargs="?", help="Natural language question about clinical trials")
    parser.add_argument("--eval", action="store_true", help="Run evaluation suite")
    parser.add_argument("--test-cases", help="Path to JSON file with evaluation test cases")
    args = parser.parse_args()

    if args.eval:
        from src.evaluation.harness import DEFAULT_TEST_CASES, run_evaluation_suite

        test_cases = DEFAULT_TEST_CASES
        if args.test_cases:
            with open(args.test_cases) as f:
                test_cases = json.load(f)

        print("Running evaluation suite...\n")
        results = run_evaluation_suite(test_cases)

        for r in results:
            print(f"\nQ: {r['question']}")
            print(f"Confidence: {r['confidence']:.0%}")
            print(f"Citations: {', '.join(r['citations']) or 'None'}")
            print(f"Eval scores:\n{json.dumps(r['eval_scores'], indent=2)}")
        return

    question = args.question or input("Enter your clinical trial question: ").strip()
    if not question:
        print("No question provided.")
        return

    print(f"\nSearching ClinicalTrials.gov for: {question}\n")

    from src.agent.graph import run_query
    state = run_query(question)

    print("=" * 60)
    print(state["aggregated_answer"])
    print("=" * 60)
    print(f"\nConfidence:  {state['confidence_score']:.0%}")
    if state["citations"]:
        print(f"Citations:   {', '.join(state['citations'])}")
    if state.get("reformulation_count", 0) > 0:
        print(f"Reformulated {state['reformulation_count']} time(s) due to weak retrieval")


if __name__ == "__main__":
    main()
