import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()


def _print_result(result: dict) -> None:
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    if result["confidence"] is not None:
        print(f"\nAgreement:   {result['confidence']:.0%} across {result['agreement_models']} models")
    else:
        print(f"\nAgreement:   n/a ({result['agreement_models']} model answered)")
    if result["citations"]:
        print(f"Citations:   {', '.join(result['citations'])}")
    if result["unverified_citations"]:
        print(f"Unverified:  {', '.join(result['unverified_citations'])}")
    if result["reformulation_count"]:
        print(f"Reformulated {result['reformulation_count']} time(s) due to weak retrieval")
    if result["revision_count"]:
        print(f"Revised {result['revision_count']} time(s) after review")
    for warning in result["warnings"]:
        print(f"Warning:     {warning}")
    if result["total_latency_s"] is not None:
        print(f"Latency:     {result['total_latency_s']:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="ClinicalTrials.gov agentic RAG pipeline")
    parser.add_argument("question", nargs="?", help="Natural language question about clinical trials")
    parser.add_argument("--config", help="Path to a models YAML (default: config/models.yaml or $MODELS_CONFIG)")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    parser.add_argument("--eval", action="store_true", help="Run the quick evaluation suite")
    parser.add_argument("--test-cases", help="Path to JSON file with evaluation test cases")
    parser.add_argument("--smoke", action="store_true", help="Check config, endpoints, and one query end to end")
    parser.add_argument("--quick", action="store_true", help="With --smoke: skip the LLM and full-pipeline checks")
    parser.add_argument("--serve", action="store_true", help="Run the REST API (uvicorn)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.config:
        os.environ["MODELS_CONFIG"] = args.config

    if args.smoke:
        from src.smoke import run_smoke
        raise SystemExit(0 if run_smoke(quick=args.quick) else 1)

    if args.serve:
        import uvicorn
        uvicorn.run("src.api.app:app", host=args.host, port=args.port)
        return

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
            conf = f"{r['confidence']:.0%}" if r["confidence"] is not None else "n/a"
            print(f"Agreement: {conf}")
            print(f"Citations: {', '.join(r['citations']) or 'None'}  (unverified: {', '.join(r['unverified_citations']) or 'None'})")
            print(f"Eval scores:\n{json.dumps(r['eval_scores'], indent=2)}")
        return

    question = args.question or input("Enter your clinical trial question: ").strip()
    if not question:
        print("No question provided.")
        return

    from src.pipeline import get_default_pipeline

    print(f"\nSearching ClinicalTrials.gov for: {question}\n")
    result = get_default_pipeline().ask(question)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_result(result)


if __name__ == "__main__":
    main()
