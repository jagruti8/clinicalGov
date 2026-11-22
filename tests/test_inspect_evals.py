"""Runs the real Inspect tasks end to end with Inspect's mock model and a fake pipeline."""
import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval  # noqa: E402

from evals import common  # noqa: E402
from evals.answer_quality import answer_quality  # noqa: E402
from evals.retrieval_check import retrieval_check  # noqa: E402
from evals.tool_selection import tool_selection  # noqa: E402


class FakePipeline:
    def __init__(self, judge_fails=False):
        self.judge_fails = judge_fails

    def ask(self, question):
        return {"answer": "Answer [NCT00000001] [NCT99999999]", "confidence": 0.8, "citations": [],
                "verified_citations": ["NCT00000001"], "unverified_citations": ["NCT99999999"],
                "evidence_support_rate": 0.5, "total_latency_s": 2.0 if "diabetes" in question else 4.0,
                "timings": {"retrieval.synthesize": 1.5}, "context_chunks": ["ctx"]}

    def judge(self, question, answer, contexts):
        if self.judge_fails:
            return {"faithfulness": None, "citation_accuracy": None, "error": "judge unavailable",
                    "faithfulness_reasoning": "", "citation_reasoning": ""}
        return {"faithfulness": 0.8, "citation_accuracy": 0.6, "error": None,
                "faithfulness_reasoning": "r", "citation_reasoning": "r"}

    def plan(self, question):
        name = "get_trial" if "NCT" in question and "Compare" not in question and "criteria" not in question \
            and "enroll" not in question else "search_trials"
        calls = [{"name": name, "args": {}}]
        return {"tool_calls_llm": calls, "plan_json_ok": True}

    def retrieve(self, question):
        return {"fetched_nct_ids": ["NCT00000001", "NCT00000002"], "context_nct_ids": ["NCT00000001"]}


def run(task, pipeline, tmp_path, **kwargs):
    common.set_pipeline(pipeline)
    log = inspect_eval(task, log_dir=str(tmp_path), display="none", **kwargs)[0]
    assert log.status == "success", log.error
    return log


def metrics(log):
    return {(s.name, k): v.value for s in log.results.scores for k, v in s.metrics.items()}


def test_answer_quality_scores_and_latency_median(tmp_path):
    log = run(answer_quality(), FakePipeline(), tmp_path)
    m = metrics(log)
    assert log.results.total_samples == 5
    assert m[("faithfulness", "mean")] == pytest.approx(0.8)
    assert m[("verified_citation_rate", "mean")] == pytest.approx(0.5)
    assert m[("seconds", "median_value")] in (2.0, 4.0)
    assert ("seconds", "p95_value") in m


def test_judge_failures_are_unscored_not_zero(tmp_path):
    log = run(answer_quality(), FakePipeline(judge_fails=True), tmp_path)
    faith = next(s for s in log.results.scores if s.name == "faithfulness")
    assert faith.scored_samples == 0 and faith.unscored_samples == 5


def test_tool_selection_accuracy(tmp_path):
    log = run(tool_selection(), FakePipeline(), tmp_path)
    accuracy = next(v.value for s in log.results.scores if s.name == "tool_selection_scorer"
                    for k, v in s.metrics.items() if k == "accuracy")
    assert 0.0 < accuracy < 1.0                       # fake planner is right on some, wrong on others
    assert log.results.total_samples == 8


def test_retrieval_check_recall(tmp_path):
    dataset = tmp_path / "r.jsonl"
    dataset.write_text('{"id":"r1","input":"q one","target":"NCT00000001","metadata":{"expected_nct":"NCT00000001"}}\n'
                       '{"id":"r2","input":"q two","target":"NCT00000002","metadata":{"expected_nct":"NCT00000002"}}\n')
    log = run(retrieval_check(dataset=str(dataset)), FakePipeline(), tmp_path / "logs")
    m = metrics(log)
    assert m[("fetched_recall", "mean")] == 1.0 and m[("context_recall", "mean")] == 0.5


def test_missing_dataset_gives_actionable_error():
    with pytest.raises(FileNotFoundError, match="generate_queries"):
        common.load_dataset("evals/datasets/does_not_exist.jsonl")
