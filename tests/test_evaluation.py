import json

from src.evaluation.harness import run_evaluation_suite
from src.evaluation.judge import JudgeScores, run_judge
from src.llm.providers import ModelRegistry
from tests.conftest import ScriptedLLM, make_config


def registry_with(reply):
    config = make_config(llm_retries=0)
    llms = {name: ScriptedLLM(reply) for name in config.models}
    return ModelRegistry(config, builder=lambda spec: llms[spec.name], sleep=lambda s: None)


def test_judge_scores_model_clamps_to_unit_interval():
    scores = JudgeScores(faithfulness=1.7, citation_accuracy=-0.2)
    assert scores.faithfulness == 1.0 and scores.citation_accuracy == 0.0


def test_judge_parses_valid_json_even_with_think_block():
    reply = '<think>hm</think>' + json.dumps({"faithfulness": 0.85, "citation_accuracy": "0.9",
                                              "faithfulness_reasoning": "Good", "citation_reasoning": "Accurate"})
    scores = run_judge("q", "a", "c", registry_with(lambda t: reply))
    assert scores.faithfulness == 0.85 and scores.citation_accuracy == 0.9 and scores.error is None


def test_judge_failure_is_none_not_zero():
    scores = run_judge("q", "a", "c", registry_with(lambda t: "not json at all"))
    assert scores.faithfulness is None and scores.citation_accuracy is None
    assert "not valid JSON" in scores.error


def test_judge_unavailable_is_reported():
    scores = run_judge("q", "a", "c", registry_with(lambda t: Exception("connection refused")))
    assert scores.faithfulness is None and "unavailable" in scores.error


def test_judge_prompt_includes_the_question():
    seen = []
    run_judge("What about metformin?", "a", "c", registry_with(lambda t: seen.append(t) or "{}"))
    assert "What about metformin?" in seen[0]


class FakePipeline:
    def ask(self, question):
        return {"answer": "A", "confidence": None, "citations": ["NCT00000001"], "verified_citations": ["NCT00000001"],
                "unverified_citations": [], "total_latency_s": 1.0, "context_chunks": ["ctx"]}

    def judge(self, question, answer, contexts):
        return {"faithfulness": 0.9, "citation_accuracy": 1.0}


def test_harness_runs_cases_without_sleeping_by_default(monkeypatch):
    monkeypatch.delenv("EVAL_DELAY_SECONDS", raising=False)
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    results = run_evaluation_suite([{"question": "q1"}, {"question": "q2"}], pipeline=FakePipeline())
    assert [r["question"] for r in results] == ["q1", "q2"] and slept == []
    assert results[0]["eval_scores"]["faithfulness"] == 0.9


def test_harness_delay_is_configurable(monkeypatch):
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    run_evaluation_suite([{"question": "q1"}, {"question": "q2"}], pipeline=FakePipeline(), delay_s=3)
    assert slept == [3]
