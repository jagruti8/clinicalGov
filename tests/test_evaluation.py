from unittest.mock import MagicMock, patch

from src.evaluation.judge import JudgeScores, run_judge


def test_judge_scores_model():
    scores = JudgeScores(
        faithfulness=0.9,
        citation_accuracy=0.8,
        faithfulness_reasoning="All claims grounded in context.",
        citation_reasoning="Citations correctly attributed.",
    )
    assert 0.0 <= scores.faithfulness <= 1.0
    assert 0.0 <= scores.citation_accuracy <= 1.0


def test_run_judge_returns_default_on_parse_error():
    mock_llm = MagicMock()

    with patch("src.evaluation.judge.JUDGE_PROMPT") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "not valid json {{}"
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        scores = run_judge("question", "answer", "context", mock_llm)
        assert isinstance(scores, JudgeScores)
        assert scores.faithfulness == 0.0
        assert "Failed" in scores.faithfulness_reasoning


def test_run_judge_parses_valid_json():
    valid_json = '{"faithfulness": 0.85, "citation_accuracy": 0.9, "faithfulness_reasoning": "Good", "citation_reasoning": "Accurate"}'
    mock_llm = MagicMock()

    with patch("src.evaluation.judge.JUDGE_PROMPT") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = valid_json
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        scores = run_judge("question", "answer", "context", mock_llm)
        assert scores.faithfulness == 0.85
        assert scores.citation_accuracy == 0.9
