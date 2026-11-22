import re
from typing import Optional

from pydantic import BaseModel, field_validator

from src.agent.prompts import JUDGE_PROMPT
from src.agent.tools import parse_json_loose
from src.llm.fallback import AllModelsFailed


class JudgeScores(BaseModel):
    """Scores are None when the judge failed, so failures never count as zeros."""
    faithfulness: Optional[float] = None
    citation_accuracy: Optional[float] = None
    faithfulness_reasoning: str = ""
    citation_reasoning: str = ""
    error: Optional[str] = None

    @field_validator("faithfulness", "citation_accuracy")
    @classmethod
    def _clamp(cls, value):
        return None if value is None else max(0.0, min(1.0, float(value)))


def _score(value) -> Optional[float]:
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        value = float(match.group()) if match else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def run_judge(question: str, answer: str, context: str, registry) -> JudgeScores:
    """Score an answer for faithfulness and citation accuracy with the configured judge model."""
    messages = JUDGE_PROMPT.format_messages(question=question, context=context, answer=answer)
    try:
        result = registry.call(registry.config.roles.judge, messages)
    except AllModelsFailed as exc:
        return JudgeScores(error=f"judge unavailable: {str(exc)[:200]}")

    parsed = parse_json_loose(result.text)
    if not parsed:
        return JudgeScores(error="judge output was not valid JSON")
    faithfulness, citation = _score(parsed.get("faithfulness")), _score(parsed.get("citation_accuracy"))
    if faithfulness is None and citation is None:
        return JudgeScores(error="judge output had no scores")
    return JudgeScores(
        faithfulness=faithfulness,
        citation_accuracy=citation,
        faithfulness_reasoning=str(parsed.get("faithfulness_reasoning", "")),
        citation_reasoning=str(parsed.get("citation_reasoning", "")),
    )
