import json
import re

from pydantic import BaseModel

from src.agent.prompts import JUDGE_PROMPT


class JudgeScores(BaseModel):
    faithfulness: float
    citation_accuracy: float
    faithfulness_reasoning: str
    citation_reasoning: str


def run_judge(question: str, answer: str, context: str, judge_llm) -> JudgeScores:
    from langchain_core.output_parsers import StrOutputParser

    try:
        chain = JUDGE_PROMPT | judge_llm | StrOutputParser()
        raw = chain.invoke({"context": context, "answer": answer})
    except Exception as exc:
        print(f"[Warning] Judge LLM unavailable ({type(exc).__name__}), skipping judge scores.")
        return JudgeScores(
            faithfulness=0.0,
            citation_accuracy=0.0,
            faithfulness_reasoning="Judge unavailable",
            citation_reasoning="Judge unavailable",
        )

    cleaned = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")
    try:
        return JudgeScores(**json.loads(cleaned))
    except Exception:
        return JudgeScores(
            faithfulness=0.0,
            citation_accuracy=0.0,
            faithfulness_reasoning="Failed to parse judge output",
            citation_reasoning="Failed to parse judge output",
        )
