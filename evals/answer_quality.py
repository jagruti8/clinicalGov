"""Answer quality: faithfulness and citation accuracy (judge), deterministic citation
checks, and latency.

  inspect eval evals/answer_quality.py --limit 5 --max-samples 2
  inspect eval evals/answer_quality.py -T dataset=evals/datasets/answer_queries.jsonl --max-samples 4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_ai import Task, task  # noqa: E402

from evals.common import (  # noqa: E402
    ask_solver, citation_scorer, judge_scorer, latency_scorer, load_dataset, pipeline_info,
)


@task
def answer_quality(dataset: str = "evals/datasets/sample_answer_queries.jsonl"):
    return Task(
        dataset=load_dataset(dataset),
        solver=ask_solver(),
        scorer=[judge_scorer(), citation_scorer(), latency_scorer()],
        model="mockllm/model",  # placeholder: the pipeline calls Ollama itself
        metadata={"pipeline": pipeline_info()},
    )
