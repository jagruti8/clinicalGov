"""Retrieval: does the generated API query surface the expected trial?

  inspect eval evals/retrieval_check.py -T dataset=evals/datasets/retrieval_queries.jsonl --max-samples 4

Questions built from one seed trial are not always unique, so recall below 100% is
expected; compare runs against each other rather than against 1.0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_ai import Task, task  # noqa: E402

from evals.common import load_dataset, pipeline_info, retrieval_recall_scorer, retrieve_solver  # noqa: E402


@task
def retrieval_check(dataset: str = "evals/datasets/retrieval_queries.jsonl"):
    return Task(
        dataset=load_dataset(dataset),
        solver=retrieve_solver(),
        scorer=retrieval_recall_scorer(),
        model="mockllm/model",
        metadata={"pipeline": pipeline_info()},
    )
