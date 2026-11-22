"""Tool selection: does the planner model pick the right tool? (planning only, fast)

  inspect eval evals/tool_selection.py --max-samples 4
  inspect eval evals/tool_selection.py -T dataset=evals/datasets/tool_queries.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_ai import Task, task  # noqa: E402

from evals.common import (  # noqa: E402
    load_dataset, pipeline_info, plan_solver, tool_selection_scorer, valid_plan_scorer,
)


@task
def tool_selection(dataset: str = "evals/datasets/sample_tool_queries.jsonl"):
    return Task(
        dataset=load_dataset(dataset),
        solver=plan_solver(),
        scorer=[tool_selection_scorer(), valid_plan_scorer()],
        model="mockllm/model",
        metadata={"pipeline": pipeline_info()},
    )
