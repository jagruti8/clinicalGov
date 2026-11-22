# Evaluating with Inspect

[Inspect](https://inspect.aisi.org.uk/) is an evaluation framework. It runs a set of questions through your system, scores the results, logs everything, and lets you resume or compare runs. This guide covers the parts you need.

## The three pieces

```
Dataset  ->  Solver  ->  Scorer(s)  ->  Metrics
questions    runs our     turn the      average the
(JSONL)      pipeline     result into   scores over all
                          numbers       questions
```

- **Dataset**: samples with an `input` (the question), optionally a `target` (the expected answer) and `metadata`. Ours are JSONL files.
- **Solver**: the thing being evaluated. Ours calls `ClinicalPipeline` and stores the result on the sample (`state.metadata`). See `evals/common.py`.
- **Scorer**: reads the stored result and returns a `Score`. A score is a number, `CORRECT`/`INCORRECT`, or a dict of named numbers.
- **Metric**: aggregates scores over samples (`mean()`, `accuracy()`, `stderr()`, or our own `median_value()` and `p95_value()`).
- **Task**: a dataset, a solver, and scorers bundled together. Each file in `evals/` defines one.

A score of `NaN` means "unscored". Inspect leaves it out of the averages and reports the count. We use this for judge failures, so a broken judge call never counts as a zero.

## Why `--model` is not used

Our pipeline calls Ollama itself, with the models set in `config/models.yaml`. Inspect only runs it as a solver. Each task therefore defaults to Inspect's built-in placeholder `mockllm/model`, and you do not pass `--model`. The log metadata records which Ollama models were configured.

## The three tasks

| Task | Measures | Dataset | Speed |
|---|---|---|---|
| `answer_quality` | judge faithfulness and citation accuracy, deterministic citation verification, evidence support, latency (median, p95) | `answer_queries.jsonl` | slow (full pipeline) |
| `tool_selection` | does the planner pick the right tool, and is its output valid | `tool_queries.jsonl` | fast (planning only) |
| `retrieval_check` | does the API query surface the expected trial | `retrieval_queries.jsonl` | medium (no synthesis) |

`faithfulness` and `citation_accuracy` come from the judge model (an LLM). `verified_citation_rate` and `evidence_support_rate` are computed by code, so they do not depend on a judge. Report both and say which is which.

## Quick start

```bash
pip install -r requirements-eval.txt
python main.py --smoke                     # check Ollama, models, network first

# 1. try the tiny sample sets
inspect eval evals/tool_selection.py --max-samples 4
inspect eval evals/answer_quality.py --limit 3 --max-samples 2

# 2. generate the real datasets (needs ClinicalTrials.gov access)
python -m evals.generate_queries --out-dir evals/datasets --seed 1

# 3. the real runs
inspect eval evals/tool_selection.py  -T dataset=evals/datasets/tool_queries.jsonl
inspect eval evals/retrieval_check.py -T dataset=evals/datasets/retrieval_queries.jsonl --max-samples 4
inspect eval evals/answer_quality.py  -T dataset=evals/datasets/answer_queries.jsonl --max-samples 4 --log-dir logs/run1

# 4. look at results
inspect view                               # web UI over ./logs
```

## Flags worth knowing

| Flag | Use |
|---|---|
| `--limit 20` | run only the first 20 samples (always do this first) |
| `--max-samples 4` | samples running at once. Match it to what Ollama can serve in parallel (`OLLAMA_NUM_PARALLEL`) |
| `--log-dir logs/run1` | where logs go. One directory per run keeps comparisons clean |
| `-T dataset=path.jsonl` | pass a task argument (here, which dataset) |
| `--epochs 3` | repeat every sample, to measure run-to-run variation |
| `inspect eval-retry logs/run1/<file>.eval` | resume a run that crashed or was interrupted. Finished samples are kept |

## Large runs on the cluster

- **Resume**: a crash does not lose finished samples. Use `inspect eval-retry`.
- **Reproducibility**: set `CTGOV_CACHE_MODE=readwrite` for the first run. It saves every API response to `CTGOV_CACHE_DIR`. Later runs with `CTGOV_CACHE_MODE=readonly` use only those snapshots, so the same questions see the same trials.
- **Dataset**: `generate_queries` writes `seed_trials.json` and `manifest.json`. Keep them with your results.
- **Held-out set**: the tool-selection questions come from conditions the answer set never uses.
- **Latency**: the pipeline records seconds per node. The `latency_scorer` reports median and p95 end to end, and the per-node breakdown is in each sample's explanation.
- **Concurrency changes latency**: running 4 samples at once on one GPU slows each one. Measure latency with `--max-samples 1`.

## Reading results

A run is only comparable to another when these match: the model tags in `config/models.yaml`, the dataset and its seed, the pipeline settings (critique mode, revisions), the cache mode, and the code version. Inspect logs the first of these in the task metadata. Record the rest next to the numbers you report.

## Adding a scorer

```python
@scorer(metrics=[mean()])
def my_check():
    async def score(state, target):
        result = state.metadata["result"]          # the pipeline result dict
        return Score(value=1.0 if "NCT" in result["answer"] else 0.0)
    return score
```

Add it to the task's `scorer=[...]` list. Metrics must annotate their argument as `list[SampleScore]` (see `median_value` in `common.py`), or Inspect treats them as a deprecated kind.
