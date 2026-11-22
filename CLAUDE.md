# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Agentic RAG pipeline that answers natural language questions about clinical trials:
1. A planner model picks tools (search_trials, get_trial, compare_trials, get_eligibility) against the live ClinicalTrials.gov v2 API
2. Search results are embedded into ChromaDB and ranked, restricted to the trials just fetched
3. Weak retrieval triggers query reformulation (max 2)
4. All configured synthesizer models answer in parallel (retry and fallback per call)
5. Agreement across models is scored (null with one model)
6. An evidence step verifies citations and quotes in code; a critique step may request one revision
7. Evaluation: Inspect tasks in evals/ and a quick judge harness (`--eval`)

## Setup and commands

```bash
cp .env.example .env
conda create -n clinicalGov python=3.11 -y && conda activate clinicalGov
conda install -c conda-forge cryptography openssl -y
pip install -r requirements.txt            # requirements-eval.txt adds Inspect

python main.py --smoke                     # check config, Ollama, network, embeddings, one query
python main.py "question"                  # ask (--json for the full result)
python main.py --serve                     # REST API
pytest tests/ -v                           # mocked, no services needed
inspect eval evals/tool_selection.py       # see evals/README.md
```

## Architecture

```
main.py
src/config.py              YAML config (models, roles, pipeline) via pydantic
src/pipeline.py            ClinicalPipeline: the one facade CLI, API, and evals use
src/observability.py       per-node timing, optional JSONL log
src/agent/
  graph.py                 LangGraph wiring; modes: full | retrieval_only | plan_only
  state.py                 RAGState
  deps.py                  Deps (registry, vector store, ctgov client, embedder), injectable
  orchestrator.py          start, route (intent), finalize (authoritative citation check)
  nodes.py                 retrieval agent: plan_tools, execute_tools, retrieve, relevance, reformulate, synthesize, agreement
  evidence.py              claim extraction (LLM) and verification (code)
  critique.py              code checks plus optional LLM review; revision routing
  tools.py                 tool registry, Pydantic arg models, JSON/native routing helpers
  prompts.py               all prompt templates
src/llm/
  providers.py             build_chat_model (ollama, openai_compatible, groq, gemini) and ModelRegistry
  fallback.py              retry with backoff, fallback chain, <think> stripping
  aggregator.py            pairwise agreement
src/retrieval/             ctgov_client.py (live API, paging, retry, snapshot cache), vector_store.py
src/evaluation/            judge.py, harness.py (quick CLI eval)
src/api/                   FastAPI app and schemas
src/smoke.py               --smoke checks
evals/                     Inspect tasks, dataset generator, README
config/                    models.yaml (Ollama), models.cloud.yaml (Groq + Gemini)
```

## Key design decisions

- **Config-driven models.** Nothing in code names a model. Roles (planner, synthesizers, evidence, critic, judge) and the fallback chain are in YAML. Provider libraries are imported lazily.
- **Dependency injection.** Graph nodes are `fn(state, deps)`. Tests pass fakes (see tests/conftest.py); production uses `build_default_deps`.
- **Code decides verification.** NCT citation existence and quote support are computed by code. LLMs only extract claims and give critique.
- **`tool_calls_llm` vs `tool_plan`.** The first is what the planner model produced (scored by tool-selection evals); the second is what runs after the NCT-ID override (`nct_routing_override`).
- **Planner repair.** If the planner reply has no usable tool call, it gets `plan_repair_retries` more tries (default 1), each told what was wrong. After that a heuristic search runs (free text plus any phase or status the question states). `plan_attempts` and `plan_raw` record what happened. Set `plan_repair_retries: 0` to measure the bare model's valid-plan rate.
- **Population mismatches.** The synthesis prompt tells models to flag or omit trials whose population differs from the question (prediabetes vs type 2 diabetes). Trials are never dropped automatically on a critic's flag: small critics are often wrong. Unresolved review issues appear in `warnings` and `critique`, not in the answer text.
- **Retrieval is restricted to the current fetch.** ChromaDB is a cache plus ranking index; results never come from unrelated cached trials. Lookups by NCT ID bypass it.
- **Single-model agreement is `None`,** not 1.0.
- **Judge failures are `None`/NaN,** never 0.0, so they do not pollute averages.
- **Ollama context.** `num_ctx` is set in config because Ollama silently truncates long prompts.
- **urllib over httpx for ClinicalTrials.gov:** httpx was blocked by the site's WAF.
- **aggFilters API format:** `aggFilters=phase:3,status:rec`, not the old `filter.phase` parameters.
- **Snapshot cache** (`CTGOV_CACHE_MODE`) makes eval runs reproducible.
- **RAGAS removed:** dependency conflicts with langchain. LLM-as-judge covers faithfulness.

## Known issues / things to verify

- The status filter code for NOT_YET_RECRUITING is `not-rec` in `ctgov_client._STATUS_MAP`. Unverified against the live API; confirm it before relying on the not-yet-recruiting filter.
- Nothing here has been run against live Ollama or ClinicalTrials.gov from CI. Run `python main.py --smoke` after any config change.
- After changing `eligibility_chars`, delete `chroma_db/`: cached trial text keeps the old length.
- Intel Mac: torch<=2.2.2 and `numpy<2`; sentence-transformers pinned `>=2.2,<3.3`.
- Gemini free tier: use gemini-2.5-flash (2.0 variants show limit:0 on some accounts). Groq retires models regularly; check `config/models.cloud.yaml`.

## Environment variables

| Variable | Purpose |
|---|---|
| `OLLAMA_BASE_URL` | Ollama endpoint (default http://localhost:11434) |
| `MODELS_CONFIG` | alternate models YAML |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | only for config/models.cloud.yaml |
| `CTGOV_CACHE_MODE`, `CTGOV_CACHE_DIR` | off, readwrite, or readonly snapshot cache |
| `PIPELINE_LOG_PATH` | JSONL per-node timing events |
| `EVAL_DELAY_SECONDS` | pause between cases in `--eval` |
