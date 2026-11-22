# ClinicalGov: agentic RAG over ClinicalTrials.gov

Answers natural language questions about clinical trials with cited, verified answers. Retrieval is live against the [ClinicalTrials.gov v2 API](https://clinicaltrials.gov/data-api/api) (no auth). Models are configured in one YAML file, so Ollama models, cloud models, or any mix can be swapped without touching code.

## What it does

- **Tool-based retrieval.** A planner model picks tools (`search_trials`, `get_trial`, `compare_trials`, `get_eligibility`) with schema-validated arguments. It works through prompted JSON routing on any model, or native tool calling where the model supports it.
- **Live retrieval with paging, retry, and ranking.** Trials are fetched live, embedded into ChromaDB, and ranked semantically. Retrieval is restricted to the trials just fetched, with a small boost for matching phase and status. Weak results trigger query reformulation (max 2).
- **Multi-model synthesis.** Every configured synthesizer answers in parallel with retry and fallback. Agreement is the mean pairwise cosine similarity of the answers (reported as `null` with one model).
- **Verification.** Citations are checked by code against the trials the models actually saw. Claims are mapped to quotes from the trial data. A critique step can send the draft back for one revision. Citations that stay unverified are tagged `[NCTxxxxxxxx unverified]` in the answer.
- **Observability.** Seconds per node and which model answered each step are returned with every answer.

Limits worth knowing: recall is bounded by the API's own search, because only trials the API returns can be ranked. Agreement between models is a consistency signal, not a correctness guarantee.

## Agents

```
question
  |
Orchestrator   start -> (plan) -> route by intent -> ... -> finalize
  |
Retrieval      plan_tools -> execute_tools -> retrieve -> relevance_check
  |                                  ^                          |
  |                                  +-- reformulate (max 2) <--+ weak
  |            synthesize (all models in parallel) -> score_agreement
  |
Evidence       claims -> quotes (LLM extracts; code verifies citations and quotes)
  |
Critique       code checks (+ LLM review in `full` mode)
  |              +-- revise: back to synthesize with feedback (max_revisions)
  v
Orchestrator   finalize: tag unverified citations, total latency
```

Lookups by NCT ID skip the vector store (the question already names the trials) and use code-only critique.

## Setup

```bash
cp .env.example .env
conda create -n clinicalGov python=3.11 -y && conda activate clinicalGov
conda install -c conda-forge cryptography openssl -y
pip install -r requirements.txt            # add requirements-eval.txt for Inspect
```

Edit `config/models.yaml`: set each model's `model:` to a tag from `ollama list`. For the Groq and Gemini free tiers use `MODELS_CONFIG=config/models.cloud.yaml` and set `GROQ_API_KEY` and `GEMINI_API_KEY`.

Check everything before a real run:

```bash
python main.py --smoke          # config, Ollama, models, ClinicalTrials.gov, embeddings, one query
python main.py --smoke --quick  # skip the LLM and full-pipeline checks
```

## Usage

```bash
python main.py "What phase 3 trials are recruiting for type 2 diabetes?"
python main.py "Compare NCT04280705 and NCT03456789"
python main.py --json "..."     # full result: citations, evidence, timings, models used
python main.py                  # interactive
python main.py --eval           # quick judge-based check on 3 cases
python main.py --serve          # REST API on :8000
```

Not sure what to ask? [docs/sample_queries.md](docs/sample_queries.md) has synthetic queries for each kind (condition search, drug search, NCT lookup, eligibility, comparison, vague, no-result, bad ID, out of scope, messy input), with what to check in each output.

## Changing models

Everything lives in `config/models.yaml` (laptop profile). `config/models.cluster.yaml` is the cluster profile: larger context, `retrieval_k: 5`, longer timeouts. Set your Ollama tags and select it:

```bash
export NEMOTRON_MODEL=<tag> DEEPSEEK_MODEL=<tag> CRITIC_MODEL=<tag>   # see `ollama list`
export MODELS_CONFIG=config/models.cluster.yaml
rm -rf chroma_db                      # first time only
python main.py --smoke
```

Example of the format:

```yaml
models:
  nemotron: {model: nemotron-mini}
  deepseek: {model: deepseek-r1:14b, strip_think: true}
roles:
  planner: nemotron
  synthesizers: [nemotron, deepseek]     # 1 or more
  critic: deepseek
  judge: deepseek
fallback: [nemotron, deepseek]
```

- Use one synthesizer: delete a name from `synthesizers`. Agreement then reports `null`.
- Swap a model: edit its `model:` tag. Add one: add an entry and reference it in a role.
- Keep `critic` and `judge` different from the primary synthesizer when you can. A model grading its own output is biased.
- Other endpoints (vLLM, TGI, LM Studio): `provider: openai_compatible` with `base_url`.
- Ollama truncates long prompts silently, so `num_ctx` is set (default 8192). Raise it if you raise `retrieval_k` or `max_context_chars`.
- `critique_mode`: `full` (code checks and LLM review), `code_only`, or `off`.

Changing models or pipeline settings changes the evaluation numbers, so rerun the evals.

## REST API

`python main.py --serve`, then open `http://localhost:8000/docs`. The schema is at `/openapi.json`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/ask` | full answer with citations, evidence, timings, models used |
| `POST /v1/plan` | which tools the planner would call (no fetching) |
| `POST /v1/retrieve` | structured search plus semantic ranking, no LLM |
| `POST /v1/tools/{name}` | run one tool with validated arguments |
| `POST /v1/evaluate/judge` | judge scores for an answer and its contexts |
| `GET /v1/config` | active model roles (no secrets) |
| `GET /healthz` | liveness |

Invalid input returns 422 with the reason. Upstream ClinicalTrials.gov failures return 502.

## Evaluation

Large, resumable, logged runs use [Inspect](https://inspect.aisi.org.uk/). See `evals/README.md` for a walkthrough.

```bash
python -m evals.generate_queries --out-dir evals/datasets --seed 1
inspect eval evals/answer_quality.py -T dataset=evals/datasets/answer_queries.jsonl --max-samples 4
inspect eval evals/tool_selection.py -T dataset=evals/datasets/tool_queries.jsonl
inspect eval evals/retrieval_check.py -T dataset=evals/datasets/retrieval_queries.jsonl
```

Reproducible runs: set `CTGOV_CACHE_MODE=readwrite` once, then `readonly`, so every run sees the same trials.

## Docker

```bash
docker compose run --rm app "your question"      # CLI
docker compose up api                            # REST API on :8000
docker build --build-arg INSTALL_EVAL=1 -t clinicalgov-eval .
docker run --rm jagruti8/clinicalgov --smoke
```

Containers reach Ollama through `OLLAMA_BASE_URL` (default `host.docker.internal:11434`).

## Tests

```bash
pytest tests/ -v        # mocked: no network, no Ollama, no embedding model
```

## CI/CD

GitHub Actions runs the unit tests on pushes to `main` and `dev` and on pull requests to `main`. The live evaluation is manual (workflow dispatch) and uses the cloud config, because CI has no Ollama. The CD workflow builds and pushes the Docker image on pushes to `main`.
