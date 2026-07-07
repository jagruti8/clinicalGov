# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Modular RAG pipeline that answers natural language questions about clinical trials by:
1. Constructing structured queries over the ClinicalTrials.gov v2 REST API (no auth required)
2. Embedding retrieved trials into ChromaDB and retrieving semantically relevant chunks
3. Detecting weak retrieval and reformulating queries (up to 2 times) before synthesizing
4. Calling three LLMs in parallel: Groq Llama-3.3-70B (primary), Groq Llama-3.1-8B (secondary), Gemini 2.5 Flash (tertiary, optional)
5. Scoring cross-model agreement as a confidence signal (pairwise cosine similarity of response embeddings)
6. Running an LLM-as-judge evaluation harness (Gemini judges Llama's output for faithfulness + citation accuracy)

## Setup

```bash
cp .env.example .env   # fill in GROQ_API_KEY and GEMINI_API_KEY
conda create -n clinicalGov python=3.11 -y
conda activate clinicalGov
conda install -c conda-forge cryptography openssl -y
pip install -r requirements.txt
```

## Commands

```bash
# Ask a question
python main.py "What phase 3 trials are recruiting for type 2 diabetes?"

# Interactive mode
python main.py

# Run evaluation suite (3 built-in test cases)
python main.py --eval

# Custom test cases
python main.py --eval --test-cases my_cases.json

# Unit tests (no API keys needed)
pytest tests/ -v

# Docker
docker compose up --build
```

## Architecture

```
main.py
└── src/agent/graph.py          # LangGraph StateGraph — wires all nodes
    ├── src/agent/state.py      # RAGState TypedDict
    ├── src/agent/prompts.py    # All ChatPromptTemplates (with CoT in SYNTHESIS_PROMPT)
    └── src/agent/nodes.py      # Node functions (injected via functools.partial)
        ├── src/retrieval/ctgov_client.py   # CT.gov v2 API + trial text formatting
        ├── src/retrieval/vector_store.py   # ChromaDB + HuggingFace embeddings
        └── src/llm/
            ├── providers.py    # Groq (Llama-70B, Llama-8B) + Gemini LLM constructors
            └── aggregator.py   # Pairwise cosine agreement scoring across 3 models

src/evaluation/
├── judge.py     # LLM-as-judge via JUDGE_PROMPT → JudgeScores (Pydantic)
└── harness.py   # Orchestrates judge; run_evaluation_suite() with 20s delay between cases
```

### LangGraph flow

```
START → query_constructor → retriever → relevance_checker
                                              │
                              [relevant / max reformulations hit]
                                              ↓
                                  multi_llm_synthesizer
                                  (Llama-70B ∥ Llama-8B ∥ Gemini)
                                              ↓
                                    agreement_scorer → END
                              │
                         [not relevant]
                              ↓
                         reformulator → retriever  (loops, max 2×)
```

### Key design decisions

- **Dependency injection via `functools.partial`**: node functions accept `state` + typed deps (llm, vector_store); `partial` binds deps at graph-build time so LangGraph sees a single-arg callable.
- **ChromaDB deduplication**: trials are stored by `nct_id` as the document ID; `store_trials` checks existing IDs before embedding to avoid re-embedding on repeat queries.
- **Relevance threshold**: `RELEVANCE_THRESHOLD = 0.4` (cosine similarity) in `nodes.py` — tune this if reformulation triggers too aggressively or too rarely.
- **Three-model synthesis**: Llama-70B (primary), Llama-8B (secondary), Gemini (tertiary). Gemini is optional — if rate-limited it retries with backoff (15s, 30s) then silently skips. Only non-rate-limit errors (e.g. BadRequestError) skip immediately without retry.
- **Pairwise agreement scoring**: confidence = average cosine similarity across all pairs of available model responses. If only one model responds, confidence = 1.0.
- **Gemini-as-judge**: Gemini judges Llama-70B's output (not its own), avoiding self-evaluation bias. Judge runs only during `--eval`.
- **CoT in synthesis prompt**: SYNTHESIS_PROMPT instructs models to think step by step — first identify relevant trials, then reason through each, then synthesize a cited answer.
- **urllib over httpx for ClinicalTrials.gov**: httpx was blocked by the site's WAF; stdlib urllib works correctly.
- **aggFilters API format**: ClinicalTrials.gov v2 API uses `aggFilters=phase:3,status:rec` not the old `filter.phase` / `filter.overallStatus` parameters.
- **RAGAS removed**: ragas<0.2 conflicts with langchain>=0.3; ragas>=0.2 has a broken dependency on removed langchain_community.chat_models.vertexai. LLM-as-judge covers faithfulness evaluation instead.

## Environment variables

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | console.groq.com (free) |
| `GEMINI_API_KEY` | aistudio.google.com (free) — use gemini-2.5-flash, not 2.0-flash (limit:0 on free tier) |

## Known issues / compatibility

- **PyTorch**: Intel Mac (x86_64) only supports torch<=2.2.2 via pip. Use `numpy<2` to avoid ABI crash.
- **sentence-transformers**: pin to `>=2.2,<3.3` to stay compatible with torch 2.2.2.
- **Gemini free tier**: gemini-2.0-flash and gemini-2.0-flash-lite show limit:0 on some accounts. Use gemini-2.5-flash instead.
- **Groq model deprecations**: gemma2-9b-it and mixtral-8x7b-32768 are decommissioned. Use llama-3.1-8b-instant as secondary model.

## CI / GitHub Actions secrets

Add `GROQ_API_KEY` and `GEMINI_API_KEY` as repository secrets. Unit tests run without secrets; the `eval` job runs only on pushes to `main`.
