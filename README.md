# ClinicalGov RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) pipeline that answers natural language questions about clinical trials using the [ClinicalTrials.gov v2 REST API](https://clinicaltrials.gov/data-api/api).

## Features

- Constructs structured queries over the ClinicalTrials.gov v2 API (no auth required)
- Embeds retrieved trials into ChromaDB and retrieves semantically relevant chunks
- Detects weak retrieval and reformulates queries (up to 2 times) before synthesizing
- Calls three LLMs in parallel: Groq Llama-3.3-70B, Groq Llama-3.1-8B, and Gemini 2.5 Flash
- Scores cross-model agreement as a confidence signal (pairwise cosine similarity)
- Runs an LLM-as-judge evaluation harness (Gemini judges Llama's output for faithfulness and citation accuracy)

## Setup

```bash
cp .env.example .env   # fill in GROQ_API_KEY and GEMINI_API_KEY
conda create -n clinicalGov python=3.11 -y
conda activate clinicalGov
conda install -c conda-forge cryptography openssl -y
pip install -r requirements.txt
```

### API Keys

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) (free) |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) (free) — use `gemini-2.5-flash` |

## Usage

```bash
# Ask a question
python main.py "What phase 3 trials are recruiting for type 2 diabetes?"

# Interactive mode
python main.py

# Run evaluation suite (3 built-in test cases)
python main.py --eval

# Custom test cases
python main.py --eval --test-cases my_cases.json
```

## Architecture

```
main.py
└── src/agent/graph.py          # LangGraph StateGraph — wires all nodes
    ├── src/agent/state.py      # RAGState TypedDict
    ├── src/agent/prompts.py    # ChatPromptTemplates
    └── src/agent/nodes.py      # Node functions
        ├── src/retrieval/ctgov_client.py   # CT.gov v2 API + trial text formatting
        ├── src/retrieval/vector_store.py   # ChromaDB + HuggingFace embeddings
        └── src/llm/
            ├── providers.py    # Groq + Gemini LLM constructors
            └── aggregator.py   # Pairwise cosine agreement scoring

src/evaluation/
├── judge.py     # LLM-as-judge → JudgeScores (Pydantic)
└── harness.py   # Orchestrates judge; run_evaluation_suite()
```

### LangGraph Flow

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

## Docker

The easiest way to run without any local setup:

```bash
# Pull from Docker Hub
docker pull jagruti8/clinicalgov:latest

# Ask a question
docker run -e GROQ_API_KEY=your_key -e GEMINI_API_KEY=your_key jagruti8/clinicalgov "What phase 3 trials are recruiting for type 2 diabetes?"

# Interactive mode
docker run -it -e GROQ_API_KEY=your_key -e GEMINI_API_KEY=your_key jagruti8/clinicalgov

# Persist ChromaDB across runs
docker run -v $(pwd)/chroma_db:/app/chroma_db -e GROQ_API_KEY=your_key -e GEMINI_API_KEY=your_key jagruti8/clinicalgov "your question"
```

Or build locally:

```bash
docker compose up --build
```

## Tests

```bash
pytest tests/ -v
```

Unit tests run without API keys.

## CI/CD

GitHub Actions runs unit tests on every push and pull request to `main`. The evaluation suite runs on pushes to `main` using `GROQ_API_KEY` and `GEMINI_API_KEY` repository secrets.
