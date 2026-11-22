"""Shared fakes: no network, no Ollama, no embedding model, no Chroma install needed."""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.deps import Deps  # noqa: E402
from src.config import AppConfig  # noqa: E402
from src.llm.providers import ModelRegistry  # noqa: E402

TRIALS = [
    {"nct_id": "NCT00000001", "title": "Metformin for Type 2 Diabetes", "status": "RECRUITING",
     "phases": ["PHASE3"], "conditions": ["Type 2 Diabetes"], "interventions": ["Metformin"],
     "summary": "A study of metformin efficacy in adults with type 2 diabetes.",
     "eligibility": "Adults aged 18 to 65 with type 2 diabetes."},
    {"nct_id": "NCT00000002", "title": "Insulin Glargine in Type 2 Diabetes", "status": "COMPLETED",
     "phases": ["PHASE2"], "conditions": ["Type 2 Diabetes"], "interventions": ["Insulin glargine"],
     "summary": "Insulin glargine compared with placebo for glycemic control.",
     "eligibility": "HbA1c above 7 percent."},
    {"nct_id": "NCT00000003", "title": "Pembrolizumab in Melanoma", "status": "RECRUITING",
     "phases": ["PHASE3"], "conditions": ["Melanoma"], "interventions": ["Pembrolizumab"],
     "summary": "Pembrolizumab versus ipilimumab in advanced melanoma.",
     "eligibility": "Unresectable stage III or IV melanoma."},
]


def make_config(**pipeline) -> AppConfig:
    return AppConfig(
        models={
            "nemo": {"provider": "ollama", "model": "nemo-tag", "name": "nemo"},
            "deep": {"provider": "ollama", "model": "deep-tag", "strip_think": True},
        },
        roles={"planner": "nemo", "synthesizers": ["nemo", "deep"], "evidence": "nemo",
               "critic": "deep", "judge": "deep"},
        fallback=["nemo", "deep"],
        pipeline={"retry_backoff_s": 0, "relevance_threshold": 0.05, **pipeline},
    )


def prompt_text(messages) -> str:
    return "\n".join(getattr(m, "content", None) or (m[1] if isinstance(m, tuple) else str(m)) for m in messages)


class ScriptedLLM:
    """Minimal chat model: invoke(messages) -> object with .content."""

    def __init__(self, fn, tool_calls=None):
        self.fn = fn
        self.calls = []
        self.tool_calls = tool_calls

    def invoke(self, messages):
        text = prompt_text(messages)
        self.calls.append(text)
        reply = self.fn(text)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply, tool_calls=self.tool_calls or [])

    def bind_tools(self, schemas):
        return self


class Brain:
    """Scripted behavior for every prompt type. Flip attributes to simulate failures."""

    def __init__(self):
        self.bad_citation_until_feedback = False   # first draft cites a trial not in context
        self.always_bad_citation = False
        self.critic_issue = None                   # LLM critic asks for a revision with this issue
        self.router_reply = None                   # override the tool-router reply
        self.reformulated = False
        self.critic_fired = False

    @staticmethod
    def _ids(text):
        return re.findall(r"Trial ID: (NCT\d{8})", text)

    def respond(self, text: str) -> str:
        if "route questions about clinical trials" in text:
            if self.router_reply is not None:
                reply = self.router_reply
                return reply(text) if callable(reply) else reply
            question = text.split("\n")[-1]
            ids = re.findall(r"NCT\d{8}", question)
            if len(ids) >= 2:
                return json.dumps({"tool_calls": [{"name": "compare_trials", "args": {"nct_ids": ids}}]})
            if ids:
                return json.dumps({"tool_calls": [{"name": "get_trial", "args": {"nct_id": ids[0]}}]})
            return json.dumps({"tool_calls": [{"name": "search_trials",
                                               "args": {"condition": "type 2 diabetes", "phase": "PHASE3"}}]})
        if "returned no relevant results" in text:
            self.reformulated = True
            return json.dumps({"condition": "type 2 diabetes", "intervention": None, "phase": None, "status": None})
        if "clinical research assistant" in text:
            ids = self._ids(text)
            has_feedback = "A reviewer found problems" in text
            if self.always_bad_citation or (self.bad_citation_until_feedback and not has_feedback):
                return "Metformin was studied [NCT99999999]."
            first = ids[0] if ids else "NCT00000001"
            return f"Trial [{first}] is a study of metformin efficacy in adults."
        if "List every factual claim" in text:
            ids = self._ids(text) or ["NCT00000001"]
            return json.dumps({"claims": [{"claim": "metformin efficacy study", "nct_id": ids[0],
                                           "quote": "study of metformin efficacy in adults"}]})
        if "review answers about clinical trials" in text:
            if self.critic_issue and not self.critic_fired:
                self.critic_fired = True               # ask for one revision, then be satisfied
                return json.dumps({"verdict": "revise", "issues": [self.critic_issue]})
            return json.dumps({"verdict": "pass", "issues": []})
        if "expert evaluator" in text:
            return json.dumps({"faithfulness": 0.9, "citation_accuracy": 1.0,
                               "faithfulness_reasoning": "ok", "citation_reasoning": "ok"})
        return "OK"


class FakeCtgov:
    """Stands in for src.retrieval.ctgov_client."""

    def __init__(self, trials=None):
        self.trials = TRIALS if trials is None else trials
        self.search_calls, self.get_calls = [], []

    def fetch_trials(self, params, max_results=None):
        self.search_calls.append((params, max_results))
        needle = " ".join(str(params.get(k, "")) for k in ("query.cond", "query.intr", "query.term")).lower()
        if not needle.strip():
            return []
        words = [w for w in re.findall(r"[a-z0-9]+", needle) if len(w) > 3]
        hits = [t for t in self.trials
                if any(w in (" ".join(t["conditions"] + t["interventions"]) + t["summary"]).lower() for w in words)]
        return hits[:max_results] if max_results else hits

    def get_trial(self, nct_id):
        self.get_calls.append(nct_id)
        return next((t for t in self.trials if t["nct_id"] == nct_id), None)


class FakeChroma:
    """Implements the slice of the Chroma API the code uses, with bag-of-words cosine."""

    def __init__(self):
        self.docs = {}
        self._collection = SimpleNamespace(get=lambda include=None: {"ids": list(self.docs)})

    def add_documents(self, docs, ids):
        for doc, doc_id in zip(docs, ids):
            self.docs[doc_id] = doc

    @staticmethod
    def _vec(text):
        return Counter(re.findall(r"[a-z0-9]+", text.lower()))

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None):
        allowed = filter["nct_id"]["$in"] if filter else None
        q = self._vec(query)
        scored = []
        for doc_id, doc in self.docs.items():
            if allowed is not None and doc_id not in allowed:
                continue
            d = self._vec(doc.page_content)
            dot = sum(q[w] * d[w] for w in q)
            norm = math.sqrt(sum(v * v for v in q.values())) * math.sqrt(sum(v * v for v in d.values()))
            scored.append((doc, dot / norm if norm else 0.0))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


class BowEmbedder:
    def encode(self, texts):
        vectors = np.zeros((len(texts), 64))
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vectors[row, hash(word) % 64] += 1
        return vectors


@pytest.fixture
def brain():
    return Brain()


@pytest.fixture
def ctgov():
    return FakeCtgov()


@pytest.fixture
def make_deps(brain, ctgov):
    def build(config=None, per_model=None, store=None):
        config = config or make_config()
        llms = {name: ScriptedLLM((per_model or {}).get(name, brain.respond)) for name in config.models}
        registry = ModelRegistry(config, builder=lambda spec: llms[spec.name], sleep=lambda s: None)
        deps = Deps(registry=registry, vector_store=store or FakeChroma(), ctgov=ctgov, embedder=BowEmbedder())
        deps.llms = llms
        return deps
    return build
