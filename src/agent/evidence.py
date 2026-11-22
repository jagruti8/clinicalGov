"""Evidence agent: maps claims to trial text and verifies citations.

The LLM only extracts (claim, NCT ID, quote) triples. Whether a citation exists in the
retrieved data and whether a quote is supported are decided by code.
"""
import re

from src.agent.deps import Deps
from src.agent.prompts import EVIDENCE_PROMPT
from src.agent.state import RAGState
from src.agent.tools import find_nct_ids, parse_json_loose
from src.llm.fallback import AllModelsFailed
from src.retrieval.ctgov_client import trial_id_from_text

MAX_CLAIMS = 25
QUOTE_OVERLAP = 0.85
MIN_QUOTE_WORDS = 3


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def quote_supported(quote: str, trial_text: str) -> bool:
    """A quote counts as supported when at least 85% of its words (3+ letters)
    appear in the trial text. Tolerates light paraphrase; rejects invented content."""
    words = [w for w in _words(quote) if len(w) >= 3]
    if len(words) < MIN_QUOTE_WORDS:
        return False
    available = set(_words(trial_text))
    return sum(w in available for w in words) / len(words) >= QUOTE_OVERLAP


def evidence_node(state: RAGState, deps: Deps) -> dict:
    answer = state.get("draft_answer", "")
    chunks = state.get("relevant_chunks", [])
    context_ids = set(state.get("context_nct_ids") or [])
    warnings = list(state.get("warnings") or [])
    models_used = dict(state.get("models_used") or {})

    cited = find_nct_ids(answer)
    verified = [i for i in cited if i in context_ids]
    unverified = [i for i in cited if i not in context_ids]

    claims: list[dict] = []
    if answer and chunks:
        context = "\n\n---\n\n".join(chunks)[:deps.config.pipeline.max_context_chars]
        try:
            messages = EVIDENCE_PROMPT.format_messages(context=context, answer=answer)
            result = deps.registry.call(deps.config.roles.evidence, messages)
            models_used["evidence"] = result.model
            parsed = parse_json_loose(result.text) or {}
            claims = [c for c in parsed.get("claims", []) if isinstance(c, dict)][:MAX_CLAIMS]
        except AllModelsFailed as exc:
            warnings.append(f"evidence extraction failed: {str(exc)[:150]}")

    text_by_id = {trial_id_from_text(c): c for c in chunks}
    evidence = []
    for claim in claims:
        nct_id = str(claim.get("nct_id", "")).upper()
        text = text_by_id.get(nct_id)
        if text is None:
            supported, reason = False, "NCT ID not in the retrieved trial data"
        elif quote_supported(str(claim.get("quote", "")), text):
            supported, reason = True, ""
        else:
            supported, reason = False, "quote not found in the trial data"
        evidence.append({
            "claim": str(claim.get("claim", ""))[:300], "nct_id": nct_id,
            "quote": str(claim.get("quote", ""))[:300], "supported": supported, "reason": reason,
        })

    rate = sum(e["supported"] for e in evidence) / len(evidence) if evidence else None
    return {
        "evidence": evidence,
        "evidence_support_rate": rate,
        "citations": cited,
        "verified_citations": verified,
        "unverified_citations": unverified,
        "warnings": warnings,
        "models_used": models_used,
    }
