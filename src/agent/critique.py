"""Critique agent: code checks first, then (in `full` mode) an LLM review.

A revision is requested only when there is something specific to fix, and never more
than `max_revisions` times.
"""
from src.agent.deps import Deps
from src.agent.prompts import CRITIQUE_PROMPT
from src.agent.state import RAGState
from src.agent.tools import parse_json_loose
from src.llm.fallback import AllModelsFailed

MIN_SUPPORT_RATE = 0.7
MAX_LISTED = 3


def code_issues(state: RAGState) -> list[str]:
    issues = [
        f"The answer cites {nct_id}, which is not in the provided trial data. Remove or correct it."
        for nct_id in state.get("unverified_citations", [])
    ]
    rate = state.get("evidence_support_rate")
    if rate is not None and rate < MIN_SUPPORT_RATE:
        weak = [e["claim"] for e in state.get("evidence", []) if not e["supported"]][:MAX_LISTED]
        issues.append(
            f"Only {rate:.0%} of the claims match the trial data. Unsupported: " + "; ".join(weak)
        )
    return issues


def critique_node(state: RAGState, deps: Deps) -> dict:
    settings = deps.config.pipeline
    mode = state.get("critique_mode", settings.critique_mode)
    answer = state.get("draft_answer", "")
    revisions = state.get("revision_count", 0)
    warnings = list(state.get("warnings") or [])
    models_used = dict(state.get("models_used") or {})

    if mode == "off" or not answer:
        return {"critique": {"verdict": "pass", "issues": [], "source": "skipped"}}

    issues = code_issues(state)
    source = "code"

    if mode == "full":
        context = "\n\n---\n\n".join(state.get("relevant_chunks", []))[:settings.max_context_chars]
        try:
            messages = CRITIQUE_PROMPT.format_messages(
                question=state["question"], context=context, answer=answer,
                code_issues="\n".join(f"- {i}" for i in issues) or "none",
            )
            result = deps.registry.call(deps.config.roles.critic, messages)
            models_used["critic"] = result.model
            parsed = parse_json_loose(result.text) or {}
            llm_issues = [str(i) for i in parsed.get("issues", []) if str(i).strip()]
            if parsed.get("verdict") == "revise" and llm_issues:
                issues += llm_issues
                source = "code+llm"
        except AllModelsFailed as exc:
            warnings.append(f"critique model failed, code checks only: {str(exc)[:150]}")

    if not issues:
        return {"critique": {"verdict": "pass", "issues": [], "source": source}, "models_used": models_used,
                "warnings": warnings}

    if revisions < settings.max_revisions:
        return {
            "critique": {"verdict": "revise", "issues": issues, "source": source},
            "revision_count": revisions + 1,
            "models_used": models_used,
            "warnings": warnings,
        }

    warnings.append(f"unresolved review issues after {revisions} revision(s): " + " | ".join(issues)[:400])
    return {
        "critique": {"verdict": "revise", "issues": issues, "source": source, "unresolved": True},
        "models_used": models_used,
        "warnings": warnings,
    }


def route_after_critique(state: RAGState, deps: Deps) -> str:
    critique = state.get("critique") or {}
    if critique.get("verdict") == "revise" and not critique.get("unresolved"):
        return "synthesize"
    return "finalize"
