"""LangGraph wiring.

Agents (node groups):
  orchestrator  start, route, finalize
  retrieval     plan_tools, execute_tools, retrieve, relevance_check, reformulate,
                synthesize, score_agreement
  evidence      evidence
  critique      critique  (may send the draft back to retrieval.synthesize)

Modes: "full" (default), "retrieval_only" (stops before synthesis),
       "plan_only" (stops after tool planning). The last two serve evaluation.
"""
from functools import partial

from langgraph.graph import END, START, StateGraph

from src.agent import critique, evidence, nodes, orchestrator
from src.agent.deps import Deps
from src.agent.state import RAGState
from src.observability import timed


def build_graph(deps: Deps, mode: str = "full"):
    if mode not in {"full", "retrieval_only", "plan_only"}:
        raise ValueError(f"Unknown graph mode: {mode}")

    graph = StateGraph(RAGState)

    def add(name: str, label: str, fn):
        graph.add_node(name, timed(label, partial(fn, deps=deps)))

    add("start", "orchestrator.start", orchestrator.start_node)
    add("plan_tools", "retrieval.plan_tools", nodes.plan_tools_node)
    add("route", "orchestrator.route", orchestrator.route_node)

    graph.add_edge(START, "start")
    graph.add_edge("start", "plan_tools")
    graph.add_edge("plan_tools", "route")

    if mode == "plan_only":
        graph.add_edge("route", END)
        return graph.compile()

    add("execute_tools", "retrieval.execute_tools", nodes.execute_tools_node)
    add("retrieve", "retrieval.retrieve", nodes.retrieve_node)
    add("relevance_check", "retrieval.relevance_check", nodes.relevance_check_node)
    add("reformulate", "retrieval.reformulate", nodes.reformulate_node)

    graph.add_edge("route", "execute_tools")
    graph.add_edge("execute_tools", "retrieve")
    graph.add_edge("retrieve", "relevance_check")
    graph.add_edge("reformulate", "execute_tools")

    after_relevance = partial(nodes.route_after_relevance, deps=deps)
    if mode == "retrieval_only":
        graph.add_conditional_edges(
            "relevance_check", after_relevance, {"synthesize": END, "reformulate": "reformulate"})
        return graph.compile()

    add("synthesize", "retrieval.synthesize", nodes.synthesize_node)
    add("score_agreement", "retrieval.score_agreement", nodes.score_agreement_node)
    add("evidence", "evidence.evidence", evidence.evidence_node)
    add("critique", "critique.critique", critique.critique_node)
    add("finalize", "orchestrator.finalize", orchestrator.finalize_node)

    graph.add_conditional_edges(
        "relevance_check", after_relevance, {"synthesize": "synthesize", "reformulate": "reformulate"})
    graph.add_edge("synthesize", "score_agreement")
    graph.add_edge("score_agreement", "evidence")
    graph.add_edge("evidence", "critique")
    graph.add_conditional_edges(
        "critique", partial(critique.route_after_critique, deps=deps),
        {"synthesize": "synthesize", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()


def run_query(question: str) -> RAGState:
    """Convenience entry point: runs one question through a shared default pipeline."""
    from src.pipeline import get_default_pipeline

    return get_default_pipeline().run_state(question)
