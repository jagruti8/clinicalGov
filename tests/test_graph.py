import pytest

from src.agent.graph import build_graph
from src.pipeline import ClinicalPipeline
from tests.conftest import make_config


def run(deps, question, mode="full"):
    return build_graph(deps, mode).invoke({"question": question})


def test_search_question_end_to_end(make_deps):
    deps = make_deps()
    state = run(deps, "What phase 3 trials are recruiting for type 2 diabetes?")
    assert state["intent"] == "search"
    assert state["verified_citations"] == ["NCT00000001"] and state["unverified_citations"] == []
    assert state["agreement_models"] == 2 and state["confidence_score"] is not None
    assert state["critique"]["verdict"] == "pass" and state["revision_count"] == 0
    assert state["evidence_support_rate"] == 1.0
    assert state["total_latency_s"] >= 0 and state["errors"] == []


def test_every_node_is_timed(make_deps):
    state = run(make_deps(), "type 2 diabetes phase 3")
    for label in ("orchestrator.start", "retrieval.plan_tools", "orchestrator.route", "retrieval.execute_tools",
                  "retrieval.retrieve", "retrieval.relevance_check", "retrieval.synthesize",
                  "retrieval.score_agreement", "evidence.evidence", "critique.critique", "orchestrator.finalize"):
        assert label in state["timings"], label


def test_models_used_records_who_answered(make_deps):
    state = run(make_deps(), "type 2 diabetes phase 3")
    assert state["models_used"]["planner"] == "nemo"
    assert state["models_used"]["synthesizer:deep"] == "deep" and state["models_used"]["critic"] == "deep"


def test_lookup_by_nct_id_skips_vector_store_and_uses_cheaper_critique(make_deps, ctgov):
    store = make_deps().vector_store
    deps = make_deps(store=store)
    state = run(deps, "Tell me about NCT00000001")
    assert state["intent"] == "detail" and state["critique_mode"] == "code_only"
    assert store.docs == {} and ctgov.get_calls == ["NCT00000001"]
    assert state["context_nct_ids"] == ["NCT00000001"]
    assert "critic" not in state["models_used"]          # code-only critique makes no LLM call


def test_compare_two_trials(make_deps, ctgov):
    state = run(make_deps(), "Compare NCT00000001 and NCT00000003")
    assert state["intent"] == "compare" and set(state["context_nct_ids"]) == {"NCT00000001", "NCT00000003"}
    assert state["critique_mode"] == "full"


def test_weak_retrieval_triggers_reformulation_then_succeeds(make_deps, brain):
    brain.router_reply = '{"tool_calls": [{"name": "search_trials", "args": {"condition": "zzzz unknownword"}}]}'
    brain_reply = brain.respond
    deps = make_deps(per_model={"nemo": lambda t: brain.router_reply if "route questions" in t else brain_reply(t),
                                "deep": brain_reply})
    state = run(deps, "diabetes metformin question")
    assert brain.reformulated and state["reformulation_count"] == 1
    assert state["context_nct_ids"] and state["verified_citations"]


def test_reformulation_is_capped(make_deps, brain):
    deps = make_deps(make_config(max_reformulations=2), per_model={
        "nemo": lambda t: '{"tool_calls": [{"name": "search_trials", "args": {"condition": "zzzz"}}]}'
        if "route questions" in t else '{"condition": "qqqq"}' if "returned no relevant" in t else brain.respond(t),
        "deep": brain.respond})
    state = run(deps, "nothing matches")
    assert state["reformulation_count"] == 2
    assert state["aggregated_answer"]                       # still answers (says data is insufficient)


def test_revision_loop_fixes_bad_citation(make_deps, brain):
    brain.bad_citation_until_feedback = True
    state = run(make_deps(), "type 2 diabetes phase 3")
    assert state["revision_count"] == 1
    assert state["unverified_citations"] == [] and "NCT99999999" not in state["aggregated_answer"]
    assert state["critique"]["verdict"] == "pass"


def test_unresolvable_bad_citation_is_tagged_and_warned(make_deps, brain):
    brain.always_bad_citation = True
    state = run(make_deps(make_config(max_revisions=1)), "type 2 diabetes phase 3")
    assert state["revision_count"] == 1                     # exactly one revision, no infinite loop
    assert "[NCT99999999 unverified]" in state["aggregated_answer"]
    assert state["unverified_citations"] == ["NCT99999999"]
    assert any("unresolved" in w for w in state["warnings"])


def test_max_revisions_zero_never_revises(make_deps, brain):
    brain.always_bad_citation = True
    state = run(make_deps(make_config(max_revisions=0)), "type 2 diabetes phase 3")
    assert state["revision_count"] == 0 and state["unverified_citations"]


def test_llm_critic_can_force_a_revision(make_deps, brain):
    brain.critic_issue = "Mention the phase explicitly."
    state = run(make_deps(), "type 2 diabetes phase 3")
    assert state["revision_count"] == 1 and state["critique"]["verdict"] == "pass"


def test_two_model_config_works(make_deps):
    config = make_config()
    config.roles.synthesizers = ["nemo"]
    state = run(make_deps(config), "type 2 diabetes phase 3")
    assert state["agreement_models"] == 1 and state["confidence_score"] is None
    assert state["aggregated_answer"]


def test_all_models_down_degrades_gracefully(make_deps):
    boom = lambda t: Exception("connection refused")
    state = run(make_deps(make_config(llm_retries=0), per_model={"nemo": boom, "deep": boom}), "type 2 diabetes")
    assert "No answer could be generated" in state["aggregated_answer"]
    assert state["errors"] and state["warnings"]


def test_api_failure_is_recorded_not_raised(make_deps, ctgov):
    def broken(params, max_results=None):
        raise RuntimeError("ClinicalTrials.gov unreachable")
    ctgov.fetch_trials = broken
    state = run(make_deps(make_config(max_reformulations=0)), "type 2 diabetes")
    assert any("unreachable" in e for e in state["errors"]) and state["aggregated_answer"]


def test_plan_only_mode_stops_after_planning(make_deps, ctgov):
    state = run(make_deps(), "Summarize NCT00000001", mode="plan_only")
    assert state["tool_calls_llm"][0]["name"] == "get_trial" and ctgov.get_calls == [] and "responses" not in state or not state["responses"]


def test_retrieval_only_mode_stops_before_synthesis(make_deps):
    deps = make_deps()
    state = run(deps, "type 2 diabetes phase 3", mode="retrieval_only")
    assert state["context_nct_ids"] and not state.get("responses")
    assert not any("clinical research assistant" in c for c in deps.llms["nemo"].calls)


def test_unknown_mode_rejected(make_deps):
    with pytest.raises(ValueError):
        build_graph(make_deps(), "nope")


def test_pipeline_facade(make_deps):
    pipeline = ClinicalPipeline(config=make_deps().config, deps=make_deps())
    result = pipeline.ask("type 2 diabetes phase 3")
    assert result["verified_citations"] == ["NCT00000001"] and result["context_chunks"]
    assert pipeline.plan("Summarize NCT00000001")["tool_calls_llm"][0]["name"] == "get_trial"
    assert pipeline.retrieve("type 2 diabetes phase 3")["fetched_nct_ids"]
    assert pipeline.call_tool("get_trial", {"nct_id": "NCT00000001"})[0]["nct_id"] == "NCT00000001"
    with pytest.raises(ValueError):
        pipeline.call_tool("get_trial", {"nct_id": "bad"})
    assert pipeline.search({"condition": "diabetes"})["n_fetched"] == 2
    assert pipeline.judge("q", "a [NCT00000001]", ["ctx"])["faithfulness"] == 0.9
    assert pipeline.describe()["planner"]["name"] == "nemo"


def test_think_blocks_are_stripped_for_reasoning_models(make_deps, brain):
    deps = make_deps(per_model={"nemo": brain.respond, "deep": lambda t: "<think>secret</think>" + brain.respond(t)})
    state = run(deps, "type 2 diabetes phase 3")
    assert "secret" not in state["responses"]["deep"] and "<think>" not in state["aggregated_answer"]


def test_ask_stream_yields_each_node_then_the_same_result_as_ask(make_deps):
    pipeline = ClinicalPipeline(config=make_deps().config, deps=make_deps())
    events = list(pipeline.ask_stream("type 2 diabetes phase 3"))
    nodes_seen = [e["node"] for e in events if e["event"] == "node"]
    assert nodes_seen[:3] == ["start", "plan_tools", "route"] and nodes_seen[-1] == "finalize"
    assert "synthesize" in nodes_seen and "evidence" in nodes_seen and "critique" in nodes_seen
    assert events[-1]["event"] == "result" and sum(e["event"] == "result" for e in events) == 1
    plain = pipeline.ask("type 2 diabetes phase 3")
    streamed = events[-1]["result"]
    for key in ("answer", "citations", "verified_citations", "intent", "tool_runs", "context_nct_ids", "plan_source"):
        assert streamed[key] == plain[key], key
