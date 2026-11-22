import json

from src.agent import critique, evidence, nodes, orchestrator
from src.agent.evidence import quote_supported
from tests.conftest import TRIALS, make_config

CTX = ["Trial ID: NCT00000001\nSummary: A study of metformin efficacy in adults with type 2 diabetes."]


def test_quote_supported_tolerates_light_paraphrase_but_not_invention():
    text = CTX[0]
    assert quote_supported("study of metformin efficacy in adults", text)
    assert quote_supported("A study of the metformin efficacy in adults with type 2 diabetes", text)
    assert not quote_supported("randomized trial of insulin pumps in children", text)
    assert not quote_supported("metformin", text)           # too short to verify


def test_evidence_verifies_citations_by_code(make_deps, brain):
    deps = make_deps()
    state = {"draft_answer": "Metformin [NCT00000001] and a ghost trial [NCT99999999].",
             "relevant_chunks": CTX, "context_nct_ids": ["NCT00000001"]}
    out = evidence.evidence_node(state, deps)
    assert out["verified_citations"] == ["NCT00000001"]
    assert out["unverified_citations"] == ["NCT99999999"]
    assert out["evidence_support_rate"] == 1.0 and out["evidence"][0]["supported"]


def test_evidence_flags_quote_not_in_trial(make_deps):
    reply = json.dumps({"claims": [{"claim": "c", "nct_id": "NCT00000001", "quote": "gene therapy for sickle cell disease"}]})
    deps = make_deps(per_model={"nemo": lambda t: reply})
    out = evidence.evidence_node({"draft_answer": "x [NCT00000001]", "relevant_chunks": CTX,
                                  "context_nct_ids": ["NCT00000001"]}, deps)
    assert out["evidence"][0]["supported"] is False and out["evidence_support_rate"] == 0.0


def test_evidence_survives_model_failure_and_still_verifies_citations(make_deps):
    deps = make_deps(per_model={"nemo": lambda t: Exception("model not found"),
                                "deep": lambda t: Exception("model not found")})
    out = evidence.evidence_node({"draft_answer": "see [NCT00000001]", "relevant_chunks": CTX,
                                  "context_nct_ids": ["NCT00000001"]}, deps)
    assert out["evidence"] == [] and out["evidence_support_rate"] is None
    assert out["verified_citations"] == ["NCT00000001"]
    assert any("evidence extraction failed" in w for w in out["warnings"])


def test_critique_passes_clean_answer(make_deps):
    deps = make_deps(make_config(critique_mode="code_only"))
    out = critique.critique_node({"draft_answer": "ok", "critique_mode": "code_only",
                                  "unverified_citations": [], "evidence_support_rate": 1.0}, deps)
    assert out["critique"]["verdict"] == "pass"


def test_critique_requests_revision_for_unverified_citation_and_counts_it(make_deps):
    deps = make_deps(make_config(critique_mode="code_only", max_revisions=1))
    out = critique.critique_node({"draft_answer": "x", "critique_mode": "code_only",
                                  "unverified_citations": ["NCT99999999"], "revision_count": 0}, deps)
    assert out["critique"]["verdict"] == "revise" and out["revision_count"] == 1
    assert "NCT99999999" in out["critique"]["issues"][0]


def test_critique_stops_after_max_revisions(make_deps):
    deps = make_deps(make_config(critique_mode="code_only", max_revisions=1))
    out = critique.critique_node({"draft_answer": "x", "critique_mode": "code_only",
                                  "unverified_citations": ["NCT99999999"], "revision_count": 1}, deps)
    assert out["critique"]["unresolved"] is True
    assert critique.route_after_critique({"critique": out["critique"]}, deps) == "finalize"
    assert any("unresolved" in w for w in out["warnings"])


def test_critique_low_support_rate_triggers_revision(make_deps):
    deps = make_deps(make_config(critique_mode="code_only"))
    state = {"draft_answer": "x", "critique_mode": "code_only", "evidence_support_rate": 0.4,
             "evidence": [{"claim": "bad claim", "supported": False}]}
    assert "bad claim" in critique.critique_node(state, deps)["critique"]["issues"][0]


def test_critique_off_never_revises(make_deps):
    deps = make_deps(make_config(critique_mode="off"))
    out = critique.critique_node({"draft_answer": "x", "critique_mode": "off",
                                  "unverified_citations": ["NCT99999999"]}, deps)
    assert out["critique"]["verdict"] == "pass"


def test_critique_full_uses_llm_issues_and_ignores_empty_revise(make_deps, brain):
    brain.critic_issue = "Phase is wrong."
    deps = make_deps(make_config(critique_mode="full"))
    out = critique.critique_node({"draft_answer": "x", "critique_mode": "full", "question": "q",
                                  "relevant_chunks": CTX}, deps)
    assert out["critique"]["verdict"] == "revise" and out["critique"]["source"] == "code+llm"

    deps = make_deps(make_config(critique_mode="full"), per_model={
        "deep": lambda t: json.dumps({"verdict": "revise", "issues": []})})
    out = critique.critique_node({"draft_answer": "x", "critique_mode": "full", "question": "q",
                                  "relevant_chunks": CTX}, deps)
    assert out["critique"]["verdict"] == "pass"


def test_critique_full_survives_critic_failure(make_deps):
    deps = make_deps(make_config(critique_mode="full"), per_model={
        "nemo": lambda t: Exception("model not found"), "deep": lambda t: Exception("model not found")})
    out = critique.critique_node({"draft_answer": "x", "critique_mode": "full", "question": "q",
                                  "relevant_chunks": CTX}, deps)
    assert out["critique"]["verdict"] == "pass"
    assert any("critique model failed" in w for w in out["warnings"])


def test_route_node_sets_intent_and_cheaper_critique_for_lookups(make_deps):
    deps = make_deps(make_config(critique_mode="full"))
    assert orchestrator.route_node({"tool_plan": [{"name": "search_trials"}]}, deps) == {
        "intent": "search", "critique_mode": "full"}
    assert orchestrator.route_node({"tool_plan": [{"name": "get_trial"}]}, deps) == {
        "intent": "detail", "critique_mode": "code_only"}
    assert orchestrator.route_node({"tool_plan": [{"name": "compare_trials"}]}, deps)["critique_mode"] == "full"


def test_finalize_tags_unverified_citations_and_recomputes_from_final_text(make_deps):
    deps = make_deps()
    out = orchestrator.finalize_node({
        "aggregated_answer": "Real [NCT00000001], fake [NCT99999999] and NCT88888888.",
        "context_nct_ids": ["NCT00000001"], "started_at": 0.0}, deps)
    assert out["verified_citations"] == ["NCT00000001"]
    assert out["unverified_citations"] == ["NCT99999999", "NCT88888888"]
    assert "[NCT99999999 unverified]" in out["aggregated_answer"]
    assert "[NCT88888888 unverified]" in out["aggregated_answer"]
    assert out["warnings"]


def test_finalize_with_no_answer_says_so(make_deps):
    out = orchestrator.finalize_node({"aggregated_answer": ""}, make_deps())
    assert "No answer could be generated" in out["aggregated_answer"] and out["warnings"]


def test_plan_tools_nct_override_when_planner_ignores_ids(make_deps, brain):
    brain.router_reply = json.dumps({"tool_calls": [{"name": "search_trials", "args": {"term": "x"}}]})
    deps = make_deps()
    out = nodes.plan_tools_node({"question": "Tell me about NCT00000001"}, deps)
    assert out["tool_calls_llm"][0]["name"] == "search_trials"      # the model's own choice is kept for scoring
    assert out["tool_plan"][0]["name"] == "get_trial" and out["plan_source"] == "llm+nct_fix"


def test_plan_tools_override_can_be_disabled(make_deps, brain):
    brain.router_reply = json.dumps({"tool_calls": [{"name": "search_trials", "args": {"term": "x"}}]})
    deps = make_deps(make_config(nct_routing_override=False))
    out = nodes.plan_tools_node({"question": "Tell me about NCT00000001"}, deps)
    assert out["tool_plan"][0]["name"] == "search_trials" and out["plan_source"] == "llm"


def test_plan_tools_defaults_to_search_when_output_is_garbage(make_deps, brain):
    brain.router_reply = "I would just look it up"
    out = nodes.plan_tools_node({"question": "diabetes trials"}, make_deps())
    assert out["tool_calls_llm"] == [] and out["plan_json_ok"] is False
    assert out["tool_plan"] == [{"name": "search_trials", "args": {"term": "diabetes trials"}}]
    assert out["plan_source"] == "default"


def test_plan_tools_falls_back_to_second_model_when_planner_is_down(make_deps, brain):
    deps = make_deps(per_model={"nemo": lambda t: Exception("model not found")})
    deps.llms["deep"].fn = brain.respond
    out = nodes.plan_tools_node({"question": "type 2 diabetes phase 3"}, deps)
    assert out["models_used"]["planner"] == "deep" and out["tool_calls_llm"]


def test_native_tool_calling_path_and_fallback_to_json(make_deps, brain):
    config = make_config()
    config.models["nemo"].tool_calling = "native"
    deps = make_deps(config)
    deps.llms["nemo"].tool_calls = [{"name": "get_trial", "args": {"nct_id": "NCT00000001"}, "id": "1"}]
    out = nodes.plan_tools_node({"question": "Tell me about NCT00000001"}, deps)
    assert out["tool_calls_llm"] == [{"name": "get_trial", "args": {"nct_id": "NCT00000001"}}]
    assert out["plan_source"] == "llm"

    deps.llms["nemo"].tool_calls = []                              # model ignores tools -> JSON routing
    out = nodes.plan_tools_node({"question": "type 2 diabetes"}, deps)
    assert out["tool_calls_llm"][0]["name"] == "search_trials"


def test_synthesizers_run_in_parallel_and_survive_one_failure(make_deps):
    deps = make_deps(per_model={"deep": lambda t: Exception("model not found")})
    out = nodes.synthesize_node({"question": "q", "relevant_chunks": CTX}, deps)
    # 'deep' fails, falls back to 'nemo', so both slots are still filled
    assert set(out["responses"]) == {"nemo", "deep"}
    assert out["models_used"]["synthesizer:deep"] == "nemo"


def test_synthesis_with_everything_down_reports_errors(make_deps):
    boom = lambda t: Exception("model not found")
    deps = make_deps(per_model={"nemo": boom, "deep": boom})
    out = nodes.synthesize_node({"question": "q", "relevant_chunks": CTX}, deps)
    assert out["responses"] == {} and len(out["errors"]) == 2


def test_synthesis_context_is_capped(make_deps):
    deps = make_deps(make_config(max_context_chars=1000))
    nodes.synthesize_node({"question": "q", "relevant_chunks": ["x" * 5000]}, deps)
    assert "x" * 1001 not in deps.llms["nemo"].calls[0]


def test_reformulate_recovers_from_invalid_phase_and_garbage(make_deps):
    reply = json.dumps({"condition": "diabetes", "phase": "PHASE9"})
    deps = make_deps(per_model={"nemo": lambda t: reply})
    out = nodes.reformulate_node({"question": "q", "query_params": {}, "reformulation_count": 0}, deps)
    assert out["tool_plan"][0]["args"]["condition"] == "diabetes" and out["reformulation_count"] == 1

    deps = make_deps(per_model={"nemo": lambda t: "garbage"})
    out = nodes.reformulate_node({"question": "my question", "query_params": {}, "reformulation_count": 1}, deps)
    assert out["tool_plan"][0]["args"]["term"] == "my question" and out["reformulation_count"] == 2


def test_agreement_with_one_model_is_none(make_deps):
    out = nodes.score_agreement_node({"responses": {"nemo": "only"}}, make_deps())
    assert out["confidence_score"] is None and out["agreement_models"] == 1 and out["draft_answer"] == "only"


def test_low_agreement_shows_all_perspectives(make_deps):
    out = nodes.score_agreement_node(
        {"responses": {"nemo": "metformin diabetes glucose", "deep": "melanoma pembrolizumab skin"}}, make_deps())
    assert "low agreement" in out["aggregated_answer"] and out["draft_answer"].startswith("metformin")


def _garbage_then_valid(text):
    """Router reply: garbage on the first call, a valid plan once the repair note is present."""
    if "previous reply was not usable" in text:
        return json.dumps({"tool_calls": [{"name": "search_trials", "args": {"condition": "melanoma"}}]})
    return "let me think about that"


def test_plan_repair_recovers_a_bad_first_reply(make_deps, brain):
    brain.router_reply = _garbage_then_valid
    out = nodes.plan_tools_node({"question": "melanoma trials"}, make_deps())
    assert out["tool_calls_llm"][0]["args"] == {"condition": "melanoma", "max_results": 25}
    assert out["plan_source"] == "llm" and out["plan_attempts"] == 2
    assert out["errors"] == []


def test_plan_repair_note_names_the_problem(make_deps, brain):
    seen = []

    def reply(text):
        seen.append(text)
        return json.dumps({"tool_calls": []})              # valid JSON, no calls

    brain.router_reply = reply
    nodes.plan_tools_node({"question": "melanoma trials"}, make_deps())
    assert "it had no valid tool call" in seen[1]
    assert "Valid tools: search_trials, get_trial, compare_trials, get_eligibility" in seen[1]


def test_plan_repair_can_be_disabled_to_measure_the_bare_model(make_deps, brain):
    brain.router_reply = _garbage_then_valid
    out = nodes.plan_tools_node({"question": "melanoma trials"}, make_deps(make_config(plan_repair_retries=0)))
    assert out["tool_calls_llm"] == [] and out["plan_attempts"] == 1 and out["plan_source"] == "default"


def test_plan_repair_gives_up_after_the_configured_retries(make_deps, brain):
    brain.router_reply = "no json here"
    out = nodes.plan_tools_node({"question": "melanoma trials"}, make_deps(make_config(plan_repair_retries=2)))
    assert out["plan_attempts"] == 3 and out["plan_source"] == "default"
    assert out["plan_raw"] == "no json here" and out["plan_json_ok"] is False
    assert len(out["errors"]) == 3


def test_default_plan_uses_phase_and_status_from_the_question(make_deps, brain):
    brain.router_reply = "no json here"
    out = nodes.plan_tools_node({"question": "Which phase 3 trials are recruiting for lupus?"},
                                make_deps(make_config(plan_repair_retries=0)))
    assert out["tool_plan"][0]["args"] == {"term": "Which phase 3 trials are recruiting for lupus?",
                                           "phase": "PHASE3", "status": "RECRUITING"}


def test_synthesis_prompt_tells_models_to_flag_population_mismatches():
    from src.agent.prompts import SYNTHESIS_PROMPT
    text = SYNTHESIS_PROMPT.messages[0].prompt.template
    assert "different population" in text and "prediabetes" in text
