from src.pipeline import ClinicalPipeline
from tests.conftest import make_config
from ui import ui_lib as lib


def real_result(make_deps, question="type 2 diabetes phase 3"):
    return ClinicalPipeline(config=make_deps().config, deps=make_deps()).ask(question)


def test_md_escape_neutralizes_markdown_and_latex():
    assert lib.md_escape("a *b* [c](d) $x$ <i>") == "a \\*b\\* \\[c\\](d) \\$x\\$ \\<i\\>"
    assert lib.md_escape(None) == ""


def test_format_call_hides_max_results():
    call = {"name": "search_trials", "args": {"condition": "asthma", "phase": "PHASE2", "max_results": 25}}
    assert lib.format_call(call) == "search_trials(condition='asthma', phase='PHASE2')"


def test_linkify_links_verified_ids_and_tags_unverified_ones():
    text = "See [NCT00000001] and NCT00000002, but not [NCT99999999 unverified]."
    out = lib.linkify(text)
    assert "[NCT00000001](https://clinicaltrials.gov/study/NCT00000001)" in out
    assert "[NCT00000002](https://clinicaltrials.gov/study/NCT00000002)" in out
    assert ":orange[⚠ NCT99999999 unverified]" in out and "https://clinicaltrials.gov/study/NCT99999999" not in out
    assert ":orange[⚠ NCT00000002 unverified]" in lib.linkify(text, unverified=["NCT00000002"])


def test_linkify_adds_no_html():
    assert "<" not in lib.linkify("[NCT00000001] is fine")


def test_progress_lines():
    plan = {"tool_plan": [{"name": "get_trial", "args": {"nct_id": "NCT00000001"}}]}
    assert lib.progress_line("plan_tools", plan) == "Planned: get\\_trial(nct\\_id='NCT00000001')"
    assert lib.progress_line("execute_tools", {"tool_runs": [{"n_trials": 3}, {"n_trials": 2}]}).startswith("Fetched 5 trials")
    assert lib.progress_line("retrieve", {"context_nct_ids": ["a", "b"]}) == "Kept 2 trials in context"
    assert "revision" in lib.progress_line("critique", {"critique": {"verdict": "revise"}})
    assert lib.progress_line("critique", {"critique": {"verdict": "pass"}}) == "Review finished"
    assert lib.progress_line("start", {}) == "" and lib.progress_line("finalize", {}) == ""


def test_trial_cards_parse_the_context_text(make_deps):
    cards = lib.trial_cards(real_result(make_deps))
    assert cards and cards[0]["nct_id"] == "NCT00000001"
    assert cards[0]["title"] == "Metformin for Type 2 Diabetes" and cards[0]["status"] == "RECRUITING"
    assert cards[0]["url"].endswith("/NCT00000001")
    assert len({c["nct_id"] for c in cards}) == len(cards)


def test_stage_summary_covers_five_stages_and_shares(make_deps):
    rows = lib.stage_summary(real_result(make_deps))
    assert [r["name"] for r in rows] == ["Plan", "Fetch and rank", "Synthesize", "Verify", "Review"]
    assert all(0.02 <= r["share"] <= 1.0 for r in rows)
    assert "search_trials" in rows[0]["detail"] and "kept in context" in rows[1]["detail"]
    assert "Agreement" in rows[2]["detail"] and "citations found" in rows[3]["detail"]


def test_stage_summary_reports_default_plan_and_open_issues():
    result = {"plan_source": "default", "tool_runs": [{"name": "search_trials", "args": {"term": "x"}, "n_trials": 0}],
              "critique": {"unresolved": True, "issues": ["a", "b"]}, "revision_count": 1,
              "models_used": {"critic": "llama"}, "timings": {"critique.critique": 5.0}}
    rows = lib.stage_summary(result)
    assert rows[0]["detail"].startswith("The planner reply was not usable")
    assert "2 issue(s) still open" in rows[4]["detail"] and rows[4]["seconds"] == 5.0
    assert rows[4]["share"] == 1.0


def test_stage_summary_code_only_review_and_single_model():
    result = {"confidence": None, "models_used": {"synthesizer:a": "tag-a"}, "revision_count": 0}
    rows = lib.stage_summary(result)
    assert rows[2]["detail"].startswith("tag-a answered") and rows[4]["detail"].startswith("Code checks only")


def test_review_issues_only_when_unresolved():
    assert lib.review_issues({"critique": {"issues": ["x"], "unresolved": True}}) == ["x"]
    assert lib.review_issues({"critique": {"issues": ["x"]}}) == []
    assert lib.review_issues({}) == []


def test_evidence_rows():
    rows = lib.evidence_rows({"evidence": [{"claim": "c", "nct_id": "NCT00000001", "quote": "q", "supported": False}]})
    assert rows[0]["Result"] == "✗ not found" and rows[0]["Trial"] == "NCT00000001"


def test_roles_table_merges_roles_per_model_tag():
    info = make_config().describe()
    table = dict(lib.roles_table(info))
    assert "planner" in table["nemo-tag"] and "synthesizer" in table["nemo-tag"]
    assert "synthesizer" in table["deep-tag"] and "reviewer" in table["deep-tag"]


def test_ollama_urls_only_for_ollama_models():
    info = make_config().describe()
    for entry in (info["planner"], *info["synthesizers"], info["evidence"], info["critic"]):
        entry["base_url"] = "http://host:11434"
    assert lib.ollama_urls(info) == ["http://host:11434"]
    for entry in (info["planner"], *info["synthesizers"], info["evidence"], info["critic"]):
        entry["provider"] = "groq"
    assert lib.ollama_urls(info) == []


def test_ollama_status_false_when_nothing_listens():
    assert lib.ollama_status("http://127.0.0.1:9", timeout=0.3) is False
