import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.pipeline import ClinicalPipeline  # noqa: E402
from tests import ui_fakes  # noqa: E402


def app_test(script, args=()):
    return AppTest.from_function(script, args=args, default_timeout=30)


def _app():
    from unittest import mock

    from tests.ui_fakes import FakePipeline
    from ui.app import main
    with mock.patch("ui.ui_lib.ollama_status", return_value=True):
        main(FakePipeline())


@pytest.fixture
def result(make_deps):
    pipeline = ClinicalPipeline(config=make_deps().config, deps=make_deps())
    out = pipeline.ask("type 2 diabetes phase 3")
    ui_fakes.RESULT = out
    return out


def ask(at, question):
    at.text_area[0].set_value(question)
    next(b for b in at.button if b.label == "Ask").click()
    return at.run()


def test_home_screen_has_form_and_examples():
    at = app_test(_app).run()
    assert not at.exception
    assert at.text_area[0].label == "Your question"
    assert at.title[0].value == "ClinTrialNavigator"
    assert "Multi-LLM Multi-Agentic RAG System for Clinical Trial Q&A" in " ".join(m.value for m in at.markdown)
    assert "Cited, evidence-grounded answers over ClinicalTrials.gov" in " ".join(m.value for m in at.markdown)
    assert len([b for b in at.button if b.label == "Ask"]) == 1
    assert len(at.button) == 7                              # Ask plus six examples
    assert not at.expander and not at.warning


def test_example_button_fills_the_question():
    at = app_test(_app).run()
    next(b for b in at.button if b.label.startswith("Compare NCT")).click()
    at.run()
    assert at.text_area[0].value.startswith("Compare NCT")


def test_asking_shows_answer_citations_and_sidebar(result):
    at = ask(app_test(_app).run(), "type 2 diabetes phase 3")
    assert not at.exception and not at.error
    text = " ".join(m.value for m in at.markdown)
    assert "https://clinicaltrials.gov/study/NCT00000001" in text
    assert ":green[✓ NCT00000001 verified]" in text
    assert "How this answer was made" in text and "Models used" in text
    assert len(at.tabs) == 3 and at.tabs[2].label.startswith("Evidence")


def test_unresolved_review_issues_show_as_a_warning(result):
    result["critique"] = {"verdict": "revise", "issues": ["NCT00000001 is the wrong population"], "unresolved": True}
    at = ask(app_test(_app).run(), "type 2 diabetes phase 3")
    assert len(at.warning) == 1 and "wrong population" in at.warning[0].value


def test_unverified_citation_is_flagged(result):
    result["answer"] = "A ghost trial [NCT99999999 unverified]."
    result["unverified_citations"] = ["NCT99999999"]
    result["verified_citations"] = []
    at = ask(app_test(_app).run(), "type 2 diabetes phase 3")
    text = " ".join(m.value for m in at.markdown)
    assert ":orange[⚠ NCT99999999 unverified]" in text


def test_blank_question_does_nothing():
    at = ask(app_test(_app).run(), "   ")
    assert not at.exception and not at.warning and not at.status


def test_startup_failure_shows_a_helpful_error():
    def broken_app():
        from unittest import mock

        from ui.app import main
        with mock.patch("ui.app.load_pipeline", side_effect=RuntimeError("config broken")):
            main()

    at = app_test(broken_app).run()
    assert not at.exception and len(at.error) == 1 and "config broken" in at.error[0].value
    assert not at.button                                  # no form when the pipeline cannot start


def _ollama_app(up):
    from unittest import mock

    import streamlit as st

    from tests.ui_fakes import FakePipelineWithOllama
    from ui.app import main
    st.cache_data.clear()
    with mock.patch("ui.ui_lib.ollama_status", return_value=up):
        main(FakePipelineWithOllama())


def test_ollama_down_shows_an_error():
    at = app_test(_ollama_app, args=(False,)).run()
    assert not at.exception and len(at.error) == 1
    assert "not reachable at http://ollama.test:11434" in at.error[0].value


def test_ollama_up_shows_nothing_extra():
    at = app_test(_ollama_app, args=(True,)).run()
    assert not at.exception and not at.error
