"""Streamlit UI for ClinTrialNavigator.

    pip install -r requirements-ui.txt
    streamlit run ui/app.py            # or: python main.py --ui

Models come from config/models.yaml, or $MODELS_CONFIG. The UI runs the pipeline in this
process; it does not need the REST API.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from ui import ui_lib as lib  # noqa: E402

CSS = """
<style>
.block-container { max-width: 1200px; padding-top: 2rem; }
h1 { letter-spacing: -0.02em; }
</style>
"""


@st.cache_resource(show_spinner="Loading models and the vector store...")
def load_pipeline():
    from src.pipeline import ClinicalPipeline
    return ClinicalPipeline()


@st.cache_data(ttl=30, show_spinner=False)
def ollama_up(url: str) -> bool:
    return lib.ollama_status(url)


def set_question(text: str) -> None:
    st.session_state["question"] = text


def render_header(info: dict) -> None:
    st.title("ClinTrialNavigator")
    with st.container(key="taglines"):
        st.html("<style>.st-key-taglines h4{padding:0.55rem 0}"
                ".st-key-taglines{gap:0}</style>")
        st.markdown("#### Multi-LLM Multi-Agentic RAG System for Clinical Trial Q&A")
        st.markdown("#### Cited, evidence-grounded answers over ClinicalTrials.gov")
    for url in lib.ollama_urls(info):               # silent when fine; the models are listed under each answer
        if not ollama_up(url):
            st.error(f"Ollama is not reachable at {url}. Start it with `ollama serve`, "
                     "or point the config at another endpoint.")


def render_form() -> tuple[str, bool]:
    st.session_state.setdefault("question", "")
    with st.container(border=True):
        question = st.text_area("Your question", key="question", height=90,
                                placeholder="What phase 3 trials are recruiting for type 2 diabetes?")
        asked = st.button("Ask", type="primary")
    return question, asked


def render_examples() -> None:
    """Example questions. Once there is an answer they fold away so the answer stays in view."""
    if st.session_state.get("result"):
        examples = st.expander("Try an example")
    else:
        st.caption("TRY AN EXAMPLE")
        examples = st.container()
    columns = examples.columns(3)
    for index, text in enumerate(lib.EXAMPLES):
        columns[index % 3].button(text, key=f"example{index}", on_click=set_question, args=(text,),
                                  width="stretch")


def run_question(pipeline, question: str) -> dict:
    started = time.perf_counter()
    result = None
    with st.status("Working on your question...", expanded=True) as status:
        for event in pipeline.ask_stream(question):
            if event["event"] == "node":
                line = lib.progress_line(event["node"], event["update"])
                if line:
                    st.write(f"{line}  ·  {time.perf_counter() - started:.0f} s")
            else:
                result = event["result"]
        seconds = (result or {}).get("total_latency_s") or time.perf_counter() - started
        status.update(label=f"Done in {seconds:.0f} s", state="complete", expanded=False)
    return result


def render_answer_tabs(result: dict) -> None:
    evidence = lib.evidence_rows(result)
    responses = result.get("responses") or {}
    tabs = st.tabs(["Answer", f"Per model ({len(responses)})", f"Evidence ({len(evidence)})"])

    with tabs[0]:
        st.markdown(lib.linkify(result.get("answer", ""), result.get("unverified_citations") or []))
        chips = [f":green[✓ {i} verified]" for i in result.get("verified_citations") or []]
        chips += [f":orange[⚠ {i} unverified]" for i in result.get("unverified_citations") or []]
        if chips:
            st.divider()
            st.markdown("**Citations**  " + "  ".join(chips))
        if result.get("warnings"):
            with st.expander("Warnings"):
                for warning in result["warnings"]:
                    st.write(warning)
        if result.get("errors"):
            with st.expander("Errors"):
                for error in result["errors"]:
                    st.write(error)

    with tabs[1]:
        if result.get("confidence") is not None:
            st.markdown(f"**Agreement across {result.get('agreement_models')} models:** {result['confidence']:.0%}")
            st.caption("Below 50%, the answer shows each model's response as a separate perspective.")
        for model, text in responses.items():
            with st.container(border=True):
                st.markdown(f"**{lib.md_escape(model)}**")
                st.markdown(text)
        if not responses:
            st.caption("No model responses were recorded.")

    with tabs[2]:
        if evidence:
            st.caption("Each claim is matched to a quote that must appear in the trial text.")
            st.dataframe(evidence, width="stretch", hide_index=True)
        else:
            st.caption("No claims were extracted for this answer.")


def render_trials(result: dict) -> None:
    cards = lib.trial_cards(result)
    if not cards:
        return
    st.caption(f"TRIALS IN CONTEXT ({len(cards)})")
    for card in cards:
        with st.container(border=True):
            meta = " · ".join(lib.md_escape(v) for v in (card["phase"], card["status"]) if v)
            st.markdown(f"**[{card['nct_id']}]({card['url']})**  ·  {meta}")
            st.write(lib.md_escape(card["title"]))


def render_sidebar(result: dict, info: dict) -> None:
    stages = lib.stage_summary(result)
    with st.container(border=True):
        total = result.get("total_latency_s")
        st.markdown("**How this answer was made**" + (f"  ·  {total:.1f} s total" if total else ""))
        for index, stage in enumerate(stages, start=1):
            left, right = st.columns([3, 1])
            left.markdown(f"**{index}. {stage['name']}**")
            right.caption(f"{stage['seconds']} s")
            st.progress(stage["share"])
            st.caption(lib.md_escape(stage["detail"]))
    with st.container(border=True):
        st.markdown("**Models used**")
        for tag, roles in lib.roles_table(info):
            left, right = st.columns([3, 2])
            left.write(tag)
            right.caption(roles)
    with st.expander("Raw JSON"):
        st.json(result)


def render_result(result: dict, info: dict) -> None:
    issues = lib.review_issues(result)
    if issues:
        st.warning(f"**Reviewer flagged {len(issues)} issue(s) the revision did not resolve**\n\n"
                   + "\n".join(f"- {lib.md_escape(issue)}" for issue in issues))
    main_column, side_column = st.columns([2, 1], gap="large")
    with main_column:
        render_answer_tabs(result)
        render_trials(result)
    with side_column:
        render_sidebar(result, info)


def main(pipeline=None) -> None:
    st.set_page_config(page_title="ClinTrialNavigator", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        pipeline = pipeline or load_pipeline()
        info = pipeline.describe()
    except Exception as exc:
        st.title("ClinTrialNavigator")
        st.error(f"Could not start the pipeline: {type(exc).__name__}: {str(exc)[:300]}")
        st.caption("Run `python main.py --smoke` to check the config, models, and packages.")
        st.stop()

    render_header(info)
    question, asked = render_form()
    progress = st.container()

    if asked and question.strip():
        with progress:
            try:
                st.session_state["result"] = run_question(pipeline, question.strip())
                st.session_state["answered"] = question.strip()
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"The pipeline failed: {type(exc).__name__}: {str(exc)[:300]}")

    render_examples()

    if st.session_state.get("result"):
        st.divider()
        st.markdown(f"#### {lib.md_escape(st.session_state.get('answered', ''))}")
        render_result(st.session_state["result"], info)
    st.caption("Research tool. Not medical advice.")


if __name__ == "__main__":
    main()
