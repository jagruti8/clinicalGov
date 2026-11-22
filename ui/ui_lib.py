"""Pure helpers for the Streamlit UI. No streamlit import here, so they are unit tested
without it."""
import json
import re
import urllib.request

TRIAL_URL = "https://clinicaltrials.gov/study/{nct_id}"
NCT_TAG = re.compile(r"\[?(NCT\d{8})( unverified)?\]?")

EXAMPLES = [
    "What phase 3 trials are recruiting for type 2 diabetes?",
    "Which trials study pembrolizumab for melanoma?",
    "Summarize trial NCT06694935",
    "What are the inclusion criteria for NCT03925714?",
    "Compare NCT06694935 and NCT03925714",
    "Completed phase 2 trials for asthma",
]

# Stage name -> the timing labels (see src/agent/graph.py) that belong to it.
STAGES = [
    ("Plan", ("retrieval.plan_tools", "orchestrator.route")),
    ("Fetch and rank", ("retrieval.execute_tools", "retrieval.retrieve",
                        "retrieval.relevance_check", "retrieval.reformulate")),
    ("Synthesize", ("retrieval.synthesize", "retrieval.score_agreement")),
    ("Verify", ("evidence.evidence",)),
    ("Review", ("critique.critique",)),
]


def md_escape(text: str) -> str:
    """Escape markdown and LaTeX characters so untrusted text renders as plain text."""
    return re.sub(r"([\\`*_\[\]<>|~$#])", r"\\\1", str(text or ""))


def format_call(call: dict) -> str:
    args = {k: v for k, v in (call.get("args") or {}).items() if k != "max_results"}
    return f"{call.get('name', '?')}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


def progress_line(node: str, update: dict) -> str:
    """One line for the live progress box when a graph node finishes ('' = say nothing)."""
    update = update or {}
    if node == "plan_tools":
        plan = update.get("tool_plan") or []
        return "Planned: " + "; ".join(md_escape(format_call(c)) for c in plan) if plan else ""
    if node == "execute_tools":
        runs = update.get("tool_runs") or []
        return f"Fetched {sum(r.get('n_trials', 0) for r in runs)} trials from ClinicalTrials.gov"
    if node == "retrieve":
        return f"Kept {len(update.get('context_nct_ids') or [])} trials in context"
    if node == "reformulate":
        return "No relevant trials found, reformulating the search"
    if node == "synthesize":
        names = ", ".join(md_escape(n) for n in (update.get("responses") or {}))
        return f"Answered by {names}" if names else "Models answered"
    if node == "evidence":
        return "Checked citations and claims against the trial text"
    if node == "critique":
        critique = update.get("critique") or {}
        if critique.get("verdict") == "revise" and not critique.get("unresolved"):
            return "Reviewer asked for a revision"
        return "Review finished"
    return ""


def linkify(answer: str, unverified=()) -> str:
    """Turn NCT IDs into links to the trial record. Unverified ones become an orange tag.
    The answer is left as markdown without raw HTML, so model output cannot inject markup."""
    unverified = set(unverified)

    def replace(match):
        nct_id = match.group(1)
        if match.group(2) or nct_id in unverified:
            return f":orange[⚠ {nct_id} unverified]"
        return f"[{nct_id}]({TRIAL_URL.format(nct_id=nct_id)})"

    return NCT_TAG.sub(replace, answer or "")


def trial_cards(result: dict) -> list[dict]:
    """Trials the models saw, parsed from the context text (see format_trial_text)."""
    cards, seen = [], set()
    for chunk in result.get("context_chunks") or []:
        fields = dict(re.findall(r"^(Trial ID|Title|Status|Phase): (.*)$", chunk, flags=re.MULTILINE))
        nct_id = fields.get("Trial ID", "")
        if not nct_id or nct_id in seen:
            continue
        seen.add(nct_id)
        cards.append({"nct_id": nct_id, "title": fields.get("Title", ""),
                      "status": fields.get("Status", ""), "phase": fields.get("Phase", ""),
                      "url": TRIAL_URL.format(nct_id=nct_id)})
    return cards


def evidence_rows(result: dict) -> list[dict]:
    return [{"Claim": e.get("claim", ""), "Trial": e.get("nct_id", ""),
             "Quote found in trial text": e.get("quote", ""),
             "Result": "✓ supported" if e.get("supported") else "✗ not found"}
            for e in result.get("evidence") or []]


def review_issues(result: dict) -> list[str]:
    critique = result.get("critique") or {}
    return list(critique.get("issues") or []) if critique.get("unresolved") else []


def _seconds(timings: dict, labels) -> float:
    return sum(timings.get(label, 0.0) for label in labels)


def stage_summary(result: dict) -> list[dict]:
    """The five stages with seconds, share of total time, and a one-line description."""
    timings = result.get("timings") or {}
    models = result.get("models_used") or {}
    runs = result.get("tool_runs") or []
    evidence = result.get("evidence") or []
    issues = review_issues(result)
    confidence = result.get("confidence")
    synthesizers = [v for k, v in models.items() if k.startswith("synthesizer:")]
    source = result.get("plan_source") or ""

    plan = "; ".join(format_call(r) for r in runs) or "no tool ran"
    if source.startswith("default"):
        plan = f"The planner reply was not usable, so a default search ran: {plan}"
    else:
        plan = f"{models.get('planner', 'planner')} ran {plan}"
        if (result.get("plan_attempts") or 0) > 1:
            plan += " (after a repair retry)"

    fetched = sum(r.get("n_trials", 0) for r in runs)
    fetch = f"{fetched} trials fetched, {len(result.get('context_nct_ids') or [])} kept in context"
    if result.get("reformulation_count"):
        fetch += f". Search reformulated {result['reformulation_count']} time(s)"

    if confidence is not None:
        synth = f"{', '.join(synthesizers)} answered in parallel. Agreement {confidence:.0%}"
    elif synthesizers:
        synth = f"{synthesizers[0]} answered. Agreement is not scored with one model"
    else:
        synth = "No model answered"

    cited = result.get("citations") or []
    verify = f"{len(result.get('verified_citations') or [])} of {len(cited)} citations found in the trial text"
    if evidence:
        verify += f". {sum(1 for e in evidence if e.get('supported'))} of {len(evidence)} claims have a matching quote"

    revisions = result.get("revision_count") or 0
    reviewer = models.get("critic")
    review = f"{reviewer} reviewed the answer" if reviewer else "Code checks only"
    if revisions:
        review += f". {revisions} revision(s) made"
    review += f". {len(issues)} issue(s) still open" if issues else ". No open issues"

    details = [plan, fetch, synth, verify, review]
    rows = [{"name": name, "seconds": _seconds(timings, labels), "detail": detail}
            for (name, labels), detail in zip(STAGES, details)]
    total = sum(r["seconds"] for r in rows) or 1.0
    for row in rows:
        row["seconds"] = round(row["seconds"], 1)
        row["share"] = max(0.02, min(1.0, row["seconds"] / total))
    return rows


def roles_table(info: dict) -> list[tuple[str, str]]:
    """(model tag, roles it plays) in order of first appearance, from AppConfig.describe()."""
    roles: dict[str, list[str]] = {}

    def add(entries, role):
        for entry in entries if isinstance(entries, list) else [entries]:
            names = roles.setdefault(entry["model"], [])
            if role not in names:
                names.append(role)

    add(info["planner"], "planner")
    add(info["synthesizers"], "synthesizer")
    add(info["evidence"], "evidence")
    add(info["critic"], "reviewer")
    return [(tag, ", ".join(names)) for tag, names in roles.items()]


def ollama_status(base_url: str, timeout: float = 1.5) -> bool:
    """True if an Ollama server answers at base_url."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=timeout) as response:
            json.load(response)
        return True
    except Exception:
        return False


def ollama_urls(info: dict) -> list[str]:
    """Distinct Ollama endpoints used by the configured roles (empty for cloud-only setups)."""
    entries = [info["planner"], *info["synthesizers"], info["evidence"], info["critic"]]
    return list(dict.fromkeys(e["base_url"] for e in entries
                              if e.get("provider") == "ollama" and e.get("base_url")))
