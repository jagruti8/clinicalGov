"""Generate evaluation datasets from live ClinicalTrials.gov data.

  python -m evals.generate_queries --out-dir evals/datasets --seed 1

Writes (JSONL, one sample per line, Inspect's default fields: id, input, target, metadata):
  answer_queries.jsonl     ~1000 questions for answer_quality (dev conditions)
  tool_queries.jsonl       100 questions for tool_selection (HELD-OUT conditions, so no
                           overlap with the answer set); target = expected tool name
  retrieval_queries.jsonl  questions built from one seed trial; metadata.expected_nct
  seed_trials.json         the trials the datasets were built from (reproducibility)
  manifest.json            seed, counts, condition split

Reproducible: same --seed and same seed_trials.json give the same datasets. Pass
--seed-file to rebuild from a saved seed_trials.json without calling the API.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONDITIONS = [
    "type 2 diabetes", "type 1 diabetes", "breast cancer", "lung cancer", "prostate cancer",
    "colorectal cancer", "melanoma", "leukemia", "lymphoma", "multiple myeloma",
    "ovarian cancer", "pancreatic cancer", "hepatocellular carcinoma", "glioblastoma",
    "Alzheimer's disease", "Parkinson's disease", "multiple sclerosis", "epilepsy", "migraine",
    "depression", "schizophrenia", "bipolar disorder", "asthma", "COPD", "cystic fibrosis",
    "rheumatoid arthritis", "psoriasis", "Crohn's disease", "ulcerative colitis", "lupus",
    "hypertension", "heart failure", "atrial fibrillation", "stroke", "chronic kidney disease",
    "HIV", "hepatitis C", "COVID-19", "influenza", "tuberculosis", "obesity", "osteoporosis",
    "sickle cell disease", "hemophilia", "ALS", "Duchenne muscular dystrophy",
]

PHASE_WORDS = {"PHASE1": "phase 1", "PHASE2": "phase 2", "PHASE3": "phase 3", "PHASE4": "phase 4"}
STATUS_WORDS = {"RECRUITING": "recruiting", "COMPLETED": "completed", "NOT_YET_RECRUITING": "not yet recruiting"}

SEARCH_TEMPLATES = [
    "What {phase} trials are {status} for {condition}?",
    "Which trials study {intervention} for {condition}?",
    "Are there {status} trials of {intervention}?",
    "List trials investigating {condition}.",
    "What is being tested in {phase} studies of {condition}?",
]
DETAIL_TEMPLATES = ["Summarize trial {nct}.", "Tell me about {nct}.", "What does {nct} study and what is its status?"]
ELIGIBILITY_TEMPLATES = ["What are the inclusion and exclusion criteria for {nct}?", "Who can enroll in {nct}?"]
COMPARE_TEMPLATES = ["Compare {nct1} and {nct2}.", "How do {nct1} and {nct2} differ in design?"]

ANSWER_MIX = {"search": 0.6, "detail": 0.2, "eligibility": 0.1, "compare": 0.1}
TOOL_MIX = {"search": 0.5, "detail": 0.2, "eligibility": 0.15, "compare": 0.15}
TOOL_FOR_KIND = {"search": "search_trials", "detail": "get_trial",
                 "eligibility": "get_eligibility", "compare": "compare_trials"}


def fetch_seed_trials(conditions: list[str], per_condition: int, ctgov) -> dict:
    seeds = {}
    for condition in conditions:
        params = ctgov.build_query_params(condition=condition, page_size=min(per_condition, 100))
        try:
            seeds[condition] = ctgov.fetch_trials(params, max_results=per_condition)
        except Exception as exc:
            print(f"warning: could not fetch {condition}: {exc}", file=sys.stderr)
            seeds[condition] = []
        time.sleep(0.2)
    return {c: t for c, t in seeds.items() if t}


def split_conditions(conditions: list[str], heldout_frac: float, rng: random.Random):
    shuffled = sorted(conditions)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * heldout_frac))
    return sorted(shuffled[cut:]), sorted(shuffled[:cut])  # dev, heldout


def _phase_status(trial: dict):
    phase = next((PHASE_WORDS[p] for p in trial.get("phases", []) if p in PHASE_WORDS), None)
    return phase, STATUS_WORDS.get(trial.get("status", ""))


def make_question(kind: str, condition: str, trials: list[dict], rng: random.Random) -> dict:
    """One sample of the given kind built from this condition's seed trials."""
    trial = rng.choice(trials)
    if kind == "detail":
        question = rng.choice(DETAIL_TEMPLATES).format(nct=trial["nct_id"])
    elif kind == "eligibility":
        question = rng.choice(ELIGIBILITY_TEMPLATES).format(nct=trial["nct_id"])
    elif kind == "compare" and len(trials) >= 2:
        first, second = rng.sample(trials, 2)
        question = rng.choice(COMPARE_TEMPLATES).format(nct1=first["nct_id"], nct2=second["nct_id"])
    else:
        kind = "search"
        phase, status = _phase_status(trial)
        intervention = (trial.get("interventions") or [condition])[0]
        template = rng.choice(SEARCH_TEMPLATES)
        if "{phase}" in template and not phase:
            template = SEARCH_TEMPLATES[1]
        if "{status}" in template and not status:
            template = SEARCH_TEMPLATES[3]
        question = template.format(phase=phase, status=status, condition=condition, intervention=intervention)
    return {"input": question, "target": TOOL_FOR_KIND[kind], "kind": kind, "condition": condition}


def build_samples(seeds: dict, conditions: list[str], n: int, mix: dict, rng: random.Random) -> list[dict]:
    """n unique samples drawn over `conditions` following `mix` (kind -> share)."""
    pool = [c for c in conditions if c in seeds]
    if not pool:
        return []
    kinds = [k for k, share in mix.items() for _ in range(round(share * 100))]
    samples, seen = [], set()
    attempts = 0
    while len(samples) < n and attempts < n * 30:
        attempts += 1
        condition = rng.choice(pool)
        sample = make_question(rng.choice(kinds), condition, seeds[condition], rng)
        if sample["input"] in seen:
            continue
        seen.add(sample["input"])
        samples.append(sample)
    return samples


def build_retrieval_samples(seeds: dict, conditions: list[str], n: int, rng: random.Random) -> list[dict]:
    """Questions built from one trial's own condition/intervention/phase/status."""
    candidates = []
    for condition in conditions:
        for trial in seeds.get(condition, []):
            phase, status = _phase_status(trial)
            if not (trial.get("interventions") and phase and status):
                continue
            question = f"What {phase} trials of {trial['interventions'][0]} for {condition} are {status}?"
            candidates.append({"input": question, "target": trial["nct_id"], "condition": condition,
                               "expected_nct": trial["nct_id"]})
    rng.shuffle(candidates)
    seen, out = set(), []
    for item in candidates:
        if item["input"] not in seen:
            seen.add(item["input"])
            out.append(item)
        if len(out) >= n:
            break
    return out


def to_jsonl(samples: list[dict], prefix: str, path: Path) -> None:
    with path.open("w") as handle:
        for index, sample in enumerate(samples):
            metadata = {k: v for k, v in sample.items() if k not in {"input", "target"}}
            handle.write(json.dumps({
                "id": f"{prefix}{index:04d}", "input": sample["input"],
                "target": sample["target"] if prefix != "a" else "", "metadata": metadata,
            }) + "\n")


def paraphrase(samples: list[dict], pipeline) -> list[dict]:
    """Optional: rewrite questions in natural language with the planner model. A rewrite
    is kept only if it still contains the same NCT IDs."""
    from src.agent.tools import find_nct_ids

    out = []
    for sample in samples:
        prompt = [("human", "Rewrite this question about clinical trials in different wording. "
                            "Keep every NCT ID, drug, and disease exactly as written. "
                            f"Return only the question.\n\n{sample['input']}")]
        try:
            text = pipeline.deps.registry.call(pipeline.config.roles.planner, prompt).text.strip().strip('"')
        except Exception:
            text = ""
        if text and set(find_nct_ids(text)) == set(find_nct_ids(sample["input"])) and len(text) < 300:
            sample = {**sample, "input": text, "paraphrased": True}
        out.append(sample)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="evals/datasets")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-answer", type=int, default=1000)
    parser.add_argument("--n-tool", type=int, default=100)
    parser.add_argument("--n-retrieval", type=int, default=200)
    parser.add_argument("--per-condition", type=int, default=40, help="seed trials fetched per condition")
    parser.add_argument("--heldout-frac", type=float, default=0.25)
    parser.add_argument("--seed-file", help="reuse a saved seed_trials.json instead of calling the API")
    parser.add_argument("--paraphrase", action="store_true", help="paraphrase questions with the planner model")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed_file:
        seeds = json.loads(Path(args.seed_file).read_text())
    else:
        from src.retrieval import ctgov_client
        seeds = fetch_seed_trials(CONDITIONS, args.per_condition, ctgov_client)
        (out_dir / "seed_trials.json").write_text(json.dumps(seeds))

    dev, heldout = split_conditions(list(seeds), args.heldout_frac, rng)
    answer = build_samples(seeds, dev, args.n_answer, ANSWER_MIX, rng)
    tool = build_samples(seeds, heldout, args.n_tool, TOOL_MIX, rng)
    retrieval = build_retrieval_samples(seeds, dev + heldout, args.n_retrieval, rng)

    if args.paraphrase:
        from src.pipeline import ClinicalPipeline
        pipeline = ClinicalPipeline()
        answer, tool = paraphrase(answer, pipeline), paraphrase(tool, pipeline)

    to_jsonl(answer, "a", out_dir / "answer_queries.jsonl")
    to_jsonl(tool, "t", out_dir / "tool_queries.jsonl")
    to_jsonl(retrieval, "r", out_dir / "retrieval_queries.jsonl")
    (out_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed, "dev_conditions": dev, "heldout_conditions": heldout,
        "counts": {"answer": len(answer), "tool": len(tool), "retrieval": len(retrieval)},
        "paraphrased": args.paraphrase,
    }, indent=2))
    print(f"answer={len(answer)} tool={len(tool)} retrieval={len(retrieval)} -> {out_dir}")


if __name__ == "__main__":
    main()
