import json
import random

from evals import generate_queries as gq
from tests.conftest import TRIALS

SEEDS = {
    "type 2 diabetes": [t for t in TRIALS if "Diabetes" in t["conditions"][0]],
    "melanoma": [t for t in TRIALS if "Melanoma" in t["conditions"][0]],
}


def test_split_is_disjoint_and_deterministic():
    conditions = [f"c{i}" for i in range(20)]
    dev1, held1 = gq.split_conditions(conditions, 0.25, random.Random(1))
    dev2, held2 = gq.split_conditions(conditions, 0.25, random.Random(1))
    assert (dev1, held1) == (dev2, held2) and not set(dev1) & set(held1) and len(held1) == 5


def test_samples_follow_mix_and_are_unique():
    samples = gq.build_samples(SEEDS, ["type 2 diabetes"], 12, gq.TOOL_MIX, random.Random(3))
    assert len({s["input"] for s in samples}) == len(samples)
    assert {s["target"] for s in samples} <= {"search_trials", "get_trial", "get_eligibility", "compare_trials"}


def test_each_kind_produces_matching_question_shape():
    rng = random.Random(0)
    trials = SEEDS["type 2 diabetes"]
    assert gq.make_question("detail", "x", trials, rng)["target"] == "get_trial"
    assert "NCT" in gq.make_question("eligibility", "x", trials, rng)["input"]
    compare = gq.make_question("compare", "x", trials, rng)
    assert compare["target"] == "compare_trials" and compare["input"].count("NCT") == 2


def test_compare_with_one_trial_degrades_to_search():
    sample = gq.make_question("compare", "melanoma", SEEDS["melanoma"], random.Random(0))
    assert sample["target"] == "search_trials"


def test_retrieval_samples_point_at_their_seed_trial():
    samples = gq.build_retrieval_samples(SEEDS, list(SEEDS), 5, random.Random(0))
    assert samples and all(s["target"] == s["expected_nct"] and s["expected_nct"].startswith("NCT") for s in samples)


def test_main_from_seed_file_writes_all_files(tmp_path):
    seed_file = tmp_path / "seeds.json"
    many = {f"cond{i}": [{**TRIALS[0], "nct_id": f"NCT{i:04d}{j:04d}"} for j in range(5)] for i in range(8)}
    seed_file.write_text(json.dumps(many))
    out = tmp_path / "out"
    gq.main(["--seed-file", str(seed_file), "--out-dir", str(out), "--n-answer", "20", "--n-tool", "8", "--n-retrieval", "5"])
    for name in ("answer_queries", "tool_queries", "retrieval_queries"):
        lines = [json.loads(l) for l in (out / f"{name}.jsonl").read_text().splitlines()]
        assert lines and {"id", "input", "target", "metadata"} <= set(lines[0])
    manifest = json.loads((out / "manifest.json").read_text())
    assert not set(manifest["dev_conditions"]) & set(manifest["heldout_conditions"])
    tool_conditions = {json.loads(l)["metadata"]["condition"] for l in (out / "tool_queries.jsonl").read_text().splitlines()}
    assert tool_conditions <= set(manifest["heldout_conditions"])


def test_same_seed_same_output(tmp_path):
    seed_file = tmp_path / "s.json"
    seed_file.write_text(json.dumps({f"c{i}": [{**TRIALS[0], "nct_id": f"NCT0000{i}00{j}"} for j in range(4)] for i in range(6)}))
    for name in ("a", "b"):
        gq.main(["--seed-file", str(seed_file), "--out-dir", str(tmp_path / name), "--n-answer", "10", "--n-tool", "5", "--n-retrieval", "3", "--seed", "7"])
    assert (tmp_path / "a" / "answer_queries.jsonl").read_text() == (tmp_path / "b" / "answer_queries.jsonl").read_text()
