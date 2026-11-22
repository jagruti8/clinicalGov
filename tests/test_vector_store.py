from src.retrieval.vector_store import retrieve_chunks, retrieve_documents, store_trials
from tests.conftest import TRIALS, FakeChroma


def test_store_trials_skips_ids_already_stored():
    store = FakeChroma()
    assert store_trials(TRIALS[:2], store) == 2
    assert store_trials(TRIALS, store) == 1
    assert set(store.docs) == {"NCT00000001", "NCT00000002", "NCT00000003"}


def test_store_trials_dedupes_within_one_batch():
    store = FakeChroma()
    assert store_trials([TRIALS[0], TRIALS[0]], store) == 1


def test_retrieval_is_restricted_to_current_fetch():
    store = FakeChroma()
    store_trials(TRIALS, store)
    docs = retrieve_documents("metformin diabetes", store, k=5, restrict_ids=["NCT00000002", "NCT00000003"])
    assert {d["nct_id"] for d in docs} <= {"NCT00000002", "NCT00000003"}


def test_rerank_boosts_matching_phase_and_status():
    store = FakeChroma()
    store_trials(TRIALS, store)
    plain = retrieve_documents("trials", store, k=3)
    params = {"aggFilters": "phase:2,status:com"}
    boosted = retrieve_documents("trials", store, k=3, query_params=params)
    assert boosted[0]["nct_id"] == "NCT00000002"          # PHASE2 + COMPLETED
    assert boosted[0]["rank_score"] > boosted[0]["score"]
    assert {d["nct_id"] for d in plain} == {d["nct_id"] for d in boosted}


def test_raw_score_is_kept_for_relevance_threshold():
    store = FakeChroma()
    store_trials(TRIALS, store)
    docs = retrieve_documents("metformin", store, k=1, query_params={"aggFilters": "phase:3"})
    assert docs[0]["score"] < docs[0]["rank_score"]


def test_retrieve_chunks_returns_texts_and_scores():
    store = FakeChroma()
    store_trials(TRIALS, store)
    chunks, scores = retrieve_chunks("melanoma", store, k=2)
    assert len(chunks) == len(scores) == 2
