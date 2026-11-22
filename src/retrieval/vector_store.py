"""ChromaDB storage and retrieval over trials fetched live from ClinicalTrials.gov.

The store is a per-query working set plus a cache: trials are embedded once (by
NCT ID) and retrieval is restricted to the IDs fetched for the current query, so
results always come from the live fetch, never from unrelated cached trials.
"""
from typing import Optional

from src.retrieval.ctgov_client import format_trial_text

COLLECTION_NAME = "clinical_trials"
CHROMA_PATH = "./chroma_db"
METADATA_BOOST = 0.05  # added per matching phase/status when reranking


def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(embeddings=None, persist_directory: str = CHROMA_PATH):
    from langchain_chroma import Chroma

    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def store_trials(trials: list[dict], vector_store, eligibility_chars: int = 1500) -> int:
    """Embed trials not already stored. Returns how many were added."""
    from langchain_core.documents import Document

    try:
        existing_ids = set(vector_store._collection.get(include=[]).get("ids", []))
    except Exception:
        existing_ids = set()

    docs, ids = [], []
    for trial in trials:
        nct_id = trial["nct_id"]
        if nct_id in existing_ids or nct_id in ids:
            continue
        docs.append(Document(
            page_content=format_trial_text(trial, eligibility_chars),
            metadata={
                "nct_id": nct_id,
                "status": trial.get("status", ""),
                "phases": ", ".join(trial.get("phases", [])),
                "conditions": ", ".join(trial.get("conditions", [])),
            },
        ))
        ids.append(nct_id)

    if docs:
        vector_store.add_documents(docs, ids=ids)
    return len(docs)


def _boost(metadata: dict, query_params: Optional[dict]) -> float:
    """Small deterministic boost when a trial's phase/status matches what was asked."""
    if not query_params:
        return 0.0
    boost = 0.0
    filters = query_params.get("aggFilters", "")
    phase = next((p.split(":")[1] for p in filters.split(",") if p.startswith("phase:")), None)
    status = next((p.split(":")[1] for p in filters.split(",") if p.startswith("status:")), None)
    if phase and f"PHASE{phase}" in (metadata.get("phases") or ""):
        boost += METADATA_BOOST
    status_names = {"rec": "RECRUITING", "com": "COMPLETED", "not-rec": "NOT_YET_RECRUITING"}
    if status and status_names.get(status) == metadata.get("status"):
        boost += METADATA_BOOST
    return boost


def retrieve_documents(
    question: str,
    vector_store,
    k: int = 5,
    restrict_ids: Optional[list[str]] = None,
    query_params: Optional[dict] = None,
) -> list[dict]:
    """Top-k trial documents for the question. Each result:
    {"text", "nct_id", "score" (raw similarity), "rank_score" (after boost)}."""
    kwargs = {"k": k * 2}
    if restrict_ids:
        kwargs["filter"] = {"nct_id": {"$in": list(restrict_ids)}}
    results = vector_store.similarity_search_with_relevance_scores(question, **kwargs)

    ranked = []
    for doc, score in results:
        ranked.append({
            "text": doc.page_content,
            "nct_id": doc.metadata.get("nct_id", ""),
            "score": float(score),
            "rank_score": float(score) + _boost(doc.metadata, query_params),
        })
    ranked.sort(key=lambda item: item["rank_score"], reverse=True)
    return ranked[:k]


def retrieve_chunks(question: str, vector_store, k: int = 5) -> tuple[list[str], list[float]]:
    """Chunk texts and their raw scores for the top k results."""
    docs = retrieve_documents(question, vector_store, k=k)
    return [d["text"] for d in docs], [d["score"] for d in docs]
