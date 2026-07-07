from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.retrieval.ctgov_client import format_trial_text

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "clinical_trials"
CHROMA_PATH = "./chroma_db"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(embeddings: HuggingFaceEmbeddings = None) -> Chroma:
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )


def store_trials(trials: list[dict], vector_store: Chroma) -> None:
    try:
        existing = vector_store._collection.get(include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    docs, ids = [], []
    for trial in trials:
        nct_id = trial["nct_id"]
        if nct_id in existing_ids:
            continue
        docs.append(Document(
            page_content=format_trial_text(trial),
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


def retrieve_chunks(
    question: str, vector_store: Chroma, k: int = 5
) -> tuple[list[str], list[float]]:
    results = vector_store.similarity_search_with_relevance_scores(question, k=k)
    chunks = [doc.page_content for doc, _ in results]
    scores = [score for _, score in results]
    return chunks, scores
