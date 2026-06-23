from functools import lru_cache
from threading import RLock

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - compatibility for older installs
    from langchain_community.vectorstores import Chroma

from backend.settings import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, ensure_runtime_dirs


_VECTOR_LOCK = RLock()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the HuggingFace sentence embedding model once per process."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Return the persistent Chroma collection used by the app."""
    ensure_runtime_dirs()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def _clamp_relevance(score: float) -> float:
    return round(max(0.0, min(score, 1.0)), 4)


def _score_from_distance(distance: float) -> float:
    if distance < 0:
        return 0.0
    return round(1 / (1 + distance), 4)


def add_documents(documents: list[Document], ids: list[str]) -> int:
    """Add chunks to Chroma and return the number of indexed chunks."""
    if not documents:
        return 0

    with _VECTOR_LOCK:
        vector_store = get_vector_store()
        vector_store.add_documents(documents=documents, ids=ids)
        persist = getattr(vector_store, "persist", None)
        if callable(persist):
            persist()
    return len(documents)


def search_similar(
    query: str,
    k: int = 5,
    source: str | None = None,
) -> list[tuple[Document, float]]:
    """Search Chroma and return documents with normalized relevance scores."""
    vector_store = get_vector_store()
    filters = {"source": source} if source else None

    try:
        relevance_pairs = vector_store.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filters,
        )
        return [(doc, _clamp_relevance(float(score))) for doc, score in relevance_pairs]
    except Exception:
        distance_pairs = vector_store.similarity_search_with_score(query, k=k, filter=filters)
        return [(doc, _score_from_distance(float(distance))) for doc, distance in distance_pairs]
