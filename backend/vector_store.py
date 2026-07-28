import re
from functools import lru_cache
from threading import RLock

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.settings import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    HASHING_EMBEDDING_DIMS,
    ensure_runtime_dirs,
)

_VECTOR_LOCK = RLock()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class HashingEmbeddings(Embeddings):
    """Fully local, zero-download embedding fallback using sklearn."""

    def __init__(self, n_features: int = HASHING_EMBEDDING_DIMS) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer
        self._vec = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._vec.transform(texts).astype("float32").toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def _huggingface_embeddings(*, local_files_only: bool) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": local_files_only},
    )


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_backend() -> str:
    if EMBEDDING_BACKEND in {"hash", "hashing", "local"}:
        return "hashing"
    if EMBEDDING_BACKEND in {"hf", "huggingface"}:
        _huggingface_embeddings(local_files_only=False)
        return "huggingface"
    # auto: try cached local model first, fall back to hashing
    try:
        _huggingface_embeddings(local_files_only=True)
        return "huggingface"
    except Exception:
        return "hashing"


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    if get_embedding_backend() == "huggingface":
        return _huggingface_embeddings(local_files_only=(EMBEDDING_BACKEND != "huggingface"))
    return HashingEmbeddings()


# ---------------------------------------------------------------------------
# Chroma collection
# ---------------------------------------------------------------------------

def _collection_name() -> str:
    """Namespace collection by backend so switching backends never mixes vectors."""
    backend = get_embedding_backend()
    if backend == "huggingface":
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBEDDING_MODEL.split("/")[-1]).strip("_").lower()
        return f"{COLLECTION_NAME}_{backend}_{slug}"
    return f"{COLLECTION_NAME}_{backend}"


@lru_cache(maxsize=1)
def get_vector_store():
    from langchain_chroma import Chroma
    ensure_runtime_dirs()
    return Chroma(
        collection_name=_collection_name(),
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def add_documents(documents: list[Document], ids: list[str]) -> int:
    if not documents:
        return 0
    with _VECTOR_LOCK:
        store = get_vector_store()
        store.add_documents(documents=documents, ids=ids)
        if callable(getattr(store, "persist", None)):
            store.persist()
    return len(documents)


def _clamp(score: float) -> float:
    return round(max(0.0, min(float(score), 1.0)), 4)


def search_similar(
    query: str,
    k: int = 5,
    source: str | None = None,
) -> list[tuple[Document, float]]:
    store = get_vector_store()
    filters = {"source": source} if source else None
    try:
        pairs = store.similarity_search_with_relevance_scores(query, k=k, filter=filters)
        return [(doc, _clamp(score)) for doc, score in pairs]
    except Exception:
        pairs = store.similarity_search_with_score(query, k=k, filter=filters)
        return [(doc, _clamp(1 / (1 + max(0.0, float(d))))) for doc, d in pairs]
