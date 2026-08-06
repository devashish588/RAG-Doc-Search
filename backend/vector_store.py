import re
from functools import lru_cache
from threading import RLock
from typing import Any

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
def _sentence_transformer_embeddings(*, local_files_only: bool) -> Embeddings:
    from sentence_transformers import SentenceTransformer

    kwargs: dict = {"model_name_or_path": EMBEDDING_MODEL}
    if local_files_only:
        kwargs["local_files_only"] = True
    try:
        model = SentenceTransformer(**kwargs)
    except TypeError:
        kwargs.pop("local_files_only", None)
        model = SentenceTransformer(**kwargs)
    return _SentenceTransformerEmbeddings(model)


class _SentenceTransformerEmbeddings(Embeddings):
    """Embeddings interface backed directly by sentence-transformers."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

# vectorstore.py
@lru_cache(maxsize=1)
def get_embedding_backend() -> str:
    if EMBEDDING_BACKEND in {"hash", "hashing"}:
        return "hashing"
    try:
        # Allow downloading from Hugging Face if not cached locally
        _sentence_transformer_embeddings(local_files_only=False)
        return "huggingface"
    except Exception as exc:
        print(f"Failed to load SentenceTransformer: {exc}")
        return "hashing"


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    if get_embedding_backend() == "huggingface":
        return _sentence_transformer_embeddings(local_files_only=False)
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
    
    # Use MMR search to ensure diverse chunks (e.g., Page 1, Page 2, Page 3)
    try:
        docs = store.max_marginal_relevance_search(
            query, 
            k=k, 
            fetch_k=12,  # Fetch top 20 similar candidates first
            lambda_mult=0.5,  # 0.5 balances relevance and diversity
            filter=filters
        )
        # Assign default score placeholder since MMR returns Documents
        return [(doc, 1.0) for doc in docs]
    except Exception:
        # Fallback to standard similarity search
        pairs = store.similarity_search_with_relevance_scores(query, k=k, filter=filters)
        return [(doc, _clamp(score)) for doc, score in pairs]


def delete_by_document(document_id: str) -> int:
    """Remove every indexed chunk belonging to a document. Returns count deleted."""
    store = get_vector_store()
    collection = getattr(store, "_collection", None)
    ids: list[str] = []
    if collection is not None:
        try:
            ids = list(collection.get(where={"document_id": document_id}).get("ids", []))
        except Exception:
            ids = []
    if not ids:
        return 0
    with _VECTOR_LOCK:
        try:
            collection.delete(ids=ids)
        except Exception:
            store.delete(ids=ids)
        if callable(getattr(store, "persist", None)):
            store.persist()
    return len(ids)
