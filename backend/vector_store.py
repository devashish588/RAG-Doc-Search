import logging
import os
import re
from functools import lru_cache
from threading import RLock

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.settings import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    HASHING_EMBEDDING_DIMS,
    ensure_runtime_dirs,
)

log = logging.getLogger(__name__)

_FASTEMBED_CACHE_DIR = DATA_DIR / "fastembed_cache"

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
def _fastembed_embeddings() -> Embeddings:
    """FastEmbed (ONNX) embeddings - lightweight, no PyTorch.

    fastembed reuses its on-disk cache and only downloads a model when it is
    missing from `cache_dir`.
    """
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_dir=str(_FASTEMBED_CACHE_DIR),
        providers=["CPUExecutionProvider"],
        threads=1,
        batch_size=int(os.getenv("FASTEMBED_BATCH_SIZE", "2")),
    )


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_backend() -> str:
    from backend.settings import ENVIRONMENT

    if EMBEDDING_BACKEND in {"hash", "hashing"}:
        return "hashing"
    try:
        _fastembed_embeddings()
        return "fastembed"
    except Exception as exc:
        if ENVIRONMENT == "production":
            log.critical(
                "CRITICAL: FastEmbed unavailable in production (EMBEDDING_BACKEND=%s). "
                "Install using: pip install -r requirements.txt. Error: %s",
                EMBEDDING_BACKEND, exc,
            )
            raise RuntimeError(
                f"FastEmbed is required in production but failed to load: {exc}. "
                "Install using: pip install -r requirements.txt"
            ) from exc
        log.warning("Failed to load FastEmbed, falling back to hashing: %s", exc)
        return "hashing"


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    if get_embedding_backend() == "fastembed":
        return _fastembed_embeddings()
    return HashingEmbeddings()

# ---------------------------------------------------------------------------
# Chroma collection
# ---------------------------------------------------------------------------

def _collection_name() -> str:
    """Namespace collection by backend so switching backends never mixes vectors."""
    backend = get_embedding_backend()
    if backend == "fastembed":
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBEDDING_MODEL.split("/")[-1]).strip("_").lower()
        return f"{COLLECTION_NAME}_{backend}_{slug}"
    return f"{COLLECTION_NAME}_{backend}"


@lru_cache(maxsize=1)
def get_vector_store():
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
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
    batch = int(os.getenv("EMBEDDING_BATCH_SIZE", "4"))
    added = 0
    with _VECTOR_LOCK:
        store = get_vector_store()
        for i in range(0, len(documents), batch):
            chunk_docs = documents[i:i + batch]
            chunk_ids  = ids[i:i + batch]
            store.add_documents(documents=chunk_docs, ids=chunk_ids)
            added += len(chunk_docs)
        if callable(getattr(store, "persist", None)):
            store.persist()
    return added


def _clamp(score: float) -> float:
    return round(max(0.0, min(float(score), 1.0)), 4)


def search_similar(
    query: str,
    k: int = 8,
    source: str | None = None,
) -> list[tuple[Document, float]]:
    store = get_vector_store()
    filters = {"source": source} if source else None
    
    try:
        # MMR search retrieves varied content across the document
        docs = store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=max(20, k),  # Fetch 20+ candidate chunks first (must be >= k)
            lambda_mult=0.5,     # 0.5 balances query relevance with context diversity
            filter=filters
        )
        return [(doc, 1.0) for doc in docs]
    except Exception:
        # Fallback to similarity search if MMR fails
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
