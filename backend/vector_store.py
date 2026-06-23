import re
from functools import lru_cache
from threading import RLock

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import HashingVectorizer

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - compatibility for older installs
    from langchain_community.vectorstores import Chroma

from backend.settings import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    HASHING_EMBEDDING_DIMENSIONS,
    ensure_runtime_dirs,
)


_VECTOR_LOCK = RLock()


class HashingEmbeddings(Embeddings):
    """Fully local embedding fallback that never needs model downloads."""

    def __init__(self, n_features: int = HASHING_EMBEDDING_DIMENSIONS) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(texts)
        return matrix.astype("float32").toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "default"


@lru_cache(maxsize=1)
def _load_huggingface_embeddings(*, local_files_only: bool) -> HuggingFaceEmbeddings:
    model_kwargs = {"local_files_only": local_files_only}
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)


@lru_cache(maxsize=1)
def get_embedding_backend() -> str:
    """Resolve the active embedding backend once per process."""
    if EMBEDDING_BACKEND in {"hash", "hashing", "local"}:
        return "hashing"
    if EMBEDDING_BACKEND in {"hf", "huggingface"}:
        _load_huggingface_embeddings(local_files_only=False)
        return "huggingface"

    try:
        _load_huggingface_embeddings(local_files_only=True)
        return "huggingface"
    except Exception:
        return "hashing"


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Load the active embedding model once per process."""
    if get_embedding_backend() == "huggingface":
        return _load_huggingface_embeddings(local_files_only=(EMBEDDING_BACKEND != "huggingface"))
    return HashingEmbeddings()


def get_collection_name() -> str:
    """Keep Chroma collections separate when the embedding backend changes."""
    backend = get_embedding_backend()
    if backend == "huggingface":
        model_slug = _slugify(EMBEDDING_MODEL.split("/")[-1])
        return f"{COLLECTION_NAME}_{backend}_{model_slug}"
    return f"{COLLECTION_NAME}_{backend}"


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Return the persistent Chroma collection used by the app."""
    ensure_runtime_dirs()
    return Chroma(
        collection_name=get_collection_name(),
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
