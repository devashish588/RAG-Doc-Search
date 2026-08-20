"""Dense retrieval implementation.

Refactors the existing dense MMR retrieval behind the BaseDenseRetriever interface.
Preserves exact existing behavior while making configuration explicit.
"""
import logging
from typing import Any

from langchain_core.documents import Document

from backend.interfaces import BaseDenseRetriever, RetrievalResult
from backend.settings import (
    DENSE_FETCH_K,
    DENSE_MMR_LAMBDA,
    DENSE_TOP_K,
    MIN_RELEVANCE_SCORE,
)
from backend.vector_store import get_vector_store

log = logging.getLogger(__name__)


def _clamp(score: float) -> float:
    """Clamp score to [0, 1] range."""
    return round(max(0.0, min(float(score), 1.0)), 4)


def _page(metadata: dict[str, Any]) -> int | None:
    """Extract page number from metadata."""
    p = metadata.get("page")
    if isinstance(p, int):
        return p
    if isinstance(p, str) and p.isdigit():
        return int(p)
    return None


def _to_retrieval_result(
    doc: Document,
    score: float,
    rank: int,
) -> RetrievalResult:
    """Convert LangChain Document to canonical RetrievalResult."""
    meta = dict(doc.metadata)
    chunk_id = f"{meta.get('document_id', '')}:{meta.get('chunk', '')}"
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        source=str(meta.get("source", "unknown")),
        content=doc.page_content,
        metadata=meta,
    )


class DenseRetriever(BaseDenseRetriever):
    """Dense vector retriever using Chroma MMR search.

    Preserves the exact existing behavior:
    - FastEmbed BAAI/bge-small-en-v1.5 embeddings
    - ChromaDB vector store
    - Max Marginal Relevance Search with configurable parameters
    - MIN_RELEVANCE_SCORE threshold
    """

    def __init__(
        self,
        top_k: int = DENSE_TOP_K,
        fetch_k: int = DENSE_FETCH_K,
        lambda_mult: float = DENSE_MMR_LAMBDA,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
    ):
        # Validate configuration
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if fetch_k < top_k:
            raise ValueError(f"fetch_k ({fetch_k}) must be >= top_k ({top_k})")
        if not (0.0 <= lambda_mult <= 1.0):
            raise ValueError(f"lambda_mult must be in [0, 1], got {lambda_mult}")
        if not (0.0 <= min_relevance_score <= 1.0):
            raise ValueError(f"min_relevance_score must be in [0, 1], got {min_relevance_score}")

        self.top_k = top_k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.min_relevance_score = min_relevance_score

    def search(
        self,
        query: str,
        k: int | None = None,
        source: str | None = None,
    ) -> list[RetrievalResult]:
        """Search for relevant chunks using MMR.

        Args:
            query: Search query text
            k: Override top_k (uses instance default if None)
            source: Optional source filename filter

        Returns:
            List of RetrievalResult ordered by rank (1 = best)
        """
        if not query or not query.strip():
            return []

        # Use instance default if k not provided
        effective_k = k if k is not None else self.top_k
        effective_fetch_k = max(self.fetch_k, effective_k)

        query = " ".join(query.split())
        store = get_vector_store()
        filters = {"source": source} if source else None

        try:
            # MMR search - retrieves varied content
            docs = store.max_marginal_relevance_search(
                query,
                k=effective_k,
                fetch_k=effective_fetch_k,
                lambda_mult=self.lambda_mult,
                filter=filters,
            )
            # MMR returns docs without scores; use 1.0 as placeholder
            pairs = [(doc, 1.0) for doc in docs]
        except Exception as exc:
            log.warning("MMR search failed, falling back to similarity: %s", exc)
            # Fallback to similarity search with relevance scores
            pairs = store.similarity_search_with_relevance_scores(
                query,
                k=effective_k,
                filter=filters,
            )
            pairs = [(doc, _clamp(score)) for doc, score in pairs]

        # Deduplicate by normalized text content
        seen_text = set()
        unique_pairs = []
        for doc, score in pairs:
            normalized_text = doc.page_content.strip()
            if normalized_text in seen_text:
                continue
            seen_text.add(normalized_text)
            unique_pairs.append((doc, score))
            if len(unique_pairs) >= effective_k:
                break

        # Apply relevance threshold
        filtered_pairs = [
            (doc, score) for doc, score in unique_pairs
            if score >= self.min_relevance_score
        ]

        # Convert to canonical results with deterministic ranking
        results = []
        for rank, (doc, score) in enumerate(filtered_pairs, 1):
            results.append(_to_retrieval_result(doc, score, rank))

        return results


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_DENSE_RETRIEVER_INSTANCE: DenseRetriever | None = None


def get_dense_retriever() -> DenseRetriever:
    """Get or create the singleton DenseRetriever instance."""
    global _DENSE_RETRIEVER_INSTANCE
    if _DENSE_RETRIEVER_INSTANCE is None:
        _DENSE_RETRIEVER_INSTANCE = DenseRetriever()
    return _DENSE_RETRIEVER_INSTANCE


def reset_dense_retriever() -> None:
    """Reset the singleton (mainly for testing)."""
    global _DENSE_RETRIEVER_INSTANCE
    _DENSE_RETRIEVER_INSTANCE = None