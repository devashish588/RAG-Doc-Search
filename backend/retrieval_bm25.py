"""BM25 sparse retrieval implementation.

Implements the BaseSparseRetriever interface using BM25Okapi.
"""
import logging
from typing import Any

from backend.interfaces import BaseSparseRetriever, RetrievalResult
from backend.bm25_index import get_bm25_index
from backend.settings import BM25_TOP_K, BM25_K1, BM25_B

log = logging.getLogger(__name__)


class BM25Retriever(BaseSparseRetriever):
    """BM25 sparse retriever using persistent index.

    Uses rank_bm25 BM25Okapi with configurable parameters.
    Indexes canonical chunks from the ingestion pipeline.
    """

    def __init__(
        self,
        top_k: int = BM25_TOP_K,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be in [0, 1], got {b}")
        if k1 <= 0:
            raise ValueError(f"k1 must be > 0, got {k1}")

        self.top_k = top_k
        self.k1 = k1
        self.b = b

        # Verify index parameters match
        index = get_bm25_index()
        if abs(index.k1 - k1) > 1e-6 or abs(index.b - b) > 1e-6:
            log.warning(
                "BM25Retriever parameters (k1=%.2f, b=%.2f) differ from index (k1=%.2f, b=%.2f)",
                k1, b, index.k1, index.b
            )

    def search(
        self,
        query: str,
        k: int | None = None,
        source: str | None = None,
    ) -> list[RetrievalResult]:
        """Search BM25 index for relevant chunks.

        Args:
            query: Search query text
            k: Override top_k (uses instance default if None)
            source: Optional source filename filter

        Returns:
            List of RetrievalResult ordered by rank (1 = best)
        """
        if not query or not query.strip():
            return []

        effective_k = k if k is not None else self.top_k

        index = get_bm25_index()
        if not index.is_healthy():
            log.warning("BM25 index not healthy, returning empty results")
            return []

        results = index.search(query=query, k=effective_k, source=source)

        retrieval_results = []
        for r in results:
            retrieval_results.append(RetrievalResult(
                chunk_id=r["chunk_id"],
                score=r["score"],
                rank=r["rank"],
                source=r["source"],
                content=r["content"],
                metadata=r["metadata"],
            ))

        return retrieval_results

    def index(self, chunks: list) -> None:
        """Add chunks to the BM25 index.

        Note: BM25Okapi doesn't support true incremental updates efficiently.
        This method rebuilds the entire index with the provided chunks.
        For incremental updates during ingestion, use the ingestion pipeline
        which calls index.rebuild() after all chunks are ready.
        """
        if not chunks:
            return

        index = get_bm25_index()
        log.info("Indexing %d chunks to BM25", len(chunks))
        index.build(chunks)
        index.save()

    def delete(self, document_id: str) -> None:
        """Remove all chunks for a document from the BM25 index."""
        index = get_bm25_index()
        removed = index.remove_document(document_id)
        if removed > 0:
            index.save()
            log.info("Removed %d BM25 chunks for document %s", removed, document_id)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_BM25_RETRIEVER_INSTANCE: BM25Retriever | None = None


def get_bm25_retriever() -> BM25Retriever:
    """Get or create the singleton BM25Retriever instance."""
    global _BM25_RETRIEVER_INSTANCE
    if _BM25_RETRIEVER_INSTANCE is None:
        _BM25_RETRIEVER_INSTANCE = BM25Retriever()
    return _BM25_RETRIEVER_INSTANCE


def reset_bm25_retriever() -> None:
    """Reset the singleton (mainly for testing)."""
    global _BM25_RETRIEVER_INSTANCE
    _BM25_RETRIEVER_INSTANCE = None