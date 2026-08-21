"""Hybrid RRF retrieval implementation.

Combines DenseRetriever and BM25Retriever using Reciprocal Rank Fusion (RRF).
"""
import logging
from time import perf_counter
from typing import Any

from backend.interfaces import RetrievalResult
from backend.retrieval_dense import get_dense_retriever
from backend.retrieval_bm25 import get_bm25_retriever
from backend.settings import (
    RRF_K,
    RRF_DENSE_WEIGHT,
    RRF_SPARSE_WEIGHT,
    RRF_TOP_K,
)

log = logging.getLogger(__name__)


def _rrf_score(
    dense_rank: int | None,
    sparse_rank: int | None,
    dense_weight: float,
    sparse_weight: float,
    k: int,
) -> float:
    """Calculate RRF score for a candidate.

    RRF(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))

    Args:
        dense_rank: 1-based rank in dense results (None if not in dense results)
        sparse_rank: 1-based rank in sparse results (None if not in sparse results)
        dense_weight: Weight for dense retriever
        sparse_weight: Weight for sparse retriever
        k: RRF constant (typically 60)

    Returns:
        RRF fusion score
    """
    score = 0.0
    if dense_rank is not None:
        score += dense_weight / (k + dense_rank)
    if sparse_rank is not None:
        score += sparse_weight / (k + sparse_rank)
    return score


class HybridRRFRetriever:
    """Hybrid retriever using Reciprocal Rank Fusion (RRF).

    Combines DenseRetriever and BM25Retriever results using weighted RRF.
    """

    def __init__(
        self,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        top_k_fused: int = 10,
        rrf_k: int = RRF_K,
        dense_weight: float = RRF_DENSE_WEIGHT,
        sparse_weight: float = RRF_SPARSE_WEIGHT,
    ):
        # Validate configuration
        if top_k_dense <= 0:
            raise ValueError("top_k_dense must be > 0")
        if top_k_sparse <= 0:
            raise ValueError("top_k_sparse must be > 0")
        if top_k_fused <= 0:
            raise ValueError("top_k_fused must be > 0")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")
        if dense_weight < 0:
            raise ValueError("dense_weight must be >= 0")
        if sparse_weight < 0:
            raise ValueError("sparse_weight must be >= 0")
        if dense_weight + sparse_weight <= 0:
            raise ValueError("At least one weight must be > 0")

        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_fused = top_k_fused
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self._dense_retriever = get_dense_retriever()
        self._bm25_retriever = get_bm25_retriever()

    def search(
        self,
        query: str,
        k: int | None = None,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        top_k_fused: int | None = None,
        source: str | None = None,
    ) -> list[RetrievalResult]:
        """Search using hybrid RRF.

        Args:
            query: Search query text
            k: Override top_k_fused (uses instance default if None)
            top_k_dense: Override dense candidate count
            top_k_sparse: Override sparse candidate count
            top_k_fused: Override fused output count
            source: Optional source filename filter

        Returns:
            List of RetrievalResult ordered by RRF rank (1 = best)
        """
        if not query or not query.strip():
            return []

        # Use instance defaults if not overridden
        effective_top_k_dense = top_k_dense if top_k_dense is not None else self.top_k_dense
        effective_top_k_sparse = top_k_sparse if top_k_sparse is not None else self.top_k_sparse
        effective_top_k_fused = top_k_fused if top_k_fused is not None else self.top_k_fused

        # Retrieve from both retrievers
        dense_results = self._dense_retriever.search(
            query=query,
            k=effective_top_k_dense,
            source=source,
        )
        bm25_results = self._bm25_retriever.search(
            query=query,
            k=effective_top_k_sparse,
            source=source,
        )

        return self._fuse(dense_results, bm25_results, effective_top_k_fused)

    def search_with_timing(
        self,
        query: str,
        k: int | None = None,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        top_k_fused: int | None = None,
        source: str | None = None,
    ) -> tuple[list[RetrievalResult], dict[str, float]]:
        """Search using hybrid RRF and return per-stage latencies in milliseconds.

        Timings are strictly retrieval-only:
            dense_ms : DenseRetriever.search duration
            bm25_ms  : BM25Retriever.search duration
            rrf_ms   : fusion (rank-map + RRF scoring + ordering) duration
            total_ms : dense_ms + bm25_ms + rrf_ms

        LLM answer generation, adjacent-chunk expansion, and corpus ingestion
        are NOT included.
        """
        timing: dict[str, float] = {
            "dense_ms": 0.0,
            "bm25_ms": 0.0,
            "rrf_ms": 0.0,
            "total_ms": 0.0,
        }
        if not query or not query.strip():
            return [], timing

        # Use instance defaults if not overridden
        effective_top_k_dense = top_k_dense if top_k_dense is not None else self.top_k_dense
        effective_top_k_sparse = top_k_sparse if top_k_sparse is not None else self.top_k_sparse
        effective_top_k_fused = top_k_fused if top_k_fused is not None else self.top_k_fused

        t0 = perf_counter()
        dense_results = self._dense_retriever.search(
            query=query,
            k=effective_top_k_dense,
            source=source,
        )
        t1 = perf_counter()
        bm25_results = self._bm25_retriever.search(
            query=query,
            k=effective_top_k_sparse,
            source=source,
        )
        t2 = perf_counter()
        results = self._fuse(dense_results, bm25_results, effective_top_k_fused)
        t3 = perf_counter()

        timing["dense_ms"] = round((t1 - t0) * 1000, 3)
        timing["bm25_ms"] = round((t2 - t1) * 1000, 3)
        timing["rrf_ms"] = round((t3 - t2) * 1000, 3)
        timing["total_ms"] = round((t3 - t0) * 1000, 3)
        return results, timing

    def _fuse(
        self,
        dense_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        top_k_fused: int,
    ) -> list[RetrievalResult]:
        """Fuse dense + sparse results via weighted RRF. RRF math unchanged."""
        # Build rank maps for RRF
        dense_rank_map = {r.chunk_id: i + 1 for i, r in enumerate(dense_results)}
        sparse_rank_map = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}

        # Collect all unique chunk IDs
        all_chunk_ids = set()  # preserve insertion order below via dict
        for r in dense_results + bm25_results:
            all_chunk_ids.add(r.chunk_id)
        all_chunk_ids = list(all_chunk_ids)

        # Calculate RRF scores
        fused_scores = {}
        for chunk_id in all_chunk_ids:
            dense_rank = dense_rank_map.get(chunk_id)
            sparse_rank = sparse_rank_map.get(chunk_id)
            rrf_score = self._calculate_rrf_score(dense_rank, sparse_rank)
            fused_scores[chunk_id] = rrf_score

        # Sort by RRF score descending, then by chunk_id for deterministic tie-breaking
        ranked_chunk_ids = sorted(
            fused_scores.keys(),
            key=lambda cid: (-fused_scores[cid], cid)
        )

        # Build final results
        results = []
        # Create a lookup for result content/metadata
        result_lookup = {}
        for r in dense_results + bm25_results:
            if r.chunk_id not in result_lookup:
                result_lookup[r.chunk_id] = r

        for rank, chunk_id in enumerate(ranked_chunk_ids[:top_k_fused], 1):
            base_result = result_lookup.get(chunk_id)
            if base_result is None:
                continue
            dense_rank = dense_rank_map.get(chunk_id)
            sparse_rank = sparse_rank_map.get(chunk_id)

            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=fused_scores[chunk_id],
                rank=rank,
                source=base_result.source,
                content=base_result.content,
                metadata={
                    **base_result.metadata,
                    "rrf_score": fused_scores[chunk_id],
                    "dense_rank": dense_rank,
                    "sparse_rank": sparse_rank,
                    "in_dense": dense_rank is not None,
                    "in_bm25": sparse_rank is not None,
                },
            ))

        return results

    def _calculate_rrf_score(self, dense_rank: int | None, sparse_rank: int | None) -> float:
        """Calculate RRF score for a candidate."""
        score = 0.0
        if dense_rank is not None:
            score += self.dense_weight / (self.rrf_k + dense_rank)
        if sparse_rank is not None:
            score += self.sparse_weight / (self.rrf_k + sparse_rank)
        return score

    def search_dense_only(
        self,
        query: str,
        k: int | None = None,
        source: str | None = None,
    ) -> list:
        """Search using only dense retriever (for comparison)."""
        return self._dense_retriever.search(query=query, k=k or self.top_k_dense, source=source)

    def search_sparse_only(
        self,
        query: str,
        k: int | None = None,
        source: str | None = None,
    ) -> list:
        """Search using only BM25 retriever (for comparison)."""
        return self._bm25_retriever.search(query=query, k=k or self.top_k_sparse, source=source)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_HYBRID_RETRIEVER_INSTANCE: "HybridRRFRetriever | None" = None


def get_hybrid_retriever() -> "HybridRRFRetriever":
    """Get or create the singleton HybridRRFRetriever instance."""
    global _HYBRID_RETRIEVER_INSTANCE
    if _HYBRID_RETRIEVER_INSTANCE is None:
        _HYBRID_RETRIEVER_INSTANCE = HybridRRFRetriever()
    return _HYBRID_RETRIEVER_INSTANCE


def reset_hybrid_retriever() -> None:
    """Reset the singleton (mainly for testing)."""
    global _HYBRID_RETRIEVER_INSTANCE
    _HYBRID_RETRIEVER_INSTANCE = None