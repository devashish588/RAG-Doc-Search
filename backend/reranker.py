"""Cross-encoder reranking (Phase 6).

Consumes the fused Hybrid RRF candidate pool and re-scores it with a lightweight
CPU cross-encoder (FlashRank / ONNX). It does NOT retrieve; it only reorders an
existing candidate list. RRF provenance (original rank/score, dense/sparse flags)
is preserved on every reranked result.

Design constraints (Render free tier, ~512 MB, CPU):
- Model is loaded lazily and reused between requests (singleton by default).
- First-call (cold) initialization time is recorded separately.
- Failure (model load / scoring) degrades gracefully to the RRF candidate pool
  with a `reranker_status="failed"` marker — it never crashes the request.
"""
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from backend.interfaces import BaseReranker, RetrievalResult
from backend.settings import (
    RERANKER_CANDIDATE_K,
    RERANKER_MODEL,
    RERANKER_TOP_K,
)

log = logging.getLogger(__name__)


class _SimpleRequest:
    """Minimal query/passages container used when a scorer is injected (tests)."""

    def __init__(self, query: str, passages: list[dict[str, Any]]):
        self.query = query
        self.passages = passages


class CrossEncoderReranker(BaseReranker):
    """Rerank RRF candidates using a lightweight cross-encoder.

    Args:
        model_name: FlashRank model id (e.g. ms-marco-MiniLM-L-12-v2,
            ms-marco-TinyBERT-L-2-v2).
        candidate_k: safety cap on the candidate pool size.
        top_k: number of final reranked candidates to return.
        cache_dir: optional on-disk cache for the ONNX model.
        _ranker: inject a pre-built scorer (used by tests to avoid network).
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        candidate_k: int = RERANKER_CANDIDATE_K,
        top_k: int = RERANKER_TOP_K,
        cache_dir: str | None = None,
        _ranker: Any | None = None,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_k > candidate_k:
            raise ValueError(f"top_k ({top_k}) must be <= candidate_k ({candidate_k})")

        self.model_name = model_name
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "flashrank")

        # Runtime state
        self.available = False
        self.status = "uninitialized"
        self.cold_init_ms: float | None = None
        self.last_rerank_ms: float | None = None
        self._ranker = _ranker
        self._Ranker = None
        self._RerankRequest = None

        if _ranker is not None:
            # Pre-built scorer injected (tests / explicit reuse)
            self._ranker = _ranker
            self._RerankRequest = _SimpleRequest
            self.available = True
            self.status = "ready"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load the cross-encoder on first use. Safe to call repeatedly."""
        if self._ranker is not None:
            return
        try:
            t0 = perf_counter()
            # Imported here so a missing/optional dependency never breaks import
            # of the rest of the backend.
            from flashrank import Ranker, RerankRequest  # type: ignore

            self._Ranker = Ranker
            self._RerankRequest = RerankRequest
            self._ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
            self.cold_init_ms = round((perf_counter() - t0) * 1000, 3)
            self.available = True
            self.status = "ready"
            log.info("Reranker model '%s' loaded in %.1f ms", self.model_name, self.cold_init_ms)
        except Exception as exc:  # pragma: no cover - depends on environment
            self.available = False
            self.status = f"failed: {exc}"
            self._ranker = None
            log.exception("Reranker model '%s' failed to load", self.model_name)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        results, _ = self.rerank_with_timing(query, candidates, top_k)
        return results

    def rerank_with_timing(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        source: str | None = None,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """Rerank candidates and return (results, timing/status dict).

        timing dict keys: reranker_ms, status ("ok"|"fallback"), cold_init_ms.
        """
        effective_top_k = top_k if top_k is not None else self.top_k

        # Safety cap on candidate pool
        pool = candidates
        if self.candidate_k and len(pool) > self.candidate_k:
            pool = pool[: self.candidate_k]

        timing: dict[str, Any] = {
            "reranker_ms": 0.0,
            "status": "ok",
            "cold_init_ms": self.cold_init_ms,
        }

        # Empty input -> empty output, no scoring.
        if not pool:
            return [], timing

        # Ensure model is loaded (cold init happens on first call).
        if self._ranker is None:
            self._ensure_loaded()

        # Graceful fallback: if the model is unavailable, return the RRF pool
        # (truncated to top_k) marked as failed so callers know reranking
        # did not occur.
        if not self.available or self._ranker is None:
            timing["status"] = "fallback"
            return self._fallback(pool, effective_top_k), timing

        t0 = perf_counter()
        try:
            passages = [{"id": c.chunk_id, "text": c.content} for c in pool]
            response = self._ranker.rerank(
                self._RerankRequest(query=query, passages=passages)
            )
            score_by_id = {r["id"]: float(r["score"]) for r in response}
        except Exception as exc:  # pragma: no cover - depends on environment
            log.exception("Reranker scoring failed: %s", exc)
            timing["status"] = "fallback"
            timing["reranker_ms"] = round((perf_counter() - t0) * 1000, 3)
            return self._fallback(pool, effective_top_k), timing

        # Order candidates by cross-encoder score (desc). Ties broken by the
        # candidate's original RRF rank for determinism.
        ranked = sorted(
            pool,
            key=lambda c: (score_by_id.get(c.chunk_id, float("-inf")), -c.rank),
            reverse=True,
        )

        out: list[RetrievalResult] = []
        for final_rank, c in enumerate(ranked[:effective_top_k], 1):
            md = dict(c.metadata)
            md["reranker_score"] = round(score_by_id.get(c.chunk_id, 0.0), 6)
            md["original_rrf_rank"] = c.rank
            md["original_rrf_score"] = c.score
            md["final_rank"] = final_rank
            md["reranker_status"] = "ok"
            # dense_rank / sparse_rank / in_dense / in_bm25 / rrf_score are
            # already present in c.metadata (carried from the RRF stage).
            out.append(
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    score=md["reranker_score"],
                    rank=final_rank,
                    source=c.source,
                    content=c.content,
                    metadata=md,
                )
            )

        timing["reranker_ms"] = round((perf_counter() - t0) * 1000, 3)
        timing["status"] = "ok"
        timing["cold_init_ms"] = self.cold_init_ms
        self.last_rerank_ms = timing["reranker_ms"]
        return out, timing

    def _fallback(
        self, pool: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        out = []
        for i, c in enumerate(pool[:top_k], 1):
            md = dict(c.metadata)
            md["reranker_status"] = "failed"
            md["original_rrf_rank"] = c.rank
            md["original_rrf_score"] = c.score
            md["final_rank"] = i
            out.append(
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    score=c.score,
                    rank=i,
                    source=c.source,
                    content=c.content,
                    metadata=md,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Singleton factory (default model / config)
# ---------------------------------------------------------------------------

_RERANKER_INSTANCE: "CrossEncoderReranker | None" = None


def get_reranker() -> "CrossEncoderReranker":
    """Get or create the singleton reranker using the configured model."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        _RERANKER_INSTANCE = CrossEncoderReranker()
    return _RERANKER_INSTANCE


def reset_reranker() -> None:
    """Reset the singleton (mainly for testing)."""
    global _RERANKER_INSTANCE
    _RERANKER_INSTANCE = None
