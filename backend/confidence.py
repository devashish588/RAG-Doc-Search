"""Confidence estimation and abstention guard (Phase 7).

Provides normalized retrieval signals and a composite confidence score
with configurable thresholds for safe abstention.
"""
import logging
from dataclasses import dataclass
from typing import Any, Optional

from backend.interfaces import RetrievalResult
from backend.settings import (
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    RRF_K,
    RRF_DENSE_WEIGHT,
    RRF_SPARSE_WEIGHT,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NormalizedSignals:
    """Normalized retrieval signals in [0, 1]."""
    dense: Optional[float] = None
    bm25: Optional[float] = None
    rrf: Optional[float] = None
    reranker: Optional[float] = None
    grounding: Optional[float] = None


@dataclass
class ConfidenceResult:
    """Composite confidence result."""
    overall_score: float
    level: str  # "high" | "medium" | "low"
    retrieval_confidence: float
    grounding_confidence: float
    abstention_flag: bool
    signals: NormalizedSignals


# ---------------------------------------------------------------------------
# Signal normalization (standalone functions for test compatibility)
# ---------------------------------------------------------------------------

def normalize_dense_score(score: float) -> float:
    """Normalize dense score to [0, 1] via (score + 1) / 2 clamped."""
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


# Alias for test compatibility
_normalize_dense = normalize_dense_score


def normalize_bm25_scores(scores: list[float]) -> list[float]:
    """Normalize BM25 scores to [0, 1] using min-max."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s <= min_s:
        return [1.0] * len(scores)
    return [max(0.0, min(1.0, (s - min_s) / (max_s - min_s))) for s in scores]


# Alias for test compatibility
_normalize_bm25 = normalize_bm25_scores


def normalize_rrf_score(score: float) -> float:
    """Normalize RRF score to [0, 1] using theoretical max."""
    max_possible = (RRF_DENSE_WEIGHT + RRF_SPARSE_WEIGHT) / (RRF_K + 1)
    if max_possible <= 0:
        return 0.0
    return max(0.0, min(1.0, score / max_possible))


# Alias for test compatibility
_normalize_rrf = normalize_rrf_score


def normalize_reranker_score(score: Optional[float]) -> Optional[float]:
    """Normalize reranker score - if None, return None."""
    if score is None:
        return None
    return max(0.0, min(1.0, score))


# Alias for test compatibility
_normalize_reranker = normalize_reranker_score


def _normalize_dense(dense_results: list[RetrievalResult]) -> Optional[float]:
    """Normalize dense retrieval score to [0, 1] using absolute mapping."""
    if not dense_results:
        return None
    top_score = dense_results[0].score
    return max(0.0, min(1.0, (top_score + 1.0) / 2.0))


def _normalize_bm25(bm25_results: list[RetrievalResult]) -> Optional[float]:
    """Normalize BM25 scores to [0, 1] using query-local min-max."""
    if not bm25_results:
        return None
    scores = [r.score for r in bm25_results]
    min_s = min(scores)
    max_s = max(scores)
    if max_s <= min_s:
        return 1.0
    top_score = bm25_results[0].score
    return max(0.0, min(1.0, (top_score - min_s) / (max_s - min_s)))


def _normalize_rrf(rrf_results: list[RetrievalResult]) -> Optional[float]:
    """Normalize RRF score to [0, 1] using theoretical max."""
    if not rrf_results:
        return None
    top_score = rrf_results[0].score
    max_possible = (RRF_DENSE_WEIGHT + RRF_SPARSE_WEIGHT) / (RRF_K + 1)
    if max_possible <= 0:
        return 0.0
    return max(0.0, min(1.0, top_score / max_possible))


def _normalize_reranker(reranker_results: list[RetrievalResult]) -> Optional[float]:
    """Normalize reranker score to [0, 1] using query-local min-max."""
    if not reranker_results:
        return None
    scores = [r.score for r in reranker_results]
    min_s = min(scores)
    max_s = max(scores)
    if max_s <= min_s:
        return 1.0
    top_score = reranker_results[0].score
    return max(0.0, min(1.0, (top_score - min_s) / (max_s - min_s)))


def _normalize_grounding(grounding_ratio: float) -> float:
    """Grounding ratio is already in [0, 1]."""
    return max(0.0, min(1.0, grounding_ratio))


# ---------------------------------------------------------------------------
# Confidence Estimator
# ---------------------------------------------------------------------------

class ConfidenceEstimator:
    """Computes composite confidence from normalized retrieval signals and grounding."""

    def __init__(
        self,
        # Weights for retrieval component
        dense_weight: float = 0.4,
        bm25_weight: float = 0.3,
        rrf_weight: float = 0.2,
        reranker_weight: float = 0.1,
        # Weight for grounding in overall composite
        grounding_weight: float = 0.5,
        retrieval_weight: float = 0.5,
        # Level thresholds
        high_threshold: float = CONFIDENCE_HIGH_THRESHOLD,
        medium_threshold: float = CONFIDENCE_MEDIUM_THRESHOLD,
        abstention_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_weight = rrf_weight
        self.reranker_weight = reranker_weight
        self.grounding_weight = grounding_weight
        self.retrieval_weight = retrieval_weight
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.abstention_threshold = abstention_threshold

    def calculate_confidence(
        self,
        retrieval_trace: dict[str, list[RetrievalResult]],
        grounding_ratio: float,
        total_claims: int,
        supported_claims: int,
        retrieval_mode: str,
    ) -> dict[str, Any]:
        """Calculate confidence (test-compatible API)."""
        dense_results = retrieval_trace.get("dense", [])
        bm25_results = retrieval_trace.get("bm25", [])
        rrf_results = retrieval_trace.get("rrf", [])
        reranker_results = retrieval_trace.get("reranker", [])

        # Normalize all signals
        dense_signal = _normalize_dense(dense_results)
        bm25_signal = _normalize_bm25(bm25_results)
        rrf_signal = _normalize_rrf(rrf_results)
        reranker_signal = _normalize_reranker(reranker_results)
        grounding_signal = _normalize_grounding(grounding_ratio)

        # Compute retrieval_confidence: weighted average of AVAILABLE retrieval signals
        retrieval_signals = {}
        if dense_signal is not None:
            retrieval_signals['dense'] = dense_signal
        if bm25_signal is not None:
            retrieval_signals['bm25'] = bm25_signal
        if rrf_signal is not None:
            retrieval_signals['rrf'] = rrf_signal
        if reranker_signal is not None:
            retrieval_signals['reranker'] = reranker_signal

        if retrieval_signals:
            total_weight = 0.0
            weighted_sum = 0.0
            for name, value in retrieval_signals.items():
                weight = getattr(self, f'{name}_weight', 0.0)
                weighted_sum += weight * value
                total_weight += weight
            retrieval_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            retrieval_confidence = 0.0

        retrieval_confidence = round(max(0.0, min(1.0, retrieval_confidence)), 4)
        grounding_confidence = round(grounding_signal, 4)

        # Overall composite: retrieval + grounding
        if retrieval_signals:
            overall_score = (
                self.retrieval_weight * retrieval_confidence +
                self.grounding_weight * grounding_confidence
            ) / (self.retrieval_weight + self.grounding_weight)
        else:
            overall_score = grounding_confidence

        overall_score = round(max(0.0, min(1.0, overall_score)), 4)

        # Determine confidence level
        if overall_score >= self.high_threshold:
            level = "high"
        elif overall_score >= self.medium_threshold:
            level = "medium"
        else:
            level = "low"

        # Abstention guard
        abstention_flag = self._should_abstain(
            overall_score=overall_score,
            retrieval_confidence=retrieval_confidence,
            grounding_confidence=grounding_confidence,
            retrieval_mode=retrieval_mode,
            total_claims=total_claims,
            supported_claims=supported_claims,
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_results=rrf_results,
            reranker_results=reranker_results,
        )

        return {
            "overall_score": overall_score,
            "level": level,
            "retrieval_confidence": retrieval_confidence,
            "grounding_confidence": grounding_confidence,
            "abstention_flag": abstention_flag,
            "signals": {
                "dense_signal": dense_signal,
                "bm25_signal": bm25_signal,
                "rrf_signal": rrf_signal,
                "reranker_signal": reranker_signal,
                "grounding_signal": grounding_signal,
            }
        }

    def estimate(
        self,
        dense_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        rrf_results: list[RetrievalResult],
        reranker_results: list[RetrievalResult],
        grounding_ratio: float,
        retrieval_mode: str,
    ) -> ConfidenceResult:
        """Compute composite confidence from all signals (new API)."""
        # Normalize all signals
        signals = NormalizedSignals(
            dense=_normalize_dense(dense_results),
            bm25=_normalize_bm25(bm25_results),
            rrf=_normalize_rrf(rrf_results),
            reranker=_normalize_reranker(reranker_results),
            grounding=_normalize_grounding(grounding_ratio),
        )

        retrieval_signals = {}
        if signals.dense is not None:
            retrieval_signals['dense'] = signals.dense
        if signals.bm25 is not None:
            retrieval_signals['bm25'] = signals.bm25
        if signals.rrf is not None:
            retrieval_signals['rrf'] = signals.rrf
        if signals.reranker is not None:
            retrieval_signals['reranker'] = signals.reranker

        if retrieval_signals:
            total_weight = 0.0
            weighted_sum = 0.0
            for name, value in retrieval_signals.items():
                weight = getattr(self, f'{name}_weight', 0.0)
                weighted_sum += weight * value
                total_weight += weight
            retrieval_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            retrieval_confidence = 0.0

        retrieval_confidence = round(max(0.0, min(1.0, retrieval_confidence)), 4)
        grounding_confidence = signals.grounding

        if retrieval_signals:
            overall_score = (
                self.retrieval_weight * retrieval_confidence +
                self.grounding_weight * grounding_confidence
            ) / (self.retrieval_weight + self.grounding_weight)
        else:
            overall_score = grounding_confidence

        overall_score = round(max(0.0, min(1.0, overall_score)), 4)

        if overall_score >= self.high_threshold:
            level = "high"
        elif overall_score >= self.medium_threshold:
            level = "medium"
        else:
            level = "low"

        abstention_flag = self._should_abstain(
            overall_score=overall_score,
            retrieval_confidence=retrieval_confidence,
            grounding_confidence=grounding_confidence,
            retrieval_mode=retrieval_mode,
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_results=rrf_results,
            reranker_results=reranker_results,
        )

        return ConfidenceResult(
            overall_score=overall_score,
            level=level,
            retrieval_confidence=retrieval_confidence,
            grounding_confidence=grounding_confidence,
            abstention_flag=abstention_flag,
            signals=signals,
        )

    def _should_abstain(
        self,
        overall_score: float,
        retrieval_confidence: float,
        grounding_confidence: float,
        retrieval_mode: str,
        total_claims: int = 0,
        supported_claims: int = 0,
        dense_results: list[RetrievalResult] = None,
        bm25_results: list[RetrievalResult] = None,
        rrf_results: list[RetrievalResult] = None,
        reranker_results: list[RetrievalResult] = None,
    ) -> bool:
        """Determine if the system should abstain from answering."""
        dense_results = dense_results or []
        bm25_results = bm25_results or []
        rrf_results = rrf_results or []
        reranker_results = reranker_results or []

        # 1. Score threshold
        if overall_score < self.abstention_threshold:
            return True

        # 2. No evidence at all
        has_any_results = bool(dense_results or bm25_results or rrf_results or reranker_results)
        if not has_any_results:
            return True

        # 3. Very weak retrieval + low grounding
        if retrieval_confidence < 0.1 and grounding_confidence < 0.3:
            return True

        # 4. Low grounding
        if grounding_confidence < 0.3:
            return True

        # 5. Reranker mode but reranker failed/fallback
        if retrieval_mode == "hybrid_rerank" and reranker_results:
            for r in reranker_results:
                if r.metadata.get("reranker_status") == "failed":
                    return True

        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ESTIMATOR_INSTANCE: Optional[ConfidenceEstimator] = None


def get_confidence_estimator() -> ConfidenceEstimator:
    """Get or create the singleton ConfidenceEstimator instance."""
    global _ESTIMATOR_INSTANCE
    if _ESTIMATOR_INSTANCE is None:
        _ESTIMATOR_INSTANCE = ConfidenceEstimator()
    return _ESTIMATOR_INSTANCE


def reset_confidence_estimator() -> None:
    """Reset the singleton (mainly for testing)."""
    global _ESTIMATOR_INSTANCE
    _ESTIMATOR_INSTANCE = None