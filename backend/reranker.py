import math
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from backend.settings import RERANK_ENABLED, RERANK_MODEL


@lru_cache(maxsize=1)
def get_reranker() -> Any:
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def reranker_available() -> bool:
    if not RERANK_ENABLED:
        return False
    try:
        get_reranker()
        return True
    except Exception:
        return False


def _sigmoid(logit: float) -> float:
    try:
        return round(1.0 / (1.0 + math.exp(-float(logit))), 4)
    except OverflowError:
        return 1.0 if logit > 0 else 0.0


def rerank(
    query: str,
    pairs: list[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    """Rescore bi-encoder candidates with a Cross-Encoder.

    Accepts [(Document, bi_encoder_score)] and returns the top_k best
    (Document, cross_encoder_score). If the reranker is unavailable or fails,
    it falls back to the incoming bi-encoder order without raising.
    """
    if not pairs:
        return []
    if top_k is None or top_k <= 0:
        top_k = len(pairs)

    try:
        if not reranker_available():
            return pairs[:top_k]
        texts = [doc.page_content for doc, _ in pairs]
        logits = get_reranker().predict([(query, text) for text in texts])
        scored = [(pairs[i], _sigmoid(logits[i])) for i in range(len(pairs))]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(doc, score) for (doc, _), score in scored[:top_k]]
    except Exception:
        return pairs[:top_k]