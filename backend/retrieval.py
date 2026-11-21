from time import perf_counter
from typing import Any
import re

from langchain_core.documents import Document

from backend.llm import generate_answer
from backend.retrieval_bm25 import get_bm25_retriever
from backend.retrieval_dense import DenseRetriever, get_dense_retriever, _to_retrieval_result
from backend.retrieval_hybrid import get_hybrid_retriever
from backend.schemas import SearchRequest, SearchResponse, SearchResult
from backend.vector_store import get_vector_store

# Defer retriever singletons to first use — avoids loading BM25 index
# (all chunk contents) at import time when only dense mode is needed.
_DenseRetrieverLazy = None
_BM25RetrieverLazy = None
_HybridRetrieverLazy = None


def _get_dense():
    global _DenseRetrieverLazy
    if _DenseRetrieverLazy is None:
        _DenseRetrieverLazy = get_dense_retriever()
    return _DenseRetrieverLazy


def _get_bm25():
    global _BM25RetrieverLazy
    if _BM25RetrieverLazy is None:
        _BM25RetrieverLazy = get_bm25_retriever()
    return _BM25RetrieverLazy


def _get_hybrid():
    global _HybridRetrieverLazy
    if _HybridRetrieverLazy is None:
        _HybridRetrieverLazy = get_hybrid_retriever()
    return _HybridRetrieverLazy


DenseRetrieverInstance = None  # lazy via _get_dense()
BM25RetrieverInstance = None   # lazy via _get_bm25()
HybridRetrieverInstance = None  # lazy via _get_hybrid()


def _page(metadata: dict[str, Any]) -> int | None:
    p = metadata.get("page")
    if isinstance(p, int):
        return p
    if isinstance(p, str) and p.isdigit():
        return int(p)
    return None


def _to_result(doc: Document, score: float) -> SearchResult:
    meta = dict(doc.metadata)
    return SearchResult(
        text=doc.page_content,
        source=str(meta.get("source", "unknown")),
        page=_page(meta),
        score=max(0.0, min(float(score), 1.0)),
        metadata=meta,
    )


_FALLBACK_STOPWORDS = {
    "what", "which", "where", "when", "why", "how", "does", "do", "is",
    "are", "the", "a", "an", "of", "in", "on", "to", "for", "and", "or",
    "this", "that", "document", "documents", "tell", "me", "about",
}


def _norm_term(token: str) -> str:
    """Light plural folding so "files" matches "file" (but "class" stays)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _extractive_answer(query: str, results: list[SearchResult]) -> str | None:
    """Build a grounded local answer when no remote LLM is configured.

    Requires meaningful lexical overlap between query terms and candidate
    sentences. Incidental word overlap (e.g., common stopwords or single
    matching terms) is not treated as sufficient evidence.
    """
    query_terms = {
        _norm_term(token) for token in re.findall(r"\b\w+\b", query.lower())
        if len(token) > 2 and token not in _FALLBACK_STOPWORDS
    }
    if not query_terms:
        return None

    candidates: list[tuple[int, int, SearchResult, str]] = []
    for result in results[:8]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", result.text.strip())
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            sentence_terms = {_norm_term(t) for t in re.findall(r"\b\w+\b", sentence.lower())}
            overlap = len(query_terms & sentence_terms)
            # Require at least 2 matching content terms. A single shared word
            # (e.g. "system") is incidental and is not grounded evidence; the
            # caller then falls back to listing the most relevant context.
            if overlap >= 2:
                candidates.append((overlap, len(sentence), result, sentence))

    if not candidates:
        return None

    best_overlap = max(item[0] for item in candidates)
    candidates = [item for item in candidates if item[0] == best_overlap]
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[tuple[SearchResult, str]] = []
    seen: set[str] = set()
    for _, _, result, sentence in candidates:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append((result, sentence))
        if len(selected) == 3:
            break

    lines = ["Based on the indexed documents:"]
    for index, (result, sentence) in enumerate(selected, 1):
        lines.append(f"{sentence} [{index}]")
    return "\n\n".join(lines)


def _answer(query: str, results: list[SearchResult]) -> str:
    if not results:
        return "No relevant context found in the indexed documents."
    llm_answer = generate_answer(query, results)
    if llm_answer:
        return llm_answer
    extractive_answer = _extractive_answer(query, results)
    if extractive_answer:
        return extractive_answer
    lines = ["Most relevant context:"]
    for i, r in enumerate(results[:3], 1):
        page = f", page {r.page}" if r.page else ""
        lines.append(f"[{i}] {r.source}{page}: {r.text[:700].strip()}")
    return "\n\n".join(lines)


# Adjacent chunk expansion - preserves existing behavior
def _fetch_adjacent_chunks(results: list[SearchResult]) -> list[SearchResult]:
    """Append the immediately following chunk for each retrieved chunk so
    multi-page lists/answers aren't cut off at a chunk boundary.
    """
    store = get_vector_store()
    collection = getattr(store, "_collection", None)
    if collection is None:
        return results

    expanded = list(results)
    seen_ids = {f"{r.metadata.get('document_id')}:{r.metadata.get('chunk')}" for r in results}

    for r in results:
        doc_id = r.metadata.get("document_id")
        chunk_num = r.metadata.get("chunk")
        if not doc_id or not isinstance(chunk_num, int):
            continue
        next_id = f"{doc_id}:{chunk_num + 1}"
        if next_id in seen_ids:
            continue
        try:
            fetched = collection.get(ids=[next_id], include=["documents", "metadatas"])
        except Exception:
            continue
        docs = fetched.get("documents") or []
        metas = fetched.get("metadatas") or []
        if not docs:
            continue
        next_text = docs[0]
        next_meta = metas[0] if metas else {}
        seen_ids.add(next_id)
        expanded.append(
            SearchResult(
                text=next_text,
                source=str(next_meta.get("source", r.source)),
                page=_page(next_meta),
                score=round(r.score * 0.9, 4),
                metadata=next_meta,
            )
        )
    return expanded


def run_search(
    request: SearchRequest,
    retrieval_results: list | None = None,
) -> SearchResponse:
    """Execute search using DenseRetriever, preserving all existing behavior.
    
    When retrieval_results is supplied (from /v1/ask), it uses those results
    directly and enforces the context limit (top_k) to prevent unbounded
    context growth. The limit aligns with the intended behavior for all
    retrieval modes: dense, sparse, hybrid, and hybrid_rerank.
    """
    query = " ".join(request.query.split())
    if not query:
        raise ValueError("Query cannot be empty.")

    start = perf_counter()

    # Use DenseRetriever for the core dense retrieval
    if retrieval_results is None:
        retriever = _get_dense()
        retrieval_results = retriever.search(
            query=query,
            k=request.top_k,
            source=request.source,
        )
    else:
        # Enforce context limit when retrieval_results is supplied
        # This ensures consistent behavior across all retrieval modes
        retrieval_results = retrieval_results[:request.top_k]

    # Convert RetrievalResult to SearchResult for backward compatibility
    pairs = []
    for rr in retrieval_results:
        # Create a minimal Document for conversion
        doc = Document(page_content=rr.content, metadata=rr.metadata)
        pairs.append((doc, rr.score))

    # Apply adjacent chunk expansion (existing behavior)
    results = [_to_result(doc, score) for doc, score in pairs]
    results = _fetch_adjacent_chunks(results)

    return SearchResponse(
        query=query,
        answer=_answer(query, results),
        results=results,
        latency_ms=round((perf_counter() - start) * 1000, 2),
    )


# ---------------------------------------------------------------------------
# New V1 API helper - returns canonical RetrievalResult
# ---------------------------------------------------------------------------

def run_search_dense(
    query: str,
    k: int = 10,
    source: str | None = None,
) -> list:
    """Direct dense search returning canonical RetrievalResult objects.

    Used by /v1/ask for structured trace.
    """
    retriever = _get_dense()
    return retriever.search(query=query, k=k, source=source)


def run_search_bm25(
    query: str,
    k: int = 10,
    source: str | None = None,
) -> list:
    """Direct BM25 search returning canonical RetrievalResult objects.

    Used by /v1/ask for structured trace.
    """
    retriever = _get_bm25()
    return retriever.search(query=query, k=k, source=source)


def run_search_hybrid(
    query: str,
    k: int = 10,
    top_k_dense: int = 10,
    top_k_sparse: int = 10,
    top_k_fused: int = 10,
    source: str | None = None,
) -> list:
    """Direct hybrid RRF search returning canonical RetrievalResult objects.

    Used by /v1/ask for structured trace in hybrid mode.
    """
    retriever = _get_hybrid()
    return retriever.search(
        query=query,
        k=k,
        top_k_dense=top_k_dense,
        top_k_sparse=top_k_sparse,
        top_k_fused=top_k_fused,
        source=source,
    )