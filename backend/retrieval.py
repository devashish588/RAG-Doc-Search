from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from backend.llm import generate_answer
from backend.retrieval_bm25 import get_bm25_retriever
from backend.retrieval_dense import DenseRetriever, get_dense_retriever, _to_retrieval_result
from backend.schemas import SearchRequest, SearchResponse, SearchResult
from backend.vector_store import get_vector_store

DenseRetrieverInstance = get_dense_retriever()
BM25RetrieverInstance = get_bm25_retriever()


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
        score=score,
        metadata=meta,
    )


def _answer(query: str, results: list[SearchResult]) -> str:
    if not results:
        return "No relevant context found in the indexed documents."
    llm_answer = generate_answer(query, results)
    if llm_answer:
        return llm_answer
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


def run_search(request: SearchRequest) -> SearchResponse:
    """Execute search using DenseRetriever, preserving all existing behavior."""
    query = " ".join(request.query.split())
    if not query:
        raise ValueError("Query cannot be empty.")

    start = perf_counter()

    # Use DenseRetriever for the core dense retrieval
    retriever = DenseRetrieverInstance
    retrieval_results = retriever.search(
        query=query,
        k=request.top_k,
        source=request.source,
    )

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
    retriever = get_dense_retriever()
    return retriever.search(query=query, k=k, source=source)


def run_search_bm25(
    query: str,
    k: int = 10,
    source: str | None = None,
) -> list:
    """Direct BM25 search returning canonical RetrievalResult objects.

    Used by /v1/ask for structured trace.
    """
    retriever = get_bm25_retriever()
    return retriever.search(query=query, k=k, source=source)