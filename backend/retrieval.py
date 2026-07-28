from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from backend.schemas import SearchRequest, SearchResponse, SearchResult
from backend.vector_store import search_similar


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


def _answer(results: list[SearchResult]) -> str:
    if not results:
        return "No relevant context found in the indexed documents."
    lines = ["Most relevant context:"]
    for i, r in enumerate(results[:3], 1):
        page = f", page {r.page}" if r.page else ""
        lines.append(f"[{i}] {r.source}{page}: {r.text[:700].strip()}")
    return "\n\n".join(lines)


def run_search(request: SearchRequest) -> SearchResponse:
    query = " ".join(request.query.split())
    if not query:
        raise ValueError("Query cannot be empty.")
    start = perf_counter()
    pairs = search_similar(query=query, k=request.top_k, source=request.source)
    results = [_to_result(doc, score) for doc, score in pairs]
    return SearchResponse(
        query=query,
        answer=_answer(results),
        results=results,
        latency_ms=round((perf_counter() - start) * 1000, 2),
    )
