from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from backend.schemas import SearchRequest, SearchResponse, SearchResult
from backend.vector_store import search_similar


def _page_from_metadata(metadata: dict[str, Any]) -> int | None:
    page = metadata.get("page")
    if isinstance(page, int):
        return page
    if isinstance(page, str) and page.isdigit():
        return int(page)
    return None


def _result_from_document(document: Document, score: float) -> SearchResult:
    metadata = dict(document.metadata)
    return SearchResult(
        text=document.page_content,
        source=str(metadata.get("source", "unknown")),
        page=_page_from_metadata(metadata),
        score=score,
        metadata=metadata,
    )


def _build_answer(results: list[SearchResult]) -> str:
    if not results:
        return "I could not find relevant context in the indexed documents."

    lines = ["Most relevant context:"]
    for index, result in enumerate(results[:3], start=1):
        page = f", page {result.page}" if result.page else ""
        snippet = result.text[:700].strip()
        lines.append(f"[{index}] {result.source}{page}: {snippet}")
    return "\n\n".join(lines)


def run_search(request: SearchRequest) -> SearchResponse:
    query = " ".join(request.query.split())
    if not query:
        raise ValueError("Query cannot be empty.")

    start = perf_counter()
    pairs = search_similar(query=query, k=request.top_k, source=request.source)
    results = [_result_from_document(document, score) for document, score in pairs]
    latency_ms = (perf_counter() - start) * 1000

    return SearchResponse(
        query=query,
        answer=_build_answer(results),
        results=results,
        latency_ms=round(latency_ms, 2),
    )

