from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from backend.llm import generate_answer
from backend.reranker import rerank
from backend.schemas import SearchRequest, SearchResponse, SearchResult
from backend.settings import RERANK_CANDIDATES
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

MIN_RELEVANCE_SCORE = 0.35

# retrieval.py
def run_search(request: SearchRequest) -> SearchResponse:
    query = " ".join(request.query.split())
    if not query:
        raise ValueError("Query cannot be empty.")
    start = perf_counter()
    pairs = search_similar(query=query, k=request.top_k, source=request.source)
    
    # Deduplicate results based on page content
    seen_texts = set()
    unique_results = []
    for doc, score in pairs:
        clean_text = doc.page_content.strip()
        if clean_text not in seen_texts:
            seen_texts.add(clean_text)
            unique_results.append(_to_result(doc, score))
            
    return SearchResponse(
        query=query,
        answer=_answer(query, unique_results),
        results=unique_results,
        latency_ms=round((perf_counter() - start) * 1000, 2),
    )
