from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from backend.llm import generate_answer
from backend.schemas import SearchRequest, SearchResponse, SearchResult
from backend.vector_store import get_vector_store, search_similar


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

def _fetch_adjacent_chunks(results: list[SearchResult]) -> list[SearchResult]:
    """Append the immediately following chunk for each retrieved chunk so
    multi-page lists/answers aren't cut off at a chunk boundary.

    Lists that span pages get split across chunks; the second half often scores
    too low to surface on its own, so it is pulled in by its document_id/chunk
    sequence index instead.
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
            continue  # no such chunk, or lookup failed — move on
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
                score=round(r.score * 0.9, 4),  # slightly lower weighting
                metadata=next_meta,
            )
        )
    return expanded


def run_search(request: SearchRequest) -> SearchResponse:
    query = " ".join(request.query.split())
    if not query:
        raise ValueError("Query cannot be empty.")

    start = perf_counter()
    pairs = search_similar(query=query, k=request.top_k, source=request.source)

    # Deduplicate chunks based on normalized text content
    seen_text = set()
    unique_pairs = []
    for doc, score in pairs:
        normalized_text = doc.page_content.strip()
        if normalized_text in seen_text:
            continue
        seen_text.add(normalized_text)
        unique_pairs.append((doc, score))
        if len(unique_pairs) == request.top_k:
            break

    # Drop low-relevance noise
    filtered_pairs = [(doc, score) for doc, score in unique_pairs if score >= MIN_RELEVANCE_SCORE]
    results = [_to_result(doc, score) for doc, score in filtered_pairs]

    # Expand retrieved chunks to include continuation pages/chunks
    results = _fetch_adjacent_chunks(results)

    return SearchResponse(
        query=query,
        answer=_answer(query, results),
        results=results,
        latency_ms=round((perf_counter() - start) * 1000, 2),
    )