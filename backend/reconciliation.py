"""Index reconciliation and health checking."""
import logging
from collections import Counter
from typing import Any

from backend.ingestion import list_document_statuses
from backend.vector_store import get_vector_store

log = logging.getLogger(__name__)


def reconcile_index() -> dict[str, Any]:
    """Check consistency between document registry and vector store.

    Detects:
    - missing expected chunks (documents marked complete but chunks missing in index)
    - orphan vector chunks (chunks in index without corresponding document record)
    - duplicate canonical chunk IDs
    - document/chunk mismatches

    Returns a structured health report.
    """
    store = get_vector_store()
    collection = getattr(store, "_collection", None)

    if collection is None:
        return {
            "status": "unavailable",
            "reason": "Chroma collection not accessible",
            "checks": {},
        }

    # Get all indexed chunks from Chroma
    try:
        all_data = collection.get(include=["metadatas"])
    except Exception as exc:
        log.exception("Failed to fetch collection data")
        return {
            "status": "error",
            "reason": f"Collection query failed: {exc}",
            "checks": {},
        }

    metadatas = all_data.get("metadatas", [])
    ids = all_data.get("ids", [])

    # Build index-side view
    indexed_by_doc: dict[str, list[dict[str, Any]]] = {}
    chunk_id_counts: Counter[str] = Counter()

    for meta, cid in zip(metadatas, ids):
        if not meta:
            continue
        doc_id = meta.get("document_id")
        if doc_id:
            indexed_by_doc.setdefault(doc_id, []).append({"chunk_id": cid, "metadata": meta})
        chunk_id_counts[cid] += 1

    # Get registry view
    registry = list_document_statuses()
    registry_by_id = {r["document_id"]: r for r in registry}

    issues = []

    # 1. Check for missing expected chunks (complete docs with no indexed chunks)
    for doc_id, record in registry_by_id.items():
        if record["status"] == "complete":
            indexed_chunks = indexed_by_doc.get(doc_id, [])
            if not indexed_chunks:
                issues.append({
                    "type": "missing_chunks",
                    "severity": "high",
                    "document_id": doc_id,
                    "filename": record["filename"],
                    "expected_chunks": record.get("chunks_indexed", 0),
                    "actual_chunks": 0,
                })
            elif len(indexed_chunks) != record.get("chunks_indexed", 0):
                issues.append({
                    "type": "chunk_count_mismatch",
                    "severity": "medium",
                    "document_id": doc_id,
                    "filename": record["filename"],
                    "expected_chunks": record.get("chunks_indexed", 0),
                    "actual_chunks": len(indexed_chunks),
                })

    # 2. Check for orphan vector chunks (chunks without registry record)
    for doc_id, chunks in indexed_by_doc.items():
        if doc_id not in registry_by_id:
            issues.append({
                "type": "orphan_chunks",
                "severity": "medium",
                "document_id": doc_id,
                "orphan_chunk_count": len(chunks),
                "sample_chunk_ids": [c["chunk_id"] for c in chunks[:5]],
            })

    # 3. Check for duplicate canonical chunk IDs
    duplicates = {cid: count for cid, count in chunk_id_counts.items() if count > 1}
    if duplicates:
        issues.append({
            "type": "duplicate_chunk_ids",
            "severity": "high",
            "duplicates": duplicates,
        })

    # 4. Check for document/chunk ID format consistency
    malformed_ids = []
    for cid in ids:
        parts = cid.rsplit(":", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            malformed_ids.append(cid)
    if malformed_ids:
        issues.append({
            "type": "malformed_chunk_ids",
            "severity": "low",
            "count": len(malformed_ids),
            "sample": malformed_ids[:10],
        })

    # Summary
    total_indexed = len(ids)
    total_registry = len(registry)
    complete_registry = sum(1 for r in registry if r["status"] == "complete")

    status = "healthy"
    if any(i["severity"] == "high" for i in issues):
        status = "degraded"
    elif issues:
        status = "warning"

    return {
        "status": status,
        "summary": {
            "total_indexed_chunks": total_indexed,
            "total_documents_registry": total_registry,
            "complete_documents": complete_registry,
            "documents_with_chunks": len(indexed_by_doc),
        },
        "issues": issues,
        "checks": {
            "missing_chunks": sum(1 for i in issues if i["type"] == "missing_chunks"),
            "orphan_chunks": sum(1 for i in issues if i["type"] == "orphan_chunks"),
            "duplicate_chunk_ids": len(duplicates),
            "malformed_chunk_ids": len(malformed_ids),
        },
    }


def health_check() -> dict[str, Any]:
    """Lightweight health check for monitoring endpoints."""
    report = reconcile_index()
    return {
        "status": report["status"],
        "summary": report["summary"],
        "issue_count": len(report["issues"]),
    }