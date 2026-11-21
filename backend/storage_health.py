"""Storage health checks for Chroma, BM25, and disk (Phase 9)."""
import logging
import shutil
from pathlib import Path
from typing import Any

from backend.settings import BM25_INDEX_DIR, CHROMA_DIR, DATA_DIR, ensure_runtime_dirs

log = logging.getLogger(__name__)

_MIN_DISK_MB = 100  # minimum free disk space


def check_chroma_health() -> dict[str, Any]:
    """Check ChromaDB accessibility."""
    try:
        from backend.vector_store import get_vector_store
        store = get_vector_store()
        collection = getattr(store, "_collection", None)
        if collection is None:
            return {"status": "error", "message": "No collection attribute"}
        # Lightweight count query
        count = collection.count()
        return {"status": "ok", "chunks": count}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:200]}


def check_bm25_health() -> dict[str, Any]:
    """Check BM25 index accessibility.

    A missing index file on a fresh deploy (no documents ingested yet) is a
    valid empty state, not an error. Real drift between Chroma and BM25 is
    reported by check_index_consistency().
    """
    try:
        ensure_runtime_dirs()
        index_file = BM25_INDEX_DIR / "bm25_index.pkl"
        if not index_file.exists() and not list(BM25_INDEX_DIR.glob("*.pkl")):
            return {"status": "ok", "empty": True}
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:200]}


def check_disk_health() -> dict[str, Any]:
    """Check disk space availability."""
    try:
        ensure_runtime_dirs()
        usage = shutil.disk_usage(str(DATA_DIR))
        free_mb = usage.free // (1024 * 1024)
        if free_mb < _MIN_DISK_MB:
            return {"status": "warning", "message": f"Low disk: {free_mb}MB free", "free_mb": free_mb}
        return {"status": "ok", "free_mb": free_mb}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:200]}


def check_index_consistency() -> dict[str, Any]:
    """Compare authoritative Chroma chunk IDs with the persisted BM25 IDs."""
    ensure_runtime_dirs()
    issues: list[dict[str, Any]] = []
    if not CHROMA_DIR.exists():
        issues.append({"type": "chroma_directory_missing"})
    if not BM25_INDEX_DIR.exists():
        issues.append({"type": "bm25_directory_missing"})
    if not UPLOAD_DIR.exists():
        issues.append({"type": "upload_directory_missing"})

    if issues:
        return {"status": "warning", "issues": issues}

    try:
        from backend.bm25_index import get_bm25_index
        from backend.vector_store import get_vector_store

        collection = getattr(get_vector_store(), "_collection", None)
        if collection is None:
            return {
                "status": "error",
                "issues": [{"type": "chroma_collection_unavailable"}],
            }

        chroma_ids = set(collection.get(include=[]).get("ids", []))
        bm25_index = get_bm25_index()
        bm25_ids = bm25_index.get_chunk_ids()
        missing_from_bm25 = chroma_ids - bm25_ids
        orphaned_in_bm25 = bm25_ids - chroma_ids
        if missing_from_bm25 or orphaned_in_bm25:
            return {
                "status": "error",
                "issues": [{
                    "type": "chunk_id_mismatch",
                    "chroma_chunks": len(chroma_ids),
                    "bm25_chunks": len(bm25_ids),
                    "missing_from_bm25": len(missing_from_bm25),
                    "orphaned_in_bm25": len(orphaned_in_bm25),
                    "sample_missing_from_bm25": sorted(missing_from_bm25)[:5],
                    "sample_orphaned_in_bm25": sorted(orphaned_in_bm25)[:5],
                }],
            }
        return {
            "status": "ok",
            "chroma_chunks": len(chroma_ids),
            "bm25_chunks": len(bm25_ids),
        }
    except Exception as exc:
        log.exception("Failed to compare Chroma and BM25 indexes")
        return {"status": "error", "issues": [{"type": "comparison_failed", "message": str(exc)[:200]}]}


# lazy import to avoid circular
from backend.settings import UPLOAD_DIR  # noqa: E402


def check_dependencies() -> dict[str, Any]:
    """Check critical runtime dependencies are importable."""
    from backend.dependency_check import check_runtime_dependencies
    report = check_runtime_dependencies()
    if report.status == "failed":
        return {"status": "error", "missing": report.missing_critical}
    if report.status == "degraded":
        return {"status": "warning", "missing_optional": report.missing_optional}
    return {"status": "ok"}


def readiness_report() -> dict[str, Any]:
    """Run all health checks and return a combined report."""
    checks = {
        "dependencies": check_dependencies(),
        "chroma": check_chroma_health(),
        "bm25": check_bm25_health(),
        "index_consistency": check_index_consistency(),
        "disk": check_disk_health(),
    }
    all_ok = all(c["status"] == "ok" for c in checks.values())
    any_error = any(c["status"] == "error" for c in checks.values())
    if any_error:
        status = "not_ready"
    elif all_ok:
        status = "ready"
    else:
        status = "degraded"
    return {"status": status, "checks": checks}
