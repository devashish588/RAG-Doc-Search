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
    """Check BM25 index accessibility."""
    try:
        if not BM25_INDEX_DIR.exists():
            return {"status": "error", "message": "BM25 index directory missing"}
        index_file = BM25_INDEX_DIR / "bm25_index.pkl"
        if not index_file.exists():
            # Try alternate naming
            files = list(BM25_INDEX_DIR.glob("*.pkl"))
            if not files:
                return {"status": "error", "message": "No BM25 index file found"}
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
    """Check that Chroma and BM25 index directories exist."""
    issues = []
    if not CHROMA_DIR.exists():
        issues.append("Chroma directory missing")
    if not BM25_INDEX_DIR.exists():
        issues.append("BM25 directory missing")
    if not UPLOAD_DIR.exists():
        issues.append("Upload directory missing")
    return {"status": "ok" if not issues else "warning", "issues": issues}


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
