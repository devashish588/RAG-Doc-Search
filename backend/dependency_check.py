"""Runtime dependency validation.

Validates that critical production dependencies are installed and functional.
Called at startup and by /readyz to ensure the application cannot silently
degrade when packages are missing.
"""
import importlib
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Packages required for production operation.
# key = import name, value = PyPI package name (for install instructions).
_CRITICAL_DEPS: dict[str, str] = {
    "chromadb": "chromadb",
    "fastembed": "fastembed",
    "langchain_chroma": "langchain-chroma",
    "numpy": "numpy",
    "rank_bm25": "rank-bm25",
    "sklearn": "scikit-learn",
}

_OPTIONAL_DEPS: dict[str, str] = {
    "flashrank": "flashrank",
    "prometheus_client": "prometheus-client",
}


@dataclass
class DepStatus:
    installed: bool
    version: str | None = None
    error: str | None = None


@dataclass
class DependencyReport:
    status: str  # "ok" | "degraded" | "failed"
    python_version: str = field(default_factory=lambda: sys.version)
    dependencies: dict[str, DepStatus] = field(default_factory=dict)
    missing_critical: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)


def _check_package(import_name: str) -> DepStatus:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", None))
        if isinstance(version, tuple):
            version = ".".join(str(v) for v in version)
        return DepStatus(installed=True, version=str(version) if version else None)
    except ImportError as exc:
        return DepStatus(installed=False, error=str(exc))


def check_runtime_dependencies() -> DependencyReport:
    """Validate all critical and optional runtime dependencies."""
    report = DependencyReport(status="ok")

    for import_name, pkg_name in _CRITICAL_DEPS.items():
        ds = _check_package(import_name)
        report.dependencies[pkg_name] = ds
        if not ds.installed:
            report.missing_critical.append(pkg_name)

    for import_name, pkg_name in _OPTIONAL_DEPS.items():
        ds = _check_package(import_name)
        report.dependencies[pkg_name] = ds
        if not ds.installed:
            report.missing_optional.append(pkg_name)

    if report.missing_critical:
        report.status = "failed"
        log.error(
            "CRITICAL: Missing runtime dependencies: %s. "
            "Install using: pip install -r requirements.txt",
            ", ".join(report.missing_critical),
        )
    elif report.missing_optional:
        report.status = "degraded"
        log.warning("Optional dependencies missing: %s", ", ".join(report.missing_optional))
    else:
        report.status = "ok"

    return report


def check_embedding_backend() -> dict[str, Any]:
    """Check if the configured embedding backend is functional."""
    from backend.settings import EMBEDDING_BACKEND

    if EMBEDDING_BACKEND in {"hash", "hashing"}:
        return {"status": "ok", "backend": "hashing", "reason": "configured"}

    ds = _check_package("fastembed")
    if ds.installed:
        return {"status": "ok", "backend": "fastembed", "version": ds.version}

    if EMBEDDING_BACKEND in {"fastembed", "auto"}:
        return {
            "status": "failed",
            "backend": EMBEDDING_BACKEND,
            "reason": f"fastembed not installed: {ds.error}",
            "fix": "pip install fastembed",
        }

    return {"status": "ok", "backend": EMBEDDING_BACKEND, "reason": "unknown-backend"}


def check_vector_store() -> dict[str, Any]:
    """Check if ChromaDB and the vector store client are functional."""
    chroma_ds = _check_package("chromadb")
    langchain_chroma_ds = _check_package("langchain_chroma")

    if not chroma_ds.installed:
        return {
            "status": "failed",
            "reason": f"chromadb not installed: {chroma_ds.error}",
            "fix": "pip install chromadb",
        }

    if not langchain_chroma_ds.installed:
        # Fallback to langchain_community.vectorstores.Chroma
        lc_ds = _check_package("langchain_community")
        if lc_ds.installed:
            return {"status": "ok", "client": "langchain_community.vectorstores.Chroma", "chromadb": chroma_ds.version}
        return {
            "status": "failed",
            "reason": "Neither langchain-chroma nor langchain-community available",
            "fix": "pip install langchain-chroma",
        }

    return {"status": "ok", "client": "langchain_chroma.Chroma", "chromadb": chroma_ds.version}


def startup_validation() -> bool:
    """Run at application startup. Returns True if safe to proceed."""
    report = check_runtime_dependencies()

    if report.status == "failed":
        log.critical(
            "Startup validation FAILED. Missing: %s. "
            "The application cannot serve requests correctly without these packages.",
            ", ".join(report.missing_critical),
        )
        return False

    embed_check = check_embedding_backend()
    if embed_check["status"] == "failed":
        log.critical(
            "Embedding backend FAILED: %s. "
            "Set EMBEDDING_BACKEND=fastembed and install fastembed, "
            "or set EMBEDDING_BACKEND=hashing for development only.",
            embed_check.get("reason", "unknown"),
        )
        return False

    vs_check = check_vector_store()
    if vs_check["status"] == "failed":
        log.critical(
            "Vector store FAILED: %s. "
            "Install using: pip install -r requirements.txt",
            vs_check.get("reason", "unknown"),
        )
        return False

    log.info(
        "Startup validation OK — chromadb=%s, embedding_backend=%s, vector_store=%s",
        report.dependencies.get("chromadb", DepStatus(False)).version,
        embed_check.get("backend"),
        vs_check.get("client"),
    )
    return True
