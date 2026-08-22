"""Runtime dependency validation.

Validates that critical production dependencies are installed in the ACTUAL
Python interpreter running the application. Detects wrong-interpreter launches
(e.g., global Python 3.14 instead of project .venv Python 3.12).
"""
import importlib
import logging
import site
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


def _is_venv() -> bool:
    import os
    from pathlib import Path
    return (
        sys.prefix != sys.base_prefix
        or hasattr(sys, "real_prefix")
        or Path(sys.prefix, "pyvenv.cfg").exists()
        or os.getenv("VIRTUAL_ENV") is not None
        or os.getenv("CI") == "true"
        or os.getenv("GITHUB_ACTIONS") == "true"
        or os.getenv("RENDER") == "true"
    )


@dataclass
class DependencyReport:
    status: str  # "ok" | "degraded" | "failed"
    executable: str = field(default_factory=lambda: sys.executable)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    prefix: str = field(default_factory=lambda: sys.prefix)
    base_prefix: str = field(default_factory=lambda: sys.base_prefix)
    is_virtualenv: bool = field(default_factory=_is_venv)
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
    elif report.missing_optional:
        report.status = "degraded"
    else:
        report.status = "ok"

    return report


def log_environment_info() -> None:
    """Log runtime environment details. Called once at startup."""
    is_venv = _is_venv()
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else "N/A"

    log.info("Runtime environment:")
    log.info("  executable:     %s", sys.executable)
    log.info("  python:         %s", sys.version.split()[0])
    log.info("  prefix:         %s", sys.prefix)
    log.info("  base_prefix:    %s", sys.base_prefix)
    log.info("  virtualenv:     %s", is_venv)
    log.info("  site-packages:  %s", site_packages)

    if not is_venv:
        log.warning(
            "NOT running inside a virtual environment. "
            "The global 'python' resolves to: %s. "
            "This WILL cause ModuleNotFoundError for chromadb, fastembed, etc.",
            sys.executable,
        )


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
            "fix": ".venv\\Scripts\\python.exe -m pip install -r requirements.txt",
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
            "fix": ".venv\\Scripts\\python.exe -m pip install -r requirements.txt",
        }

    if not langchain_chroma_ds.installed:
        lc_ds = _check_package("langchain_community")
        if lc_ds.installed:
            return {"status": "ok", "client": "langchain_community.vectorstores.Chroma", "chromadb": chroma_ds.version}
        return {
            "status": "failed",
            "reason": "Neither langchain-chroma nor langchain-community available",
            "fix": ".venv\\Scripts\\python.exe -m pip install -r requirements.txt",
        }

    return {"status": "ok", "client": "langchain_chroma.Chroma", "chromadb": chroma_ds.version}


class StartupValidationError(RuntimeError):
    """Exception raised when application startup validation fails."""

    def __init__(self, message: str, executable: str, missing_critical: list[str]):
        self.executable = executable
        self.missing_critical = missing_critical
        super().__init__(message)


def startup_validation(raise_on_failure: bool = True) -> bool:
    """Run at application startup.

    Logs environment info, checks critical dependencies and interpreter status.
    If raise_on_failure=True and validation fails, raises StartupValidationError
    with actionable command instructions to immediately halt application startup.
    """
    log_environment_info()
    report = check_runtime_dependencies()
    failure_msg = None

    if sys.version_info >= (3, 13):
        failure_msg = (
            f"UNSUPPORTED PYTHON RUNTIME\n"
            f"  Detected: Python {sys.version.split()[0]}\n"
            f"  Supported production runtime: Python 3.12.x\n"
            f"  Executable: {report.executable}\n"
            f"  Use the project's supported Python 3.12 environment."
        )
    elif not report.is_virtualenv:
        missing = ", ".join(report.missing_critical) if report.missing_critical else "global interpreter without virtualenv"
        failure_msg = (
            f"STARTUP FAILED — Wrong Python interpreter detected.\n"
            f"  Executable: {report.executable}\n"
            f"  Missing / Environment issue: {missing}\n"
            f"  The global Python does NOT contain the required project virtualenv packages.\n"
            f"  FIX: Use the project virtual environment:\n"
            f"    .venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 9826"
        )
    elif report.status == "failed":
        missing = ", ".join(report.missing_critical)
        failure_msg = (
            f"STARTUP FAILED — Missing runtime dependencies in virtual environment.\n"
            f"  Executable: {report.executable}\n"
            f"  Missing: {missing}\n"
            f"  FIX: .venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
            f"  Command: .venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 9826"
        )

    if not failure_msg:
        embed_check = check_embedding_backend()
        if embed_check["status"] == "failed":
            failure_msg = (
                f"STARTUP FAILED — Embedding backend unavailable.\n"
                f"  Executable: {report.executable}\n"
                f"  Reason: {embed_check.get('reason', 'unknown')}\n"
                f"  FIX: .venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
                f"  Command: .venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 9826"
            )

    if not failure_msg:
        vs_check = check_vector_store()
        if vs_check["status"] == "failed":
            failure_msg = (
                f"STARTUP FAILED — Vector store unavailable.\n"
                f"  Executable: {report.executable}\n"
                f"  Reason: {vs_check.get('reason', 'unknown')}\n"
                f"  FIX: .venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
                f"  Command: .venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 9826"
            )

    if failure_msg:
        log.critical("%s", failure_msg)
        if raise_on_failure:
            raise StartupValidationError(failure_msg, report.executable, report.missing_critical)
        return False

    log.info(
        "Startup validation OK — chromadb=%s, embedding=%s, vector_store=%s",
        report.dependencies.get("chromadb", DepStatus(False)).version,
        embed_check.get("backend"),
        vs_check.get("client"),
    )
    return True
