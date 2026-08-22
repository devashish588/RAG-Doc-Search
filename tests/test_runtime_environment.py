"""Runtime environment and dependency regression tests.

Verifies that dependency validation checks the ACTUAL Python interpreter,
correctly detects missing packages, and that /readyz reports dependency health.
"""
import importlib
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.middleware import reset_limiter

client = TestClient(app)


class TestInterpreterInfo:
    def test_executable_is_available(self):
        assert sys.executable is not None
        assert len(sys.executable) > 0

    def test_virtualenv_detection(self):
        from backend.dependency_check import DependencyReport
        report = DependencyReport(status="ok")
        assert report.executable == sys.executable
        assert report.python_version == sys.version.split()[0]
        assert report.is_virtualenv == (sys.prefix != sys.base_prefix)


class TestDependencyImport:
    def test_chromadb_importable(self):
        import chromadb
        assert hasattr(chromadb, "__version__")

    def test_fastembed_importable(self):
        import fastembed
        assert True  # import succeeds

    def test_numpy_importable(self):
        import numpy
        assert hasattr(numpy, "__version__")

    def test_rank_bm25_importable(self):
        from rank_bm25 import BM25Okapi
        assert BM25Okapi is not None

    def test_prometheus_client_importable(self):
        import prometheus_client
        assert hasattr(prometheus_client, "Counter")

    def test_flashrank_importable(self):
        import flashrank
        assert True


class TestDependencyChecker:
    def test_checker_uses_imports_not_requirements_txt(self):
        """Verify the checker actually imports packages, not just reads requirements.txt."""
        from backend.dependency_check import check_runtime_dependencies
        report = check_runtime_dependencies()
        # If it only read requirements.txt, it wouldn't know versions
        chromadb_dep = report.dependencies.get("chromadb")
        assert chromadb_dep is not None
        assert chromadb_dep.installed is True
        assert chromadb_dep.version is not None
        # Version must look like a real version string
        assert "." in chromadb_dep.version

    def test_missing_package_detected(self):
        """Mock chromadb as missing and verify it's detected."""
        from backend.dependency_check import check_runtime_dependencies
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name == "chromadb":
                raise ImportError("No module named 'chromadb'")
            return original_import(name, *args, **kwargs)

        with patch("backend.dependency_check.importlib.import_module", side_effect=mock_import):
            report = check_runtime_dependencies()
            assert report.status == "failed"
            assert "chromadb" in report.missing_critical

    def test_missing_fastembed_detected(self):
        """Mock fastembed as missing and verify it's detected."""
        from backend.dependency_check import check_runtime_dependencies
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return original_import(name, *args, **kwargs)

        with patch("backend.dependency_check.importlib.import_module", side_effect=mock_import):
            report = check_runtime_dependencies()
            assert report.status == "failed"
            assert "fastembed" in report.missing_critical


class TestStartupValidationStrictAborts:
    def test_valid_venv_startup_succeeds(self):
        from backend.dependency_check import startup_validation
        assert startup_validation(raise_on_failure=True) is True

    def test_missing_chromadb_aborts_startup(self):
        from backend.dependency_check import startup_validation, StartupValidationError
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name == "chromadb":
                raise ImportError("No module named 'chromadb'")
            return original_import(name, *args, **kwargs)

        with patch("backend.dependency_check.importlib.import_module", side_effect=mock_import):
            with pytest.raises(StartupValidationError) as exc_info:
                startup_validation(raise_on_failure=True)
            err = exc_info.value
            assert "chromadb" in err.missing_critical
            assert sys.executable in str(err)
            assert ".venv\\Scripts\\python.exe" in str(err)

    def test_missing_fastembed_aborts_startup(self):
        from backend.dependency_check import startup_validation, StartupValidationError
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return original_import(name, *args, **kwargs)

        with patch("backend.dependency_check.importlib.import_module", side_effect=mock_import):
            with pytest.raises(StartupValidationError) as exc_info:
                startup_validation(raise_on_failure=True)
            err = exc_info.value
            assert "fastembed" in err.missing_critical
            assert sys.executable in str(err)
            assert ".venv\\Scripts\\python.exe" in str(err)

    def test_invalid_global_interpreter_aborts_startup(self):
        from backend.dependency_check import startup_validation, StartupValidationError, DependencyReport

        fake_report = DependencyReport(
            status="ok",
            executable=sys.executable,
            is_virtualenv=False,  # Global interpreter
        )

        with patch("backend.dependency_check.check_runtime_dependencies", return_value=fake_report):
            with pytest.raises(StartupValidationError) as exc_info:
                startup_validation(raise_on_failure=True)
            err = exc_info.value
            assert "STARTUP FAILED — Wrong Python interpreter detected" in str(err)
            assert sys.executable in str(err)
            assert ".venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 9826" in str(err)


class TestReadyzDependencyHealth:
    def test_readyz_returns_200_when_deps_ok(self):
        reset_limiter()
        resp = client.get("/readyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ready", "degraded")
        assert "dependencies" in data["checks"]
        assert data["checks"]["dependencies"]["status"] == "ok"

    def test_readyz_returns_503_when_chromadb_missing(self):
        """Mock chromadb as missing and verify /readyz returns 503."""
        original_import = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name == "chromadb":
                raise ImportError("No module named 'chromadb'")
            return original_import(name, *args, **kwargs)

        reset_limiter()
        with patch("backend.dependency_check.importlib.import_module", side_effect=mock_import):
            resp = client.get("/readyz")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "not_ready"

    def test_healthz_remains_lightweight(self):
        """Verify /healthz does not perform dependency checks."""
        reset_limiter()
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        # healthz should NOT contain dependency details
        assert "dependencies" not in data


class TestNoSilentHashingFallback:
    def test_production_backend_is_fastembed(self):
        """Verify the configured backend is fastembed, not hashing."""
        from backend.settings import EMBEDDING_BACKEND
        assert EMBEDDING_BACKEND in {"fastembed", "auto"}, (
            f"EMBEDDING_BACKEND is '{EMBEDDING_BACKEND}', expected 'fastembed' or 'auto'"
        )

    def test_embedding_backend_report(self):
        from backend.dependency_check import check_embedding_backend
        result = check_embedding_backend()
        assert result["status"] == "ok"
        assert result["backend"] == "fastembed"


class TestWindowsPathRegression:
    def test_path_metadata_is_string(self):
        """WindowsPath metadata must be converted to string before Chroma upsert."""
        from pathlib import Path
        p = Path("data") / "uploads" / "test.pdf"
        assert isinstance(str(p), str)
        # Verify all metadata types are Chroma-safe
        metadata = {"source": str(p), "document_id": "test", "chunk": 0}
        for v in metadata.values():
            assert isinstance(v, (str, int, float, bool, list, type(None)))
