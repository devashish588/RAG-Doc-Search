"""Phase 14.0 — Render Python Runtime & Dependency Reproducibility Regression Tests."""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.dependency_check import (
    check_runtime_dependencies,
    startup_validation,
    StartupValidationError,
    DependencyReport,
)
from backend.settings import BASE_DIR


class TestPhase14RuntimeReproducibility:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_python_312_accepted(self):
        with patch("sys.version_info", (3, 12, 13, "final", 0)), patch("sys.version", "3.12.13 (main)"):
            with patch("backend.dependency_check.check_runtime_dependencies") as mock_check:
                mock_report = DependencyReport(status="ok", is_virtualenv=True)
                mock_check.return_value = mock_report
                with patch("backend.dependency_check.check_embedding_backend", return_value={"status": "ok", "backend": "fastembed"}), \
                     patch("backend.dependency_check.check_vector_store", return_value={"status": "ok", "client": "langchain_chroma.Chroma"}):
                    res = startup_validation(raise_on_failure=False)
                    assert res is True

    def test_python_314_rejected(self):
        with patch("sys.version_info", (3, 14, 0, "alpha", 0)), patch("sys.version", "3.14.0a1"):
            with pytest.raises(StartupValidationError) as exc_info:
                startup_validation(raise_on_failure=True)
            assert "UNSUPPORTED PYTHON RUNTIME" in str(exc_info.value)
            assert "3.14.0a1" in str(exc_info.value)
            assert "Supported production runtime: Python 3.12.x" in str(exc_info.value)

    def test_critical_dependencies_imported(self):
        report = check_runtime_dependencies()
        assert "chromadb" in report.dependencies
        assert report.dependencies["chromadb"].installed is True
        assert "fastembed" in report.dependencies
        assert report.dependencies["fastembed"].installed is True
        assert "langchain-chroma" in report.dependencies
        assert report.dependencies["langchain-chroma"].installed is True
        assert "numpy" in report.dependencies
        assert report.dependencies["numpy"].installed is True
        assert "rank-bm25" in report.dependencies
        assert report.dependencies["rank-bm25"].installed is True

    def test_chroma_availability(self):
        import chromadb
        assert chromadb.__version__ is not None

    def test_fastembed_availability(self):
        import fastembed
        assert fastembed is not None

    def test_langchain_chroma_availability(self):
        import langchain_chroma
        assert langchain_chroma.Chroma is not None

    def test_application_startup_validation_succeeds_under_venv(self):
        res = startup_validation(raise_on_failure=False)
        assert res is True

    def test_healthz_endpoint_behavior(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_readyz_dependency_reporting(self, client):
        resp = client.get("/readyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ready", "ok", "degraded")
        assert "checks" in data
        assert "dependencies" in data["checks"]

    def test_render_and_docker_config_files_exist(self):
        runtime_txt = BASE_DIR / "runtime.txt"
        python_version_file = BASE_DIR / ".python-version"
        render_yaml = BASE_DIR / "render.yaml"
        dockerfile = BASE_DIR / "Dockerfile"

        assert runtime_txt.exists(), "runtime.txt missing"
        assert "python-3.12" in runtime_txt.read_text(encoding="utf-8")

        assert python_version_file.exists(), ".python-version missing"
        assert "3.12" in python_version_file.read_text(encoding="utf-8")

        assert render_yaml.exists(), "render.yaml missing"
        render_content = render_yaml.read_text(encoding="utf-8")
        assert "PYTHON_VERSION" in render_content
        assert "3.12.9" in render_content

        assert dockerfile.exists(), "Dockerfile missing"
        docker_content = dockerfile.read_text(encoding="utf-8")
        assert "python:3.12-slim" in docker_content
