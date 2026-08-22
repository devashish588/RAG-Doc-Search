"""Ingestion runtime smoke test.

Tests the full ingestion pipeline: upload → chunk → embed → Chroma → BM25.
Uses the TestClient to simulate real API ingestion.
"""
import io
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.middleware import reset_limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_limiter():
    """Ensure generous rate limit for ingestion tests."""
    reset_limiter(max_requests=1000, window_seconds=60)
    yield


class TestIngestionPipeline:
    def test_upload_accepted(self):
        """Upload a small .txt file and verify it is accepted."""
        test_content = b"HybridRAG ingestion smoke test. This is a test document for pipeline validation."
        test_filename = f"smoke_test_{uuid.uuid4().hex[:8]}.txt"

        files = {"file": (test_filename, io.BytesIO(test_content), "text/plain")}
        resp = client.post("/v1/ingest", files=files)
        assert resp.status_code == 202, f"Upload failed: {resp.text}"
        data = resp.json()
        assert "document_id" in data
        assert data["status"] == "queued"

    def test_retrieve_after_ingestion(self):
        """After prior ingestion, verify retrieval works across modes."""
        for mode in ["dense", "sparse", "hybrid", "hybrid_rerank"]:
            resp = client.post("/v1/ask", json={
                "question": "ingestion smoke test",
                "retrieval_mode": mode,
            })
            assert resp.status_code == 200, f"Mode {mode} failed: {resp.text}"
            data = resp.json()
            assert "answer" in data
            assert "retrieval_trace" in data

    def test_ask_all_modes_structure(self):
        """Verify all four modes return complete response structure."""
        for mode in ["dense", "sparse", "hybrid", "hybrid_rerank"]:
            resp = client.post("/v1/ask", json={
                "question": "test document",
                "retrieval_mode": mode,
            })
            assert resp.status_code == 200, f"Mode {mode} failed: {resp.text}"
            data = resp.json()
            assert "answer" in data
            assert "confidence" in data
            assert "retrieval_trace" in data
