import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.ingestion import get_document_status, delete_document, list_document_statuses
from backend.retrieval import run_search
from backend.schemas import SearchRequest
from backend.vector_store import get_vector_store

client = TestClient(app)
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
EVALS_DIR = WORKSPACE_DIR / "evals"


def test_health_endpoint():
    """Verify backend health endpoint returns status ok and system configuration."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "vector_store" in data
    assert "embedding_backend" in data


def test_upload_validation():
    """Verify file upload endpoint validates filenames and supported extensions."""
    # Invalid extension
    response = client.post(
        "/upload",
        files={"file": ("malicious_script.exe", b"print('hack')", "application/x-msdownload")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

    # Valid TXT file upload
    response = client.post(
        "/upload",
        files={"file": ("test_doc.txt", b"This is a test document for baseline regression testing.", "text/plain")}
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert "document_id" in data

    # Clean up test document
    doc_id = data["document_id"]
    delete_document(doc_id)


def test_retrieval_pipeline_execution():
    """Verify search pipeline handles valid queries and returns formatted SearchResponse."""
    req = SearchRequest(query="What is the maximum file upload size?", top_k=5)
    search_res = run_search(req)

    assert search_res.query == "What is the maximum file upload size?"
    assert isinstance(search_res.latency_ms, float)
    assert isinstance(search_res.results, list)
    assert isinstance(search_res.answer, str)


def test_golden_dataset_schema():
    """Verify golden_dataset.json contains at least 50 questions with complete schema."""
    dataset_path = EVALS_DIR / "golden_dataset.json"
    assert dataset_path.exists(), "golden_dataset.json missing"

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    assert len(records) >= 50, f"Expected 50+ questions, found {len(records)}"

    taxonomy_counts = {}
    for r in records:
        assert "id" in r
        assert "question" in r
        assert "type" in r
        assert "expected_answer" in r
        assert "expected_sources" in r

        qtype = r["type"]
        taxonomy_counts[qtype] = taxonomy_counts.get(qtype, 0) + 1

    assert taxonomy_counts.get("single-hop", 0) >= 15
    assert taxonomy_counts.get("exact-term", 0) >= 10
    assert taxonomy_counts.get("multi-hop", 0) >= 10
    assert taxonomy_counts.get("unanswerable", 0) >= 10
    assert taxonomy_counts.get("ambiguous", 0) >= 5


def test_corpus_manifest_integrity():
    """Verify corpus_manifest.json exists and tracks valid file metadata."""
    manifest_path = EVALS_DIR / "corpus_manifest.json"
    assert manifest_path.exists(), "corpus_manifest.json missing"

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("version") == "1.0"
    assert manifest.get("total_documents", 0) >= 4
    documents = manifest.get("documents", [])
    for doc in documents:
        assert "filename" in doc
        assert "sha256" in doc
        assert len(doc["sha256"]) == 64
        assert doc["size_bytes"] > 0
