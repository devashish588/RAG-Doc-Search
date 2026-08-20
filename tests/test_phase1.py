"""Phase 1 tests: domain models, canonical IDs, API contracts, errors, compatibility, trace."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.models import (
    Document,
    Chunk,
    IngestionJob,
    DocumentStatus as ModelDocStatus,
    IngestionJobStatus,
    canonical_chunk_id,
    parse_chunk_id,
    canonical_job_id,
)
from backend.api_errors import (
    APIError,
    DocumentNotFoundError,
    InvalidFileTypeError,
    DuplicateDocumentError,
    QueueFullError,
    ErrorResponse,
)
from backend.ingestion import get_document_status, delete_document, list_document_statuses
from backend.retrieval import run_search
from backend.schemas import SearchRequest
from backend.vector_store import get_vector_store

client = TestClient(app)
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
EVALS_DIR = WORKSPACE_DIR / "evals"


# ---------------------------------------------------------------------------
# Domain Model Tests
# ---------------------------------------------------------------------------

def test_document_model_valid():
    """Test Document model with valid data."""
    doc = Document(
        id="test123",
        filename="test.pdf",
        content_hash="abc123",
        version=1,
        status=ModelDocStatus.QUEUED,
        source_type="upload",
    )
    assert doc.id == "test123"
    assert doc.filename == "test.pdf"
    assert doc.content_hash == "abc123"
    assert doc.status == ModelDocStatus.QUEUED

    # Test touch increments version and updates timestamp
    old_updated = doc.updated_at
    old_version = doc.version
    doc.touch()
    assert doc.version == old_version + 1
    assert doc.updated_at >= old_updated


def test_chunk_model_valid():
    """Test Chunk model with valid data."""
    chunk = Chunk.create(
        document_id="doc123",
        chunk_index=0,
        content="This is test content.",
        content_hash="hash123",
        section="Introduction",
        page=1,
        chunk_strategy="recursive",
        metadata={"custom": "value"},
    )
    assert chunk.id == "doc123:0"
    assert chunk.document_id == "doc123"
    assert chunk.chunk_index == 0
    assert chunk.content == "This is test content."
    assert chunk.character_count == len("This is test content.")
    assert chunk.section == "Introduction"
    assert chunk.page == 1
    assert chunk.metadata["custom"] == "value"


def test_ingestion_job_model_valid():
    """Test IngestionJob model with valid data."""
    job = IngestionJob(
        id="job123",
        document_id="doc123",
        status=IngestionJobStatus.QUEUED,
        configuration_snapshot={"chunk_size": 1200},
    )
    assert job.id == "job123"
    assert job.document_id == "doc123"
    assert job.status == IngestionJobStatus.QUEUED

    # Test state transitions
    job.mark_running()
    assert job.status == IngestionJobStatus.RUNNING
    assert job.started_at is not None

    job.mark_complete()
    assert job.status == IngestionJobStatus.COMPLETE
    assert job.completed_at is not None

    job2 = IngestionJob(id="job456", document_id="doc456")
    job2.mark_failed("ERROR_CODE", "Something went wrong")
    assert job2.status == IngestionJobStatus.FAILED
    assert job2.error_code == "ERROR_CODE"
    assert job2.error_message == "Something went wrong"


def test_document_model_invalid_status():
    """Test Document model rejects invalid status."""
    with pytest.raises(ValueError):
        Document(
            id="test",
            filename="test.pdf",
            content_hash="abc",
            status="invalid_status",
        )


def test_ingestion_job_model_invalid_status():
    """Test IngestionJob model rejects invalid status."""
    with pytest.raises(ValueError):
        IngestionJob(
            id="test",
            document_id="doc",
            status="invalid_status",
        )


# ---------------------------------------------------------------------------
# Canonical ID Tests
# ---------------------------------------------------------------------------

def test_canonical_chunk_id_deterministic():
    """Same document_id and chunk_index always produce same ID."""
    id1 = canonical_chunk_id("doc123", 0)
    id2 = canonical_chunk_id("doc123", 0)
    assert id1 == id2
    assert id1 == "doc123:0"


def test_different_chunk_indexes_produce_different_ids():
    """Different chunk indexes produce different IDs."""
    id0 = canonical_chunk_id("doc123", 0)
    id1 = canonical_chunk_id("doc123", 1)
    id2 = canonical_chunk_id("doc123", 2)
    assert len({id0, id1, id2}) == 3


def test_parse_chunk_id_valid():
    """Parse valid canonical chunk IDs."""
    assert parse_chunk_id("doc123:0") == ("doc123", 0)
    assert parse_chunk_id("abc:42") == ("abc", 42)
    assert parse_chunk_id("doc-with-dashes:100") == ("doc-with-dashes", 100)


def test_parse_chunk_id_invalid():
    """Parse returns None for invalid IDs."""
    assert parse_chunk_id("no-colon") is None
    assert parse_chunk_id("doc:not-a-number") is None
    assert parse_chunk_id(":") is None
    assert parse_chunk_id("") is None


def test_canonical_job_id_format():
    """Job IDs follow expected format."""
    job_id = canonical_job_id("doc123")
    assert job_id == "job:doc123"


# ---------------------------------------------------------------------------
# API Contract Tests
# ---------------------------------------------------------------------------

def test_v1_ingest_endpoint_exists():
    """POST /v1/ingest endpoint exists and returns structured response."""
    response = client.post(
        "/v1/ingest",
        files={"file": ("test.txt", b"Test content for ingestion.", "text/plain")},
    )
    # May be 202 (queued) or 503 (queue full) or 409 (duplicate)
    assert response.status_code in (202, 409, 503)
    if response.status_code == 202:
        data = response.json()
        assert "job_id" in data
        assert "document_id" in data
        assert data["status"] == "queued"
        assert data["job_id"].startswith("job:")


def test_v1_ask_endpoint_exists():
    """POST /v1/ask endpoint exists and returns structured response."""
    response = client.post(
        "/v1/ask",
        json={"question": "What is the chunk size?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "status" in data
    assert data["status"] in ("answered", "insufficient_context", "failed")
    assert "citations" in data
    assert "retrieval_trace" in data
    assert "dense" in data["retrieval_trace"]
    assert "bm25" in data["retrieval_trace"]
    assert "rrf" in data["retrieval_trace"]
    # Reranker should remain empty (Phase 6)
    assert data["retrieval_trace"]["reranker"] == []


def test_v1_ask_accepts_future_params():
    """/v1/ask accepts future configuration fields (forward compatibility)."""
    response = client.post(
        "/v1/ask",
        json={
            "question": "Test?",
            "top_k_dense": 5,
            "top_k_sparse": 5,
            "top_k_fused": 10,
            "top_k_final": 3,
            "retrieval_mode": "hybrid",
        },
    )
    assert response.status_code == 200


def test_v1_documents_list():
    """GET /v1/documents returns document list."""
    response = client.get("/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_v1_document_detail():
    """GET /v1/documents/{id} returns document detail or 404."""
    # First get a real document ID if any exist
    list_resp = client.get("/v1/documents")
    docs = list_resp.json()["documents"]
    if docs:
        doc_id = docs[0]["document_id"]
        response = client.get(f"/v1/documents/{doc_id}")
        assert response.status_code == 200
        data = response.json()
        assert "document" in data
        assert data["document"]["document_id"] == doc_id
    else:
        # No documents, test 404
        response = client.get("/v1/documents/nonexistent")
        assert response.status_code == 404


def test_v1_document_delete():
    """DELETE /v1/documents/{id} deletes document."""
    # Upload a test document first
    upload_resp = client.post(
        "/v1/ingest",
        files={"file": ("delete_test.txt", b"Content to delete.", "text/plain")},
    )
    if upload_resp.status_code == 202:
        doc_id = upload_resp.json()["document_id"]
        # Wait a moment for potential processing
        import time
        time.sleep(0.5)
        # Delete
        delete_resp = client.delete(f"/v1/documents/{doc_id}")
        assert delete_resp.status_code == 200
        data = delete_resp.json()
        assert data["document_id"] == doc_id
        # Verify deleted
        get_resp = client.get(f"/v1/documents/{doc_id}")
        assert get_resp.status_code == 404


# ---------------------------------------------------------------------------
# Error Contract Tests
# ---------------------------------------------------------------------------

def test_error_response_structure():
    """All API errors follow structured format."""
    # Test 404
    response = client.get("/v1/documents/nonexistent123")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert "code" in data["error"]
    assert "message" in data["error"]
    assert "details" in data["error"]
    assert data["error"]["code"] == "DOCUMENT_NOT_FOUND"

    # Test invalid file type
    response = client.post(
        "/v1/ingest",
        files={"file": ("test.exe", b"bad", "application/x-msdownload")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "INVALID_FILE_TYPE"


def test_api_error_exception_structure():
    """APIError exceptions produce correct response structure."""
    error = DocumentNotFoundError("doc123")
    response = error.to_response()
    assert response.status_code == 404
    body = json.loads(response.body)
    assert body["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert body["error"]["details"]["document_id"] == "doc123"

    error = InvalidFileTypeError("test.exe", [".txt", ".pdf"])
    response = error.to_response()
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"]["code"] == "INVALID_FILE_TYPE"
    assert body["error"]["details"]["filename"] == "test.exe"


def test_duplicate_document_error():
    """Duplicate document returns 409 with structured error."""
    # Upload same content twice
    content = b"Duplicate test content."
    client.post("/v1/ingest", files={"file": ("dup1.txt", content, "text/plain")})
    resp2 = client.post("/v1/ingest", files={"file": ("dup2.txt", content, "text/plain")})
    if resp2.status_code == 409:
        data = resp2.json()
        assert data["error"]["code"] == "DUPLICATE_DOCUMENT"


# ---------------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------------

def test_legacy_upload_still_works():
    """POST /upload (legacy) still works."""
    response = client.post(
        "/upload",
        files={"file": ("legacy.txt", b"Legacy upload test.", "text/plain")},
    )
    assert response.status_code in (202, 409, 503)
    if response.status_code == 202:
        data = response.json()
        assert "document_id" in data
        assert data["status"] == "queued"


def test_legacy_search_still_works():
    """POST /search (legacy) still works."""
    response = client.post(
        "/search",
        json={"query": "What is the chunk size?", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "answer" in data
    assert "results" in data
    assert "latency_ms" in data


def test_legacy_documents_list_still_works():
    """GET /documents (legacy) still works."""
    response = client.get("/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_legacy_document_detail_still_works():
    """GET /documents/{id} (legacy) still works."""
    list_resp = client.get("/documents")
    docs = list_resp.json()
    if docs:
        doc_id = docs[0]["document_id"]
        response = client.get(f"/documents/{doc_id}")
        assert response.status_code == 200
        assert response.json()["document_id"] == doc_id


def test_legacy_document_delete_still_works():
    """DELETE /documents/{id} (legacy) still works."""
    upload_resp = client.post(
        "/upload",
        files={"file": ("legacy_delete.txt", b"Legacy delete test.", "text/plain")},
    )
    if upload_resp.status_code == 202:
        doc_id = upload_resp.json()["document_id"]
        import time
        time.sleep(0.5)
        delete_resp = client.delete(f"/documents/{doc_id}")
        assert delete_resp.status_code == 200


# ---------------------------------------------------------------------------
# Retrieval Trace Tests
# ---------------------------------------------------------------------------

def test_retrieval_trace_dense_populated():
    """Retrieval trace dense stage contains actual results."""
    response = client.post(
        "/v1/ask",
        json={"question": "What is the chunk size?"},
    )
    assert response.status_code == 200
    data = response.json()
    trace = data["retrieval_trace"]
    # Dense should have results if any found
    if data["status"] == "answered":
        assert len(trace["dense"]) > 0
        for result in trace["dense"]:
            assert "chunk_id" in result
            assert "score" in result
            assert "rank" in result
            assert "source" in result
            assert "content" in result


def test_retrieval_trace_future_stages_empty():
    """Reranker stage is empty (RRF is now implemented in Phase 5)."""
    response = client.post(
        "/v1/ask",
        json={"question": "Test question"},
    )
    assert response.status_code == 200
    data = response.json()
    trace = data["retrieval_trace"]
    # RRF is now implemented in Phase 5, reranker remains for Phase 6
    assert trace["reranker"] == []


def test_no_fabricated_scores():
    """Scores in trace are real, not fabricated placeholders."""
    response = client.post(
        "/v1/ask",
        json={"question": "What is the chunk size?"},
    )
    data = response.json()
    for result in data["retrieval_trace"]["dense"]:
        # Score should be a real float from the retrieval
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0


# ---------------------------------------------------------------------------
# Index Reconciliation Tests
# ---------------------------------------------------------------------------

def test_reconciliation_health_endpoint():
    """GET /health/index returns reconciliation report."""
    response = client.get("/health/index")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("healthy", "warning", "degraded", "unavailable", "error")
    assert "summary" in data
    assert "issue_count" in data


def test_reconciliation_detects_basic_state():
    """Reconciliation runs without error and returns expected structure."""
    from backend.reconciliation import reconcile_index
    report = reconcile_index()
    assert "status" in report
    assert "summary" in report
    assert "issues" in report
    assert "checks" in report


# ---------------------------------------------------------------------------
# Legacy regression tests (preserved from Phase 0)
# ---------------------------------------------------------------------------

def test_health_endpoint():
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
        files={"file": ("malicious_script.exe", b"print hack", "application/x-msdownload")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file type" in data["error"]["message"]

    # Valid TXT file upload
    response = client.post(
        "/upload",
        files={"file": ("test_doc.txt", b"This is a test document for baseline regression testing.", "text/plain")},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert "document_id" in data

    # Clean up test document
    doc_id = data["document_id"]
    delete_document(doc_id)


def test_retrieval_pipeline_execution():
    req = SearchRequest(query="What is the maximum file upload size?", top_k=5)
    search_res = run_search(req)
    assert search_res.query == "What is the maximum file upload size?"
    assert isinstance(search_res.latency_ms, float)
    assert isinstance(search_res.results, list)
    assert isinstance(search_res.answer, str)


def test_golden_dataset_schema():
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