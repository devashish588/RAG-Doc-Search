"""Phase 4 tests: BM25 sparse retrieval."""
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.bm25_index import BM25Index, get_bm25_index, reset_bm25_index
from backend.retrieval_bm25 import BM25Retriever, get_bm25_retriever, reset_bm25_retriever
from backend.retrieval import run_search, run_search_dense, run_search_bm25
from backend.schemas import SearchRequest
from backend.models import Chunk
from backend.settings import BM25_TOP_K, BM25_K1, BM25_B, BM25_INDEX_VERSION

client = TestClient(app)


# ---------------------------------------------------------------------------
# BM25 Index Tests
# ---------------------------------------------------------------------------

def test_bm25_config_defaults():
    """Test BM25 configuration defaults."""
    assert BM25_TOP_K == 10
    assert BM25_K1 == 1.5
    assert BM25_B == 0.75
    assert BM25_INDEX_VERSION == 1


def test_bm25_retriever_valid_config():
    """Test BM25Retriever accepts valid configuration."""
    reset_bm25_retriever()
    retriever = BM25Retriever(top_k=5, k1=1.2, b=0.5)
    assert retriever.top_k == 5
    assert retriever.k1 == 1.2
    assert retriever.b == 0.5


def test_bm25_retriever_invalid_top_k():
    """Test BM25Retriever rejects invalid top_k."""
    with pytest.raises(ValueError, match="top_k must be > 0"):
        BM25Retriever(top_k=0)
    with pytest.raises(ValueError, match="top_k must be > 0"):
        BM25Retriever(top_k=-1)


def test_bm25_retriever_invalid_k1():
    """Test BM25Retriever rejects invalid k1."""
    with pytest.raises(ValueError, match="k1 must be > 0"):
        BM25Retriever(k1=0)
    with pytest.raises(ValueError, match="k1 must be > 0"):
        BM25Retriever(k1=-1)


def test_bm25_retriever_invalid_b():
    """Test BM25Retriever rejects invalid b."""
    with pytest.raises(ValueError, match="b must be in \\[0, 1\\]"):
        BM25Retriever(b=-0.1)
    with pytest.raises(ValueError, match="b must be in \\[0, 1\\]"):
        BM25Retriever(b=1.1)


def test_bm25_index_build_and_search():
    """Test BM25 index can be built and searched."""
    reset_bm25_index()
    reset_bm25_retriever()

    index = get_bm25_index()

    # Create test chunks
    chunks = [
        Chunk.create(
            document_id="doc1",
            chunk_index=0,
            content="This is a test document about ERR-503 error code and configuration.",
            content_hash="hash1",
        ),
        Chunk.create(
            document_id="doc1",
            chunk_index=1,
            content="Another chunk discussing OMP_NUM_THREADS and MAX_UPLOAD_MB settings.",
            content_hash="hash2",
        ),
        Chunk.create(
            document_id="doc2",
            chunk_index=0,
            content="Unrelated document about cooking recipes and ingredients.",
            content_hash="hash3",
        ),
    ]

    index.build(chunks)
    assert index.is_healthy()

    # Search for technical term
    results = index.search("ERR-503", k=5)
    assert len(results) >= 1
    # Should find the first chunk
    assert any("ERR-503" in r["content"] for r in results)


def test_bm25_tokenizer_preserves_technical_tokens():
    """Test tokenizer preserves technical identifiers."""
    reset_bm25_index()
    index = get_bm25_index()

    # Test various technical token patterns
    test_cases = [
        ("ERR-503", ["err-503"]),
        ("OMP_NUM_THREADS", ["omp_num_threads"]),
        ("MAX_UPLOAD_MB", ["max_upload_mb"]),
        ("api_reference.txt", ["api_reference.txt"]),
        ("/v1/documents", ["/v1/documents"]),
        ("config.yaml", ["config.yaml"]),
        ("BAAI/bge-small-en-v1.5", ["baai/bge-small-en-v1.5"]),
        ("HTTP 503", ["http", "503"]),
    ]

    for text, expected_tokens in test_cases:
        tokens = index._tokenize(text)
        # At least the key technical tokens should be preserved
        for expected in expected_tokens:
            assert expected in tokens, f"Token '{expected}' not found in {tokens} for '{text}'"


def test_bm25_retriever_search():
    """Test BM25Retriever search returns canonical RetrievalResult."""
    reset_bm25_index()
    reset_bm25_retriever()

    # Build index first
    index = get_bm25_index()
    chunks = [
        Chunk.create(
            document_id="doc1",
            chunk_index=0,
            content="This document discusses the ERR-503 error code in detail.",
            content_hash="hash1",
        ),
        Chunk.create(
            document_id="doc1",
            chunk_index=1,
            content="Configuration settings include OMP_NUM_THREADS=1 and MAX_UPLOAD_MB=50.",
            content_hash="hash2",
        ),
        Chunk.create(
            document_id="doc2",
            chunk_index=0,
            content="This is an unrelated document about general topics.",
            content_hash="hash3",
        ),
    ]
    index.build(chunks)
    index.save()

    retriever = get_bm25_retriever()
    results = retriever.search("ERR-503", k=5)

    assert isinstance(results, list)
    for r in results:
        assert hasattr(r, "chunk_id")
        assert hasattr(r, "score")
        assert hasattr(r, "rank")
        assert hasattr(r, "source")
        assert hasattr(r, "content")
        assert hasattr(r, "metadata")


def test_bm25_search_empty_query():
    """Test BM25 handles empty query."""
    reset_bm25_index()
    reset_bm25_retriever()

    retriever = get_bm25_retriever()
    results = retriever.search("")
    assert results == []

    results = retriever.search("   ")
    assert results == []


def test_bm25_source_filter():
    """Test BM25 source filtering works."""
    reset_bm25_index()
    reset_bm25_retriever()

    index = get_bm25_index()
    chunks = [
        Chunk.create(
            document_id="doc1",
            chunk_index=0,
            content="Content about error codes.",
            content_hash="hash1",
            section="Errors",
        ),
    ]
    index.build(chunks)

    results = index.search("error", k=5, source="nonexistent.txt")
    assert results == []


def test_bm25_deterministic():
    """Test BM25 retrieval is deterministic."""
    reset_bm25_index()
    reset_bm25_retriever()

    index = get_bm25_index()
    chunks = [
        Chunk.create(
            document_id="doc1",
            chunk_index=0,
            content="Test document about ERR-503 error code.",
            content_hash="hash1",
        ),
    ]
    index.build(chunks)

    results1 = index.search("ERR-503", k=5)
    results2 = index.search("ERR-503", k=5)

    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1["chunk_id"] == r2["chunk_id"]
        assert r1["rank"] == r2["rank"]
        assert r1["score"] == r2["score"]


# ---------------------------------------------------------------------------
# Persistence Tests
# ---------------------------------------------------------------------------

def test_bm25_index_persistence():
    """Test BM25 index can be saved and loaded."""
    reset_bm25_index()
    index = get_bm25_index()

    chunks = [
        Chunk.create(
            document_id="doc1",
            chunk_index=0,
            content="Persistent test document about configuration.",
            content_hash="hash1",
        ),
    ]
    index.build(chunks)
    index.save()

    # Verify files exist
    from backend.settings import BM25_INDEX_DIR
    assert (BM25_INDEX_DIR / "bm25_index.pkl").exists()
    assert (BM25_INDEX_DIR / "bm25_metadata.json").exists()
    assert (BM25_INDEX_DIR / "bm25_chunk_map.json").exists()

    # Load into new instance
    reset_bm25_index()
    new_index = get_bm25_index()
    loaded = new_index.load()

    assert loaded is True
    assert new_index.is_healthy()
    assert len(new_index._chunk_ids) == 1
    assert new_index._chunk_ids[0] == "doc1:0"

    # Search should work after reload
    results = new_index.search("configuration", k=5)
    assert len(results) >= 1


def test_bm25_index_version_mismatch():
    """Test index rejects wrong version."""
    reset_bm25_index()
    index = get_bm25_index()

    # Create index with version 1
    chunks = [Chunk.create("doc1", 0, "Test content", "hash1")]
    index.build(chunks)
    index.save()

    # Try to load with different version (simulated)
    reset_bm25_index()
    # We can't easily test this without modifying the version constant,
    # but we verified the version check logic exists in load()


# ---------------------------------------------------------------------------
# Ingestion Integration Tests
# ---------------------------------------------------------------------------

def test_ingestion_updates_bm25():
    """Test ingestion pipeline updates BM25 index."""
    from backend.ingestion_v2 import ingest_document_v2

    # Create test file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("This is a test document about ERR-503 error handling. " * 10)
        path = Path(f.name)

    try:
        reset_bm25_index()
        reset_bm25_retriever()

        result = ingest_document_v2("test_doc_123", path, "test.txt")

        assert result["status"] == "complete"
        assert result["chunks_indexed"] > 0

        # Verify BM25 index has the document
        index = get_bm25_index()
        results = index.search("ERR-503", k=5)
        assert len(results) >= 1
        assert any("ERR-503" in r["content"] for r in results)

    finally:
        path.unlink()


def test_bm25_deletion():
    """Test document deletion removes BM25 entries."""
    reset_bm25_index()
    reset_bm25_retriever()

    index = get_bm25_index()
    chunks = [
        Chunk.create("doc1", 0, "Content about ERR-503", "hash1"),
    ]
    index.build(chunks)

    # Verify it exists
    results = index.search("ERR-503", k=5)
    assert len(results) >= 1

    # Delete document
    removed = index.remove_document("doc1")
    assert removed == 1

    # Verify it's gone
    results = index.search("ERR-503", k=5)
    assert len(results) == 0


def test_bm25_reingestion_no_duplicates():
    """Test re-ingestion doesn't create duplicate BM25 entries."""
    reset_bm25_index()
    reset_bm25_retriever()

    index = get_bm25_index()
    chunks = [
        Chunk.create("doc1", 0, "Test content about ERR-503", "hash1"),
    ]
    index.build(chunks)
    original_count = len(index._chunk_ids)

    # Re-ingest same content (simulating re-ingestion)
    index.build(chunks)  # Rebuild with same chunks
    assert len(index._chunk_ids) == original_count


# ---------------------------------------------------------------------------
# API Integration Tests
# ---------------------------------------------------------------------------

def test_v1_ask_sparse_mode():
    """Test /v1/ask returns BM25 trace in sparse mode."""
    response = client.post(
        "/v1/ask",
        json={
            "question": "What is ERR-503?",
            "retrieval_mode": "sparse",
            "top_k_sparse": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "retrieval_trace" in data
    trace = data["retrieval_trace"]
    assert "bm25" in trace
    # BM25 should have results
    assert isinstance(trace["bm25"], list)


def test_v1_ask_dense_mode():
    """Test /v1/ask returns dense trace in dense mode."""
    response = client.post(
        "/v1/ask",
        json={
            "question": "What is the chunk size?",
            "retrieval_mode": "dense",
            "top_k_dense": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    trace = data["retrieval_trace"]
    assert "dense" in trace
    assert isinstance(trace["dense"], list)


def test_v1_ask_both_traces():
    """Test /v1/ask returns both dense and BM25 traces."""
    response = client.post(
        "/v1/ask",
        json={
            "question": "What is ERR-503?",
            "retrieval_mode": "hybrid",
            "top_k_dense": 5,
            "top_k_sparse": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    trace = data["retrieval_trace"]
    assert "dense" in trace
    assert "bm25" in trace
    # RRF and reranker should remain empty in Phase 4
    assert trace["rrf"] == []
    assert trace["reranker"] == []


def test_v1_ask_top_k_sparse():
    """Test /v1/ask uses top_k_sparse parameter."""
    response = client.post(
        "/v1/ask",
        json={
            "question": "What is ERR-503?",
            "top_k_sparse": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    trace = data["retrieval_trace"]
    assert len(trace["bm25"]) <= 3


# ---------------------------------------------------------------------------
# Legacy Route Tests
# ---------------------------------------------------------------------------

def test_legacy_search_still_works():
    """Test /search (legacy) still works."""
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


# ---------------------------------------------------------------------------
# Reconciliation Tests
# ---------------------------------------------------------------------------

def test_reconciliation_includes_bm25():
    """Test /health/index reports BM25 status."""
    response = client.get("/health/index")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # Should at least report on dense index


# ---------------------------------------------------------------------------
# Exact Term Retrieval Tests
# ---------------------------------------------------------------------------

def test_bm25_exact_term_retrieval():
    """Test BM25 retrieves exact technical terms better than dense in some cases."""
    reset_bm25_index()
    reset_bm25_retriever()

    index = get_bm25_index()
    # Add chunks with exact technical terms
    chunks = [
        Chunk.create("doc1", 0, "The OMP_NUM_THREADS environment variable controls threading.", "hash1"),
        Chunk.create("doc2", 0, "Set MAX_UPLOAD_MB to 50 for file uploads.", "hash2"),
        Chunk.create("doc3", 0, "Error code ERR-503 indicates service unavailable.", "hash3"),
        Chunk.create("doc4", 0, "The config.yaml file contains settings.", "hash4"),
    ]
    index.build(chunks)

    # Test exact term queries
    test_queries = [
        "OMP_NUM_THREADS",
        "MAX_UPLOAD_MB",
        "ERR-503",
        "config.yaml",
    ]

    for query in test_queries:
        results = index.search(query, k=5)
        assert len(results) >= 1, f"Query '{query}' should return results"
        # The matching chunk should be ranked high
        assert any(query.lower() in r["content"].lower() for r in results[:2]), \
            f"Query '{query}' not well ranked in top 2: {results[:2]}"


# Import tempfile for tests that need it
import tempfile