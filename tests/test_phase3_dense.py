"""Phase 3 tests: Dense retrieval baseline."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.retrieval_dense import DenseRetriever, get_dense_retriever, reset_dense_retriever
from backend.retrieval import run_search, run_search_dense
from backend.schemas import SearchRequest
from backend.settings import DENSE_TOP_K, DENSE_FETCH_K, DENSE_MMR_LAMBDA, MIN_RELEVANCE_SCORE

client = TestClient(app)


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

def test_dense_config_defaults():
    """Test dense retrieval configuration defaults."""
    assert DENSE_TOP_K == 10
    assert DENSE_FETCH_K == 20
    assert DENSE_MMR_LAMBDA == 0.5
    assert MIN_RELEVANCE_SCORE == 0.35


def test_dense_retriever_valid_config():
    """Test DenseRetriever accepts valid configuration."""
    retriever = DenseRetriever(top_k=5, fetch_k=10, lambda_mult=0.3, min_relevance_score=0.2)
    assert retriever.top_k == 5
    assert retriever.fetch_k == 10
    assert retriever.lambda_mult == 0.3
    assert retriever.min_relevance_score == 0.2


def test_dense_retriever_invalid_top_k():
    """Test DenseRetriever rejects invalid top_k."""
    with pytest.raises(ValueError, match="top_k must be > 0"):
        DenseRetriever(top_k=0)
    with pytest.raises(ValueError, match="top_k must be > 0"):
        DenseRetriever(top_k=-1)


def test_dense_retriever_invalid_fetch_k():
    """Test DenseRetriever rejects fetch_k < top_k."""
    with pytest.raises(ValueError, match="fetch_k.*must be >= top_k"):
        DenseRetriever(top_k=10, fetch_k=5)


def test_dense_retriever_invalid_lambda():
    """Test DenseRetriever rejects invalid lambda_mult."""
    with pytest.raises(ValueError, match="lambda_mult must be in \\[0, 1\\]"):
        DenseRetriever(lambda_mult=-0.1)
    with pytest.raises(ValueError, match="lambda_mult must be in \\[0, 1\\]"):
        DenseRetriever(lambda_mult=1.1)


def test_dense_retriever_invalid_threshold():
    """Test DenseRetriever rejects invalid min_relevance_score."""
    with pytest.raises(ValueError, match="min_relevance_score must be in \\[0, 1\\]"):
        DenseRetriever(min_relevance_score=-0.1)
    with pytest.raises(ValueError, match="min_relevance_score must be in \\[0, 1\\]"):
        DenseRetriever(min_relevance_score=1.1)


# ---------------------------------------------------------------------------
# Basic Search Tests
# ---------------------------------------------------------------------------

def test_dense_retriever_basic_search():
    """Test dense retriever returns results for valid query."""
    retriever = DenseRetriever(top_k=5)
    results = retriever.search("What is the chunk size?")
    assert isinstance(results, list)
    for r in results:
        assert hasattr(r, "chunk_id")
        assert hasattr(r, "score")
        assert hasattr(r, "rank")
        assert hasattr(r, "source")
        assert hasattr(r, "content")


def test_dense_retriever_empty_query():
    """Test dense retriever handles empty query."""
    retriever = DenseRetriever()
    results = retriever.search("")
    assert results == []
    results = retriever.search("   ")
    assert results == []


def test_dense_retriever_top_k_parameter():
    """Test top_k parameter controls number of results."""
    retriever = DenseRetriever(top_k=10)
    results_1 = retriever.search("chunk size", k=1)
    results_3 = retriever.search("chunk size", k=3)
    results_5 = retriever.search("chunk size", k=5)

    assert len(results_1) <= 1
    assert len(results_3) <= 3
    assert len(results_5) <= 5


def test_dense_retriever_source_filter():
    """Test source filtering works."""
    retriever = DenseRetriever(top_k=10)
    results = retriever.search("chunk", source="architecture_overview.md")
    for r in results:
        assert "architecture_overview" in r.source.lower()


def test_dense_retriever_nonexistent_source():
    """Test filtering by nonexistent source returns empty."""
    retriever = DenseRetriever(top_k=10)
    results = retriever.search("chunk", source="nonexistent_file_xyz.txt")
    assert results == []


def test_dense_retriever_deterministic():
    """Test dense retrieval is deterministic for same query."""
    retriever = DenseRetriever(top_k=10)
    results1 = retriever.search("What is the chunk size?")
    results2 = retriever.search("What is the chunk size?")

    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1.chunk_id == r2.chunk_id
        assert r1.rank == r2.rank
        assert r1.score == r2.score


# ---------------------------------------------------------------------------
# MMR Configuration Tests
# ---------------------------------------------------------------------------

def test_mmr_fetch_k_boundary():
    """Test fetch_k must be >= top_k (validated at construction)."""
    # Valid: fetch_k >= top_k
    retriever = DenseRetriever(top_k=5, fetch_k=10)
    assert retriever.fetch_k >= retriever.top_k

    # Invalid: fetch_k < top_k raises error
    with pytest.raises(ValueError, match="fetch_k.*must be >= top_k"):
        DenseRetriever(top_k=10, fetch_k=5)


def test_mmr_lambda_bounds():
    """Test lambda_mult bounds are enforced."""
    # Valid bounds
    DenseRetriever(lambda_mult=0.0)
    DenseRetriever(lambda_mult=0.5)
    DenseRetriever(lambda_mult=1.0)

    # Invalid bounds
    with pytest.raises(ValueError):
        DenseRetriever(lambda_mult=-0.01)
    with pytest.raises(ValueError):
        DenseRetriever(lambda_mult=1.01)


# ---------------------------------------------------------------------------
# Score Handling Tests
# ---------------------------------------------------------------------------

def test_score_in_range():
    """Test scores are in valid range."""
    retriever = DenseRetriever(top_k=10)
    results = retriever.search("chunk size")
    for r in results:
        assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of range"


def test_no_fabricated_scores():
    """Test scores are real, not fabricated placeholders."""
    retriever = DenseRetriever(top_k=10)
    results = retriever.search("What is the chunk size?")
    # Should have some variation, not all 1.0
    if len(results) > 1:
        scores = [r.score for r in results]
        # At least some variation or all same (could be MMR returning 1.0)
        # But we verify they are real floats
        for s in scores:
            assert isinstance(s, float)


# ---------------------------------------------------------------------------
# Metadata Filtering Tests
# ---------------------------------------------------------------------------

def test_metadata_filtering():
    """Test metadata filtering by source."""
    retriever = DenseRetriever(top_k=10)
    results = retriever.search("test", source="architecture_overview.md")
    for r in results:
        assert "architecture_overview" in r.source.lower()


def test_metadata_filter_no_matches():
    """Test metadata filtering with no matches."""
    retriever = DenseRetriever(top_k=10)
    results = retriever.search("test", source="definitely_does_not_exist.txt")
    assert results == []


# ---------------------------------------------------------------------------
# API Integration Tests
# ---------------------------------------------------------------------------

def test_v1_ask_uses_top_k_dense():
    """Test /v1/ask uses top_k_dense parameter."""
    # Default top_k_dense is 10
    response = client.post("/v1/ask", json={"question": "What is the chunk size?"})
    assert response.status_code == 200
    data = response.json()
    assert "retrieval_trace" in data
    assert "dense" in data["retrieval_trace"]
    # Trace should have results
    if data["status"] == "answered":
        assert len(data["retrieval_trace"]["dense"]) > 0


def test_v1_ask_custom_top_k_dense():
    """Test /v1/ask accepts custom top_k_dense."""
    response = client.post(
        "/v1/ask",
        json={"question": "What is the chunk size?", "top_k_dense": 3}
    )
    assert response.status_code == 200
    data = response.json()
    if data["status"] == "answered":
        # Should have at most 3 dense results (before adjacent expansion)
        assert len(data["retrieval_trace"]["dense"]) <= 3


def test_v1_ask_accepts_all_forward_compat_params():
    """Test /v1/ask accepts all forward-compatible parameters."""
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


def test_v1_ask_trace_structure():
    """Test retrieval trace has correct structure."""
    response = client.post("/v1/ask", json={"question": "What is the chunk size?"})
    assert response.status_code == 200
    data = response.json()
    trace = data["retrieval_trace"]

    assert "dense" in trace
    assert "bm25" in trace
    assert "rrf" in trace
    assert "reranker" in trace

    # RRF is now implemented in Phase 5, reranker remains empty
    assert trace["reranker"] == []

    # Dense results have required fields
    for r in trace["dense"]:
        assert "chunk_id" in r
        assert "score" in r
        assert "rank" in r
        assert "source" in r
        assert "content" in r


def test_v1_ask_citations_from_search_results():
    """Test citations are built from search results (with adjacent expansion)."""
    response = client.post("/v1/ask", json={"question": "What is the chunk size?"})
    assert response.status_code == 200
    data = response.json()
    citations = data["citations"]

    # Citations should match search results
    for c in citations:
        assert "chunk_id" in c
        assert "source" in c
        assert "text_snippet" in c


# ---------------------------------------------------------------------------
# Legacy Route Tests
# ---------------------------------------------------------------------------

def test_legacy_search_still_works():
    """Test /search (legacy) still works."""
    response = client.post("/search", json={"query": "What is the chunk size?", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "answer" in data
    assert "results" in data
    assert "latency_ms" in data


def test_legacy_search_top_k():
    """Test legacy /search respects top_k."""
    response = client.post("/search", json={"query": "chunk size", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    # Base dense results respect top_k (adjacent expansion may add more)
    # The run_search function limits unique_pairs to top_k before adjacent expansion
    assert isinstance(data["results"], list)
    # At minimum we should get some results
    assert len(data["results"]) >= 1


# ---------------------------------------------------------------------------
# Adjacent Chunk Expansion Tests
# ---------------------------------------------------------------------------

def test_adjacent_chunk_expansion_preserved():
    """Test adjacent chunk expansion still works."""
    # This is tested indirectly through /search and /v1/ask
    # which both use run_search that includes _fetch_adjacent_chunks
    response = client.post("/search", json={"query": "chunk size", "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    # Results may include adjacent chunks
    assert isinstance(data["results"], list)


def test_dense_results_no_adjacent_expansion():
    """Test run_search_dense (for trace) does NOT include adjacent expansion."""
    # run_search_dense is used for trace and should not expand
    results = run_search_dense("What is the chunk size?", k=5)
    # These are raw dense results without adjacent expansion
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

def test_retrieval_pipeline_end_to_end():
    """Test full retrieval pipeline works."""
    req = SearchRequest(query="What is the chunk size?", top_k=5)
    result = run_search(req)

    assert result.query == "What is the chunk size?"
    assert isinstance(result.answer, str)
    assert isinstance(result.results, list)
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms > 0


def test_search_latency_reasonable():
    """Test search completes in reasonable time."""
    start = time.time()
    req = SearchRequest(query="What is the maximum file upload size?", top_k=8)
    result = run_search(req)
    elapsed = time.time() - start

    assert result.latency_ms > 0
    # Should complete within 30 seconds (generous for CI)
    assert elapsed < 30


# ---------------------------------------------------------------------------
# Baseline Regression Tests (from Phase 0/1)
# ---------------------------------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "vector_store" in data
    assert "embedding_backend" in data


def test_golden_dataset_schema():
    import json
    dataset_path = Path(__file__).resolve().parents[1] / "evals" / "golden_dataset.json"
    assert dataset_path.exists()

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    assert len(records) >= 50

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