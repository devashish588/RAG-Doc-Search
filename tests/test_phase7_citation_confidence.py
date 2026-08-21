import pytest
from fastapi.testclient import TestClient

from backend.api_v1 import router
from backend.confidence import (
    ConfidenceEstimator,
    normalize_bm25_scores,
    normalize_dense_score,
    normalize_reranker_score,
    normalize_rrf_score,
)
from backend.interfaces import RetrievalResult, RetrievalTrace
from backend.main import app
from backend.verifier import CitationVerifier

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Claim Extraction Tests
# ---------------------------------------------------------------------------

def test_claim_extraction_single_claim():
    verifier = CitationVerifier()
    answer = "The ingestion queue uses a background daemon worker."
    claims = verifier.extract_claims(answer)
    assert len(claims) == 1
    assert claims[0]["claim"] == "The ingestion queue uses a background daemon worker."


def test_claim_extraction_multiple_claims():
    verifier = CitationVerifier()
    answer = "FastAPI powers the REST backend. ChromaDB stores the dense vector embeddings. OpenRouter provides optional LLM answers."
    claims = verifier.extract_claims(answer)
    assert len(claims) == 3


def test_claim_extraction_bullet_list():
    verifier = CitationVerifier()
    answer = """
Here are the key points:
* Maximum upload file size is 50 MB.
* Supported formats are PDF, TXT, and Markdown.
* Server listens on port 9826.
    """
    claims = verifier.extract_claims(answer)
    assert len(claims) == 3
    assert any("50 MB" in c["claim"] for c in claims)


def test_claim_extraction_empty_answer():
    verifier = CitationVerifier()
    claims = verifier.extract_claims("")
    assert claims == []


def test_claim_extraction_with_inline_citations():
    verifier = CitationVerifier()
    answer = "The default chunk size is 1200 characters [1] and chunk overlap is 200 characters [architecture_overview.md, page 1]."
    claims = verifier.extract_claims(answer)
    assert len(claims) >= 1
    assert "[1]" not in claims[0]["claim"]
    assert "[architecture_overview.md" not in claims[0]["claim"]


# ---------------------------------------------------------------------------
# 2. Evidence Verification & Citations Tests
# ---------------------------------------------------------------------------

def test_evidence_verification_fully_supported():
    verifier = CitationVerifier()
    context = [
        RetrievalResult(
            chunk_id="doc1:0",
            score=0.9,
            rank=1,
            source="guide.pdf",
            content="The maximum upload file size limit is 50 MB.",
            metadata={"page": 1}
        )
    ]
    answer = "The maximum upload file size limit is 50 MB."
    result = verifier.verify(answer, context)

    assert result["metrics"]["total_claims"] == 1
    assert result["metrics"]["supported_claims"] == 1
    assert result["metrics"]["grounding_ratio"] == 1.0
    assert result["citations"][0]["verdict"] == "supported"
    assert result["citations"][0]["chunk_id"] == "doc1:0"
    assert result["citations"][0]["source"] == "guide.pdf"
    assert result["citations"][0]["page"] == 1


def test_evidence_verification_unsupported_claim():
    verifier = CitationVerifier()
    context = [
        RetrievalResult(
            chunk_id="doc1:0",
            score=0.9,
            rank=1,
            source="guide.pdf",
            content="The maximum upload file size limit is 50 MB.",
            metadata={"page": 1}
        )
    ]
    answer = "The server uses PostgreSQL for storing persistent user sessions."
    result = verifier.verify(answer, context)

    assert result["metrics"]["supported_claims"] == 0
    assert result["metrics"]["grounding_ratio"] == 0.0
    assert result["citations"][0]["verdict"] == "unsupported"
    assert result["citations"][0]["source"] is None
    assert result["citations"][0]["chunk_id"] is None


def test_evidence_verification_partially_supported():
    verifier = CitationVerifier()
    context = [
        RetrievalResult(
            chunk_id="doc1:0",
            score=0.85,
            rank=1,
            source="spec.txt",
            content="Maximum file upload limit is 50 MB.",
            metadata={}
        )
    ]
    answer = "Maximum file upload limit is 50 MB. Redis cluster handles endpoint caching."
    result = verifier.verify(answer, context)

    assert result["metrics"]["total_claims"] == 2
    assert result["metrics"]["supported_claims"] == 1
    assert result["metrics"]["unsupported_claims"] == 1
    assert result["metrics"]["grounding_ratio"] == 0.5


def test_evidence_verification_empty_evidence():
    verifier = CitationVerifier()
    result = verifier.verify("FastAPI handles REST routes.", [])
    assert result["metrics"]["supported_claims"] == 0
    assert result["citations"][0]["verdict"] == "unsupported"


# ---------------------------------------------------------------------------
# 3. Score Normalization & Confidence Estimator Tests
# ---------------------------------------------------------------------------

def test_dense_normalization_bounds():
    assert normalize_dense_score(1.5) == 1.0
    assert normalize_dense_score(-1.0) == 0.0
    assert normalize_dense_score(0.85) == 0.925


def test_bm25_normalization_negative_and_edge_cases():
    norm_neg = normalize_bm25_scores([-5.0, -2.0, 10.0])
    assert len(norm_neg) == 3
    assert all(0.0 <= s <= 1.0 for s in norm_neg)
    assert norm_neg[2] > norm_neg[0]

    norm_single = normalize_bm25_scores([5.0])
    assert len(norm_single) == 1
    assert 0.0 <= norm_single[0] <= 1.0

    norm_identical = normalize_bm25_scores([4.0, 4.0, 4.0])
    assert len(norm_identical) == 3
    assert norm_identical[0] == norm_identical[1]

    assert normalize_bm25_scores([]) == []


def test_rrf_normalization_bounds():
    assert 0.0 <= normalize_rrf_score(0.016) <= 1.0
    assert normalize_rrf_score(-0.01) == 0.0


def test_reranker_normalization_missing_value():
    assert normalize_reranker_score(None) is None
    assert 0.0 <= normalize_reranker_score(0.8) <= 1.0


def test_confidence_estimator_high_level():
    estimator = ConfidenceEstimator(abstention_threshold=0.50)
    dense_res = [RetrievalResult(chunk_id="d1:0", score=0.9, rank=1, source="s.txt", content="c")]
    conf = estimator.estimate(
        dense_results=dense_res,
        bm25_results=[],
        rrf_results=[],
        reranker_results=[],
        grounding_ratio=1.0,
        retrieval_mode="dense"
    )
    assert conf.level in {"high", "medium"}
    assert conf.overall_score >= 0.50
    assert conf.abstention_flag is False


def test_confidence_estimator_abstention_trigger():
    estimator = ConfidenceEstimator(abstention_threshold=0.50)
    conf = estimator.estimate(
        dense_results=[],
        bm25_results=[],
        rrf_results=[],
        reranker_results=[],
        grounding_ratio=0.0,
        retrieval_mode="dense"
    )
    assert conf.abstention_flag is True
    assert conf.level == "low"


# ---------------------------------------------------------------------------
# 4. Targeted Grounding Fixtures (1-8)
# ---------------------------------------------------------------------------

def test_fixture_1_fully_supported():
    verifier = CitationVerifier()
    context = [RetrievalResult(chunk_id="f1:0", score=0.95, rank=1, source="doc.txt", content="Port is 9826.")]
    res = verifier.verify("Port is 9826.", context)
    assert res["metrics"]["grounding_ratio"] == 1.0


def test_fixture_2_multi_source_supported():
    verifier = CitationVerifier()
    context = [
        RetrievalResult(chunk_id="f2:0", score=0.9, rank=1, source="a.txt", content="Port 9826 is used."),
        RetrievalResult(chunk_id="f2:1", score=0.9, rank=2, source="b.pdf", content="Limit is 50 MB.")
    ]
    res = verifier.verify("Port 9826 is used. Limit is 50 MB.", context)
    assert res["metrics"]["supported_claims"] == 2


def test_fixture_3_partially_supported():
    verifier = CitationVerifier()
    context = [RetrievalResult(chunk_id="f3:0", score=0.9, rank=1, source="a.txt", content="Port is 9826.")]
    res = verifier.verify("Port is 9826. System runs on Kubernetes engine.", context)
    assert res["metrics"]["supported_claims"] >= 1
    assert res["metrics"]["unsupported_claims"] >= 1


def test_fixture_4_completely_unsupported():
    verifier = CitationVerifier()
    context = [RetrievalResult(chunk_id="f4:0", score=0.5, rank=1, source="a.txt", content="Port is 9826.")]
    res = verifier.verify("System uses MySQL database cluster with master slave replication.", context)
    assert res["metrics"]["supported_claims"] == 0


def test_fixture_5_conflicting_evidence():
    verifier = CitationVerifier()
    context = [
        RetrievalResult(chunk_id="f5:0", score=0.8, rank=1, source="old.txt", content="Port is 8000."),
        RetrievalResult(chunk_id="f5:1", score=0.9, rank=2, source="new.txt", content="Port is 9826.")
    ]
    res = verifier.verify("Port is 9826.", context)
    assert res["metrics"]["supported_claims"] == 1


def test_fixture_6_ambiguous_question():
    estimator = ConfidenceEstimator(abstention_threshold=0.50)
    dense_res = [RetrievalResult(chunk_id="f6:0", score=0.5, rank=1, source="a.txt", content="Port 9826 or 5500.")]
    conf = estimator.estimate(
        dense_results=dense_res,
        bm25_results=[],
        rrf_results=[],
        reranker_results=[],
        grounding_ratio=0.5,
        retrieval_mode="dense"
    )
    assert 0.0 <= conf.overall_score <= 1.0


def test_fixture_7_empty_evidence():
    verifier = CitationVerifier()
    res = verifier.verify("Chunk size is 1200.", [])
    assert res["metrics"]["supported_claims"] == 0


def test_fixture_8_multiple_claims():
    verifier = CitationVerifier()
    context = [RetrievalResult(chunk_id="f8:0", score=0.9, rank=1, source="a.txt", content="Chunk size 1200. Overlap 200.")]
    res = verifier.verify("Chunk size 1200. Overlap 200.", context)
    assert res["metrics"]["total_claims"] == 2


# ---------------------------------------------------------------------------
# 5. API Integration Tests (Dense, Sparse, Hybrid, Hybrid_Rerank)
# ---------------------------------------------------------------------------

def test_api_v1_ask_dense_mode():
    payload = {
        "question": "What is the maximum file upload size?",
        "top_k_dense": 5,
        "retrieval_mode": "dense"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in {"answered", "insufficient_context"}
    assert "confidence" in data
    assert "overall_score" in data["confidence"]
    assert "grounding_metrics" in data


def test_api_v1_ask_sparse_mode():
    payload = {
        "question": "MAX_UPLOAD_MB",
        "top_k_sparse": 5,
        "retrieval_mode": "sparse"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "retrieval_trace" in data


def test_api_v1_ask_hybrid_mode():
    payload = {
        "question": "What is the default port?",
        "top_k_dense": 5,
        "top_k_sparse": 5,
        "top_k_fused": 10,
        "retrieval_mode": "hybrid"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieval_trace"]["rrf"] is not None


def test_api_v1_ask_hybrid_rerank_mode():
    payload = {
        "question": "What is the chunk size?",
        "top_k_dense": 5,
        "top_k_sparse": 5,
        "top_k_fused": 10,
        "top_k_final": 3,
        "retrieval_mode": "hybrid_rerank"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "confidence" in data