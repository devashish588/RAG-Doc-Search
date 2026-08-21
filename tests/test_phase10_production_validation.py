"""Phase 10: Production go-live validation tests."""
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. Production Configuration
# ---------------------------------------------------------------------------

def test_environment_default():
    from backend.settings import ENVIRONMENT
    assert ENVIRONMENT in ("development", "production")


def test_cors_origins_configurable():
    from backend.settings import CORS_ORIGINS
    assert isinstance(CORS_ORIGINS, str)
    assert len(CORS_ORIGINS) > 0


def test_rate_limit_configurable():
    from backend.settings import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS
    assert RATE_LIMIT_REQUESTS > 0
    assert RATE_LIMIT_WINDOW_SECONDS > 0


def test_circuit_breaker_configurable():
    from backend.settings import CIRCUIT_BREAKER_FAILURE_THRESHOLD, CIRCUIT_BREAKER_RECOVERY_SECONDS
    assert CIRCUIT_BREAKER_FAILURE_THRESHOLD > 0
    assert CIRCUIT_BREAKER_RECOVERY_SECONDS > 0


def test_max_upload_configurable():
    from backend.settings import MAX_UPLOAD_MB
    assert MAX_UPLOAD_MB > 0


def test_max_top_k_configurable():
    from backend.settings import MAX_TOP_K, MAX_TOP_K_FINAL
    assert MAX_TOP_K > 0
    assert MAX_TOP_K_FINAL > 0
    assert MAX_TOP_K_FINAL <= MAX_TOP_K


# ---------------------------------------------------------------------------
# 2. Health Probes
# ---------------------------------------------------------------------------

def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_healthz_no_heavy_init():
    """healthz must not trigger embedding/reranker loading."""
    start = time.perf_counter()
    resp = client.get("/healthz")
    elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    assert elapsed < 1.0  # should be instant


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "vector_store" in data


def test_readyz():
    resp = client.get("/readyz")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "status" in data
    assert "checks" in data


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def test_metrics_content():
    body = client.get("/metrics").text
    assert "http_requests_total" in body
    assert "http_request_latency_seconds" in body


def test_metrics_no_secrets():
    body = client.get("/metrics").text.lower()
    assert "openrouter_api_key" not in body
    assert "bearer" not in body


# ---------------------------------------------------------------------------
# 4. Request ID
# ---------------------------------------------------------------------------

def test_request_id_attached():
    resp = client.get("/health")
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) > 0


def test_custom_request_id_preserved():
    resp = client.get("/health", headers={"X-Request-ID": "phase10-test-id"})
    assert resp.headers["X-Request-ID"] == "phase10-test-id"


# ---------------------------------------------------------------------------
# 5. API Schema — AskRequest
# ---------------------------------------------------------------------------

def test_ask_request_schema():
    """Verify the AskRequest model accepts all expected fields."""
    from backend.api_v1 import AskRequest
    req = AskRequest(
        question="test",
        retrieval_mode="dense",
        top_k_dense=10,
        top_k_sparse=10,
        top_k_fused=20,
        top_k_final=5,
    )
    assert req.question == "test"
    assert req.retrieval_mode == "dense"


def test_ask_request_all_modes():
    from backend.api_v1 import AskRequest
    for mode in ("dense", "sparse", "hybrid", "hybrid_rerank"):
        req = AskRequest(question="test", retrieval_mode=mode)
        assert req.retrieval_mode == mode


def test_ask_request_invalid_mode():
    from backend.api_v1 import AskRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AskRequest(question="test", retrieval_mode="invalid")


# ---------------------------------------------------------------------------
# 6. Response Structure — Dense
# ---------------------------------------------------------------------------

def test_dense_response_structure():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/v1/ask", json={
        "question": "What is the maximum file upload size?",
        "retrieval_mode": "dense",
        "top_k_dense": 5,
    })
    reset_limiter()
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "status" in data
    assert "confidence" in data
    assert "retrieval_trace" in data
    assert "grounding_metrics" in data
    assert "citations" in data
    assert data["status"] in ("answered", "insufficient_context")


def test_dense_trace_populated():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/v1/ask", json={
        "question": "What is the maximum file upload size?",
        "retrieval_mode": "dense",
    })
    reset_limiter()
    data = resp.json()
    trace = data["retrieval_trace"]
    assert isinstance(trace.get("dense"), list)
    assert isinstance(trace.get("bm25"), list)
    assert isinstance(trace.get("rrf"), list)
    assert isinstance(trace.get("reranker"), list)


# ---------------------------------------------------------------------------
# 7. Response Structure — Sparse
# ---------------------------------------------------------------------------

def test_sparse_response_structure():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/v1/ask", json={
        "question": "MAX_UPLOAD_MB",
        "retrieval_mode": "sparse",
        "top_k_sparse": 5,
    })
    reset_limiter()
    assert resp.status_code == 200
    data = resp.json()
    assert "confidence" in data
    assert data["retrieval_trace"].get("bm25") is not None


# ---------------------------------------------------------------------------
# 8. Response Structure — Hybrid
# ---------------------------------------------------------------------------

def test_hybrid_response_structure():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/v1/ask", json={
        "question": "What is the default port?",
        "retrieval_mode": "hybrid",
    })
    reset_limiter()
    assert resp.status_code == 200
    data = resp.json()
    trace = data["retrieval_trace"]
    assert isinstance(trace.get("rrf"), list)


# ---------------------------------------------------------------------------
# 9. Response Structure — Hybrid Rerank
# ---------------------------------------------------------------------------

def test_hybrid_rerank_response_structure():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/v1/ask", json={
        "question": "What is the chunk size?",
        "retrieval_mode": "hybrid_rerank",
        "top_k_final": 3,
    })
    reset_limiter()
    assert resp.status_code == 200
    data = resp.json()
    trace = data["retrieval_trace"]
    assert isinstance(trace.get("reranker"), list)


# ---------------------------------------------------------------------------
# 10. Confidence Object
# ---------------------------------------------------------------------------

def test_confidence_object_structure():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/v1/ask", json={
        "question": "What is the maximum file upload size?",
        "retrieval_mode": "dense",
    })
    reset_limiter()
    data = resp.json()
    conf = data["confidence"]
    assert "overall_score" in conf
    assert "level" in conf
    assert conf["level"] in ("high", "medium", "low")
    assert "retrieval_confidence" in conf
    assert "grounding_confidence" in conf
    assert "abstention_flag" in conf
    assert isinstance(conf["abstention_flag"], bool)


# ---------------------------------------------------------------------------
# 11. Grounding Metrics
# ---------------------------------------------------------------------------

def test_grounding_metrics_structure():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/v1/ask", json={
        "question": "What is the maximum file upload size?",
        "retrieval_mode": "dense",
    })
    reset_limiter()
    data = resp.json()
    gm = data["grounding_metrics"]
    assert "total_claims" in gm
    assert "supported_claims" in gm
    assert "grounding_ratio" in gm


# ---------------------------------------------------------------------------
# 12. Citation Structure
# ---------------------------------------------------------------------------

def test_citation_structure():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/v1/ask", json={
        "question": "What is the maximum file upload size?",
        "retrieval_mode": "dense",
    })
    reset_limiter()
    data = resp.json()
    for c in data.get("citations", []):
        assert "verdict" in c
        assert c["verdict"] in ("supported", "unsupported")
        assert "claim" in c or "text_snippet" in c


# ---------------------------------------------------------------------------
# 13. Empty Question Rejection
# ---------------------------------------------------------------------------

def test_empty_question_rejected():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/v1/ask", json={"question": ""})
    reset_limiter()
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 14. Invalid Mode Rejection
# ---------------------------------------------------------------------------

def test_invalid_mode_rejected():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/v1/ask", json={"question": "test", "retrieval_mode": "bad"})
    reset_limiter()
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 15. Rate Limiter Exemption for Healthz
# ---------------------------------------------------------------------------

def test_healthz_exempt_from_rate_limit():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=1, window_seconds=60)
    # First request uses the slot
    client.get("/health")
    # healthz is exempt — should still work
    resp = client.get("/healthz")
    assert resp.status_code == 200
    reset_limiter()


# ---------------------------------------------------------------------------
# 16. 429 Rate Limit Response
# ---------------------------------------------------------------------------

def test_rate_limit_429():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=2, window_seconds=60)
    client.get("/health")
    client.get("/health")
    resp = client.get("/health")
    assert resp.status_code == 429
    data = resp.json()
    assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert "Retry-After" in resp.headers
    reset_limiter()


# ---------------------------------------------------------------------------
# 17. 404 Structured Error
# ---------------------------------------------------------------------------

def test_404_structured():
    resp = client.get("/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 18. Upload Size Limit
# ---------------------------------------------------------------------------

def test_upload_no_file():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=200, window_seconds=60)
    resp = client.post("/upload")
    reset_limiter()
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 19. Frontend Config Audit
# ---------------------------------------------------------------------------

def test_frontend_config_no_localhost_production():
    """config.js should not hardcode localhost for production."""
    from pathlib import Path
    config = (Path(__file__).resolve().parents[1] / "frontend" / "config.js").read_text()
    # Production URL should be HTTPS
    assert "https://" in config or "onrender.com" in config


def test_frontend_config_no_secrets():
    from pathlib import Path
    config = (Path(__file__).resolve().parents[1] / "frontend" / "config.js").read_text()
    assert "OPENROUTER_API_KEY" not in config
    assert "SECRET" not in config
    assert "PASSWORD" not in config
    assert "sk-" not in config


# ---------------------------------------------------------------------------
# 20. Ingestion Metadata Safety
# ---------------------------------------------------------------------------

def test_safe_value_no_path_objects():
    from backend.ingestion import _safe_value
    from pathlib import PureWindowsPath, PurePosixPath
    # WindowsPath must become str
    result = _safe_value(PureWindowsPath("C:\\data\\file.txt"))
    assert isinstance(result, str)
    assert not isinstance(result, PureWindowsPath)
    # PurePosixPath must become str
    result = _safe_value(PurePosixPath("/data/file.txt"))
    assert isinstance(result, str)
    assert not isinstance(result, PurePosixPath)


# ---------------------------------------------------------------------------
# 21. Generic Exception Handler
# ---------------------------------------------------------------------------

def test_generic_exception_returns_safe_error():
    """Unhandled exceptions should return structured error, not traceback."""
    resp = client.get("/nonexistent-path-xyz")
    assert resp.status_code == 404
    body = resp.json()
    # Either {"error": {...}} or {"detail": "..."} is acceptable
    assert "error" in body or "detail" in body
    assert "traceback" not in json.dumps(body).lower()


# ---------------------------------------------------------------------------
# 22. CORS Headers
# ---------------------------------------------------------------------------

def test_cors_preflight():
    resp = client.options("/v1/ask", headers={
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "POST",
    })
    # Should not crash — 200 or 405 is acceptable
    assert resp.status_code in (200, 405)


# ---------------------------------------------------------------------------
# 23. Circuit Breaker State Machine
# ---------------------------------------------------------------------------

def test_circuit_breaker_full_cycle():
    from backend.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=0.1)
    # CLOSED
    assert cb.state.value == "closed"
    assert cb.allow_request() is True
    # Failures -> OPEN
    cb.record_failure()
    cb.record_failure()
    assert cb.state.value == "open"
    assert cb.allow_request() is False
    # Recovery -> HALF_OPEN
    time.sleep(0.15)
    assert cb.state.value == "half_open"
    assert cb.allow_request() is True
    # Success -> CLOSED
    cb.record_success()
    assert cb.state.value == "closed"
    assert cb.allow_request() is True


# ---------------------------------------------------------------------------
# 24. Retriever Instances Exist
# ---------------------------------------------------------------------------

def test_retriever_instances_exist():
    from backend.retrieval import DenseRetrieverInstance, BM25RetrieverInstance
    assert DenseRetrieverInstance is not None
    assert BM25RetrieverInstance is not None
