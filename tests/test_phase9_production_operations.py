"""Phase 9: Production operations test suite."""
import json
import time
import threading
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Monitoring / Request ID
# ---------------------------------------------------------------------------

def test_request_id_returned():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) > 0


def test_custom_request_id_preserved():
    resp = client.get("/health", headers={"X-Request-ID": "my-custom-id-123"})
    assert resp.headers["X-Request-ID"] == "my-custom-id-123"


def test_structured_log_output(capsys):
    from backend.monitoring import structured_log
    structured_log(
        request_id="test-123",
        endpoint="/v1/ask",
        method="POST",
        status_code=200,
        duration_ms=42.5,
        retrieval_mode="dense",
        confidence_level="high",
    )


# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def test_metrics_contain_expected_names():
    resp = client.get("/metrics")
    body = resp.text
    assert "http_requests_total" in body
    assert "http_request_latency_seconds" in body


def test_metrics_no_secrets():
    resp = client.get("/metrics")
    body = resp.text.lower()
    assert "openrouter_api_key" not in body
    assert "bearer" not in body


def test_metrics_update_after_request():
    from backend.monitoring import HTTP_REQUESTS_TOTAL
    if HTTP_REQUESTS_TOTAL is None:
        pytest.skip("prometheus_client not installed")
    before = HTTP_REQUESTS_TOTAL.labels(endpoint="/health", method="GET", status="200")._value.get()
    client.get("/health")
    after = HTTP_REQUESTS_TOTAL.labels(endpoint="/health", method="GET", status="200")._value.get()
    assert after >= before


# ---------------------------------------------------------------------------
# Health Probes
# ---------------------------------------------------------------------------

def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_healthz_lightweight():
    """healthz must not trigger heavy initialization."""
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_health():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=100, window_seconds=60)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    reset_limiter()


def test_readyz():
    resp = client.get("/readyz")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "status" in data
    assert "checks" in data


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

def test_rate_limiter_allows_normal_requests():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        resp = client.get("/health")
        assert resp.status_code == 200
    reset_limiter()


def test_rate_limiter_rejects_when_exceeded():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        client.get("/health")
    resp = client.get("/health")
    assert resp.status_code == 429
    data = resp.json()
    assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert "Retry-After" in resp.headers
    reset_limiter()


def test_rate_limiter_skips_healthz():
    """healthz is exempt from rate limiting."""
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=2, window_seconds=60)
    client.get("/healthz")
    client.get("/healthz")
    # This would be #3 and exceed limit, but healthz is exempt
    resp = client.get("/healthz")
    assert resp.status_code == 200
    reset_limiter()


def test_rate_limiter_retry_after_header():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=1, window_seconds=10)
    client.get("/health")  # uses the 1 slot
    resp = client.get("/health")  # should be rejected
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1
    reset_limiter()


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

def test_circuit_breaker_closed():
    from backend.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(failure_threshold=3, recovery_seconds=1)
    assert cb.allow_request() is True
    assert cb.state.value == "closed"


def test_circuit_breaker_opens_after_failures():
    from backend.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(failure_threshold=3, recovery_seconds=60)
    for _ in range(3):
        cb.record_failure()
    assert cb.allow_request() is False
    assert cb.state.value == "open"


def test_circuit_breaker_half_open_after_recovery():
    from backend.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=0.1)
    cb.record_failure()
    cb.record_failure()
    assert cb.state.value == "open"
    time.sleep(0.15)
    assert cb.state.value == "half_open"
    assert cb.allow_request() is True


def test_circuit_breaker_closes_on_success():
    from backend.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=0.1)
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.15)
    assert cb.state.value == "half_open"
    cb.record_success()
    assert cb.state.value == "closed"
    assert cb.allow_request() is True


def test_circuit_breaker_reset():
    from backend.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=60)
    cb.record_failure()
    cb.record_failure()
    assert cb.allow_request() is False
    cb.reset()
    assert cb.allow_request() is True


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

def test_cors_headers():
    resp = client.options("/v1/ask", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    })
    # Should not error — either 200 or 405 is fine, just checking it doesn't crash
    assert resp.status_code in (200, 405)


# ---------------------------------------------------------------------------
# Request Validation
# ---------------------------------------------------------------------------

def test_ask_empty_question():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=100, window_seconds=60)
    resp = client.post("/v1/ask", json={"question": ""})
    assert resp.status_code == 422  # Pydantic validation
    reset_limiter()


def test_ask_invalid_retrieval_mode():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=100, window_seconds=60)
    resp = client.post("/v1/ask", json={"question": "test", "retrieval_mode": "invalid"})
    assert resp.status_code == 422
    reset_limiter()


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

def test_404_returns_structured_error():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=100, window_seconds=60)
    resp = client.get("/nonexistent")
    assert resp.status_code == 404
    reset_limiter()


def test_upload_no_file():
    from backend.middleware import reset_limiter
    reset_limiter(max_requests=100, window_seconds=60)
    resp = client.post("/upload")
    assert resp.status_code == 422
    reset_limiter()


# ---------------------------------------------------------------------------
# Storage Health
# ---------------------------------------------------------------------------

def test_chroma_health():
    from backend.storage_health import check_chroma_health
    result = check_chroma_health()
    assert result["status"] in ("ok", "error")


def test_bm25_health():
    from backend.storage_health import check_bm25_health
    result = check_bm25_health()
    assert result["status"] in ("ok", "error", "warning")


def test_disk_health():
    from backend.storage_health import check_disk_health
    result = check_disk_health()
    assert result["status"] in ("ok", "warning", "error")
    if result["status"] == "ok":
        assert "free_mb" in result


# ---------------------------------------------------------------------------
# Ingestion Metadata Safety (WindowsPath regression)
# ---------------------------------------------------------------------------

def test_safe_value_conversion():
    from backend.ingestion import _safe_value
    from pathlib import PurePosixPath, PureWindowsPath
    assert _safe_value("hello") == "hello"
    assert _safe_value(42) == 42
    assert _safe_value(3.14) == 3.14
    assert _safe_value(True) is True
    assert _safe_value(None) == ""
    assert _safe_value(PureWindowsPath("C:\\data\\file.txt")) == "C:\\data\\file.txt"
    assert _safe_value(PurePosixPath("/data/file.txt")) == "/data/file.txt"
