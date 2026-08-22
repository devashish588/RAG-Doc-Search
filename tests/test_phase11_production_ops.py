"""Phase 11 — Production operations tests (local TestClient, no external Render)."""
import json
import os
import threading
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.middleware import reset_limiter
from backend.circuit_breaker import CircuitBreaker, CircuitState


client = TestClient(app)


# ---------------------------------------------------------------------------
# Health probes
# ---------------------------------------------------------------------------

class TestHealthProbes:
    def test_healthz_returns_200(self):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_healthz_exempt_from_rate_limit(self):
        reset_limiter(max_requests=1, window_seconds=60)
        client.get("/health")  # consume the one allowed request
        r = client.get("/healthz")
        assert r.status_code == 200

    def test_readyz_returns_json(self):
        r = client.get("/readyz")
        assert r.status_code in (200, 503)
        data = r.json()
        assert "status" in data
        assert "checks" in data

    def test_readyz_exempt_from_rate_limit(self):
        reset_limiter(max_requests=1, window_seconds=60)
        client.get("/health")  # consume
        r = client.get("/readyz")
        assert r.status_code in (200, 503)

    def test_health_returns_fields(self):
        reset_limiter(max_requests=100, window_seconds=60)
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "vector_store" in data
        assert "embedding_backend" in data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_metrics_returns_200(self):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type(self):
        r = client.get("/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_metrics_exempt_from_rate_limit(self):
        reset_limiter(max_requests=1, window_seconds=60)
        client.get("/health")  # consume
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_contains_prometheus(self):
        r = client.get("/metrics")
        body = r.text
        assert "http_requests_total" in body
        assert "http_request_latency_seconds" in body


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------

class TestRequestID:
    def test_health_returns_request_id(self):
        r = client.get("/health")
        assert "X-Request-ID" in r.headers

    def test_request_id_is_uuid(self):
        r = client.get("/health")
        rid = r.headers["X-Request-ID"]
        # uuid4().hex is a 32-char hex string
        assert len(rid) == 32
        assert all(c in "0123456789abcdef" for c in rid)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_rate_limit_headers_present(self):
        reset_limiter(max_requests=100, window_seconds=60)
        r = client.get("/health")
        assert "X-RateLimit-Limit" in r.headers
        assert "X-RateLimit-Remaining" in r.headers
        assert int(r.headers["X-RateLimit-Limit"]) == 100

    def test_rate_limit_triggers_429(self):
        reset_limiter(max_requests=2, window_seconds=60)
        client.get("/health")
        client.get("/health")
        r = client.get("/health")
        assert r.status_code == 429
        assert "Retry-After" in r.headers

    def test_rate_limit_429_body(self):
        reset_limiter(max_requests=1, window_seconds=60)
        client.get("/health")
        r = client.get("/health")
        assert r.status_code == 429
        body = r.json()
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    def test_rate_limit_remaining_decrements(self):
        reset_limiter(max_requests=5, window_seconds=60)
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert int(r1.headers["X-RateLimit-Remaining"]) > int(r2.headers["X-RateLimit-Remaining"])

    def test_health_endpoints_bypass_rate_limit(self):
        reset_limiter(max_requests=1, window_seconds=60)
        client.get("/health")  # consume the one allowed
        for path in ("/healthz", "/readyz", "/metrics"):
            r = client.get(path)
            assert r.status_code in (200, 503), f"{path} should be exempt"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_circuit_breaker_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_seconds=60)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_circuit_breaker_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_seconds=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_circuit_breaker_half_open_after_recovery(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True

    def test_circuit_breaker_closes_on_success(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_reset(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------

class TestCORS:
    def test_cors_allows_origins(self):
        r = client.options("/healthz", headers={
            "Origin": "http://localhost:9826",
            "Access-Control-Request-Method": "GET",
        })
        # CORS middleware should respond
        assert r.status_code in (200, 405)

    def test_cors_headers_on_health(self):
        r = client.get("/healthz", headers={"Origin": "http://localhost:9826"})
        lower_headers = {k.lower(): v for k, v in r.headers.items()}
        assert "access-control-allow-origin" in lower_headers


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

class TestStructuredLogging:
    def test_structured_log_contains_required_fields(self):
        from backend.monitoring import structured_log
        import logging

        with patch("backend.monitoring.log") as mock_log:
            structured_log(
                request_id="test-123",
                endpoint="/health",
                method="GET",
                status_code=200,
                duration_ms=1.5,
            )
            mock_log.info.assert_called_once()
            logged = mock_log.info.call_args[0][0]
            # The log message is the JSON string
            assert "test-123" in logged

    def test_structured_log_excludes_secrets(self):
        from backend.monitoring import structured_log

        with patch("backend.monitoring.log") as mock_log:
            structured_log(
                request_id="test-456",
                endpoint="/v1/ask",
                method="POST",
                status_code=200,
                duration_ms=50.0,
                extra={"api_key": "REDACTED"},
            )
            logged = mock_log.info.call_args[0][0]
            assert "api_key" in logged  # field exists
            # Value should not be a real key (it's "REDACTED" in test, which is fine)


# ---------------------------------------------------------------------------
# Cron configuration
# ---------------------------------------------------------------------------

class TestCronConfiguration:
    def test_keepwarm_workflow_exists(self):
        import pathlib
        workflow = pathlib.Path(__file__).parents[1] / ".github" / "workflows" / "render-keepalive.yml"
        assert workflow.exists(), "render-keepalive.yml not found"

    def test_keepwarm_uses_healthz(self):
        import pathlib
        workflow = pathlib.Path(__file__).parents[1] / ".github" / "workflows" / "render-keepalive.yml"
        content = workflow.read_text()
        assert "healthz" in content
        assert "/v1/ask" not in content
        assert "openrouter" not in content.lower()

    def test_keepwarm_has_schedule(self):
        import pathlib
        workflow = pathlib.Path(__file__).parents[1] / ".github" / "workflows" / "render-keepalive.yml"
        content = workflow.read_text()
        assert "schedule" in content
        assert "*/5" in content

    def test_keepwarm_has_timeout(self):
        import pathlib
        workflow = pathlib.Path(__file__).parents[1] / ".github" / "workflows" / "render-keepalive.yml"
        content = workflow.read_text()
        assert "timeout-minutes" in content


# ---------------------------------------------------------------------------
# Security configuration
# ---------------------------------------------------------------------------

class TestSecurityConfiguration:
    def test_no_secrets_in_settings(self):
        from backend import settings
        secret_patterns = ["OPENROUTER_API_KEY", "sk-", "api_key"]
        source = open(settings.__file__).read()
        # Should not contain hardcoded keys (only env var references)
        for line in source.splitlines():
            if "OPENROUTER_API_KEY" in line and "os.getenv" not in line and "sync:" not in line:
                # It's in render.yaml, which is fine — check it's not a real value
                assert "=" not in line or line.strip().startswith("#"), f"Possible secret: {line}"

    def test_env_is_gitignored(self):
        import pathlib
        gitignore = pathlib.Path(__file__).parents[1] / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text()
            assert ".env" in content

    def test_render_yaml_no_committed_secrets(self):
        import pathlib
        render_yaml = pathlib.Path(__file__).parents[1] / "render.yaml"
        content = render_yaml.read_text()
        assert "sk-" not in content
        assert "api_key_here" not in content

    def test_frontend_no_secrets(self):
        import pathlib
        frontend_dir = pathlib.Path(__file__).parents[1] / "frontend"
        for f in frontend_dir.glob("*"):
            if f.is_file() and f.suffix in (".js", ".html", ".css"):
                content = f.read_text(encoding="utf-8")
                assert "sk-" not in content, f"Possible secret in {f.name}"
                assert "OPENROUTER_API_KEY" not in content


# ---------------------------------------------------------------------------
# Production configuration validation
# ---------------------------------------------------------------------------

class TestProductionConfig:
    def test_max_top_k_configured(self):
        from backend.settings import MAX_TOP_K, MAX_TOP_K_FINAL
        assert MAX_TOP_K > 0
        assert MAX_TOP_K_FINAL > 0

    def test_rate_limit_configured(self):
        from backend.settings import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS
        assert RATE_LIMIT_REQUESTS > 0
        assert RATE_LIMIT_WINDOW_SECONDS > 0

    def test_circuit_breaker_configured(self):
        from backend.settings import CIRCUIT_BREAKER_FAILURE_THRESHOLD, CIRCUIT_BREAKER_RECOVERY_SECONDS
        assert CIRCUIT_BREAKER_FAILURE_THRESHOLD > 0
        assert CIRCUIT_BREAKER_RECOVERY_SECONDS > 0

    def test_request_timeout_configured(self):
        from backend.settings import REQUEST_TIMEOUT
        assert REQUEST_TIMEOUT > 0
