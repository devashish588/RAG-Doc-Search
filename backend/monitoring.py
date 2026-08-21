"""Structured logging, request-ID middleware, and Prometheus metrics (Phase 9)."""
import json
import logging
import time
from typing import Optional
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
except ImportError:  # pragma: no cover
    Counter = Histogram = generate_latest = CONTENT_TYPE_LATEST = None

log = logging.getLogger("rag.structured")


# ---------------------------------------------------------------------------
# Structured JSON logger
# ---------------------------------------------------------------------------

def structured_log(
    *,
    request_id: str,
    endpoint: str,
    method: str,
    status_code: int,
    duration_ms: float,
    retrieval_mode: str = "",
    confidence_level: str = "",
    abstention: bool = False,
    extra: Optional[dict] = None,
) -> None:
    """Emit a single structured JSON log line.  Never logs secrets."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request_id": request_id,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
    }
    if retrieval_mode:
        record["retrieval_mode"] = retrieval_mode
    if confidence_level:
        record["confidence_level"] = confidence_level
    if abstention:
        record["abstention"] = True
    if extra:
        record.update(extra)
    log.info(json.dumps(record, default=str))


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

if Counter is not None:
    HTTP_REQUESTS_TOTAL = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["endpoint", "method", "status"],
    )
    HTTP_REQUEST_LATENCY = Histogram(
        "http_request_latency_seconds",
        "HTTP request latency in seconds",
        ["endpoint", "method"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    RAG_RETRIEVAL_LATENCY = Histogram(
        "rag_retrieval_latency_seconds",
        "RAG retrieval latency in seconds",
        ["retrieval_mode"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    RAG_GENERATION_LATENCY = Histogram(
        "rag_generation_latency_seconds",
        "LLM generation latency in seconds",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    RAG_RERANKER_LATENCY = Histogram(
        "rag_reranker_latency_seconds",
        "Reranker latency in seconds",
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    )
    RAG_VERIFICATION_LATENCY = Histogram(
        "rag_verification_latency_seconds",
        "Citation verification latency in seconds",
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    )
    RAG_CONFIDENCE_SCORE = Histogram(
        "rag_confidence_score",
        "Confidence score distribution",
        ["level"],
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    RAG_ABSTENTIONS_TOTAL = Counter(
        "rag_abstentions_total",
        "Total abstentions",
        ["retrieval_mode"],
    )
    RAG_INGESTION_TOTAL = Counter(
        "rag_ingestion_total",
        "Total ingestion operations",
        ["status"],
    )
    RAG_ERRORS_TOTAL = Counter(
        "rag_errors_total",
        "Total errors",
        ["endpoint", "code"],
    )
    RAG_DOCUMENTS_TOTAL = Counter(
        "rag_documents_total",
        "Total documents processed",
    )
    RAG_CHUNKS_TOTAL = Counter(
        "rag_chunks_total",
        "Total chunks indexed",
    )
else:  # pragma: no cover
    HTTP_REQUESTS_TOTAL = HTTP_REQUEST_LATENCY = None
    RAG_RETRIEVAL_LATENCY = RAG_GENERATION_LATENCY = RAG_RERANKER_LATENCY = None
    RAG_VERIFICATION_LATENCY = RAG_CONFIDENCE_SCORE = RAG_ABSTENTIONS_TOTAL = None
    RAG_INGESTION_TOTAL = RAG_ERRORS_TOTAL = RAG_DOCUMENTS_TOTAL = RAG_CHUNKS_TOTAL = None


def metrics_snapshot() -> bytes:
    """Return Prometheus text exposition."""
    if generate_latest is None:
        return b"# prometheus_client not installed\n"
    return generate_latest()


# ---------------------------------------------------------------------------
# Request-ID + structured logging middleware
# ---------------------------------------------------------------------------

class RequestMonitoringMiddleware(BaseHTTPMiddleware):
    """Attach X-Request-ID, measure latency, emit structured log + metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # --- request ID ---
        incoming_id = request.headers.get("x-request-id", "").strip()
        request_id = incoming_id if 0 < len(incoming_id) <= 128 else uuid4().hex

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            # Prometheus counters
            if HTTP_REQUESTS_TOTAL is not None:
                endpoint = request.url.path
                HTTP_REQUESTS_TOTAL.labels(
                    endpoint=endpoint, method=request.method, status=str(status_code)
                ).inc()
                HTTP_REQUEST_LATENCY.labels(endpoint=endpoint, method=request.method).observe(
                    duration_ms / 1000.0
                )

            # Structured log (skip noisy health-check probes)
            if request.url.path not in ("/healthz", "/metrics"):
                structured_log(
                    request_id=request_id,
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )

        # Attach request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
