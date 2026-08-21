# Phase 9 Release Checklist

## Monitoring & Observability
- [x] Structured JSON logging (`backend/monitoring.py`)
- [x] Request ID middleware (X-Request-ID)
- [x] Prometheus metrics (`/metrics` endpoint)
- [x] Bounded metric labels (no question/user content)
- [x] No secrets in logs or metrics

## Rate Limiting
- [x] In-memory sliding-window rate limiter (`backend/middleware.py`)
- [x] HTTP 429 with Retry-After header
- [x] Health probes exempt from rate limiting

## Circuit Breaker
- [x] OpenRouter circuit breaker (`backend/circuit_breaker.py`)
- [x] CLOSED → OPEN → HALF_OPEN states
- [x] Configurable failure threshold + recovery timeout

## Health Probes
- [x] `/healthz` liveness (lightweight)
- [x] `/readyz` readiness (Chroma + BM25 + disk)
- [x] `/health` legacy endpoint preserved
- [x] `/metrics` Prometheus endpoint

## Storage Health
- [x] Chroma health check (`storage_health.py`)
- [x] BM25 health check
- [x] Disk space check
- [x] Index consistency check

## Ingestion
- [x] Fresh ingestion works (PDF, TXT, Markdown)
- [x] WindowsPath metadata regression fixed (`_safe_value`)
- [x] Duplicate detection
- [x] Upload size limit (MAX_UPLOAD_MB=25)

## Request Validation
- [x] MAX_TOP_K=50 enforced
- [x] MAX_TOP_K_FINAL=20 enforced
- [x] Invalid retrieval modes rejected (422)
- [x] Empty questions rejected (422)

## CORS
- [x] Configurable via CORS_ORIGINS
- [x] Development default: *
- [x] Production: explicit origins required

## Error Handling
- [x] No Python tracebacks exposed
- [x] No internal filesystem paths
- [x] No API keys in responses
- [x] Structured error codes (RATE_LIMIT_EXCEEDED, LLM_UNAVAILABLE, etc.)

## Docker
- [x] Dockerfile (Python 3.12 slim, non-root)
- [x] docker-compose.yml (2 GB memory limit)
- [x] HEALTHCHECK against /healthz
- [x] .env.example with placeholders

## Tests
- [x] 236 existing tests pass (no regressions)
- [x] 29 Phase 9 tests pass
- [x] Total: 265 tests

## Documentation
- [x] docs/PRODUCTION_OPERATIONS.md
- [x] docs/phase9_release_checklist.md (this file)

## Security
- [x] .env is git-ignored
- [x] .env.example contains placeholders only
- [x] No secrets in logs
- [x] No secrets in metrics
- [x] No secrets in code comments

## Resource Requirements
- [x] 512 MB: NOT supported (documented)
- [x] 1 GB: minimum practical
- [x] 2 GB: recommended for hybrid_rerank

## Historical Tag Integrity
- [x] v1.0-baseline unchanged
- [x] v1.1-foundation unchanged
- [x] v1.2-ingestion unchanged
- [x] v1.3-dense unchanged
- [x] v1.4-bm25 unchanged
- [x] v1.5-hybrid unchanged
- [x] v1.6-reranker unchanged
- [x] v1.7-grounding unchanged
- [x] v1.8-evaluation unchanged
