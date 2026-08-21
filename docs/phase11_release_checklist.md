# Phase 11 Release Checklist

## Configuration
- [x] Render configuration audited
- [x] GitHub Pages configuration audited
- [x] No secrets in render.yaml
- [x] No secrets in frontend/
- [x] No secrets in .github/
- [x] No secrets in Dockerfile
- [x] CORS configurable via env var
- [x] .env is gitignored

## Health Probes
- [x] GET /health → 200
- [x] GET /healthz → 200 (exempt from rate limiting)
- [x] GET /readyz → 200/503 (exempt from rate limiting)
- [x] GET /metrics → 200 (Prometheus text format)

## Four-Mode API
- [x] Dense mode → 200
- [x] BM25/sparse mode → 200
- [x] Hybrid mode → 200
- [x] Hybrid rerank mode → 200

## Frontend E2E
- [x] config.js auto-detects environment
- [x] No localhost in production URL
- [x] No secrets in frontend
- [x] CORS allows GitHub Pages origin (configurable)

## Rate Limiting
- [x] 429 returned when limit exceeded
- [x] Retry-After header present
- [x] X-RateLimit-Limit header present
- [x] X-RateLimit-Remaining header present
- [x] /healthz exempt from rate limiting
- [x] /readyz exempt from rate limiting
- [x] /metrics exempt from rate limiting

## Circuit Breaker
- [x] CLOSED → OPEN → HALF_OPEN → CLOSED cycle verified
- [x] OpenRouter failure does not crash API
- [x] Fallback message returned when OPEN

## Monitoring
- [x] Structured JSON logs with request_id, timestamp, endpoint, status, duration
- [x] Prometheus metrics at /metrics
- [x] No secrets in logs or metrics
- [x] Request ID attached to every response

## Security
- [x] No API keys committed
- [x] No hardcoded secrets
- [x] .env gitignored
- [x] CORS configurable
- [x] Generic exception handler hides tracebacks

## Docker
- [x] docker build succeeds
- [x] Container runs as non-root
- [x] HEALTHCHECK configured
- [x] Exposes correct port
- [x] 512 MB unsupported documented
- [x] 1 GB minimum practical documented
- [x] 2 GB recommended documented

## Keep-Warm
- [x] GitHub Actions workflow created
- [x] Uses /healthz endpoint
- [x] No internal self-ping
- [x] No LLM call in cron
- [x] Schedule: */5 * * * *
- [x] Timeout configured
- [x] RENDER_HEALTH_URL variable documented

## Cold-Start Measurement
- [x] Measurement script created (evals/measure_production_latency.py)
- [x] Classification methodology documented
- [x] Not fabricated — requires live Render to execute

## Documentation
- [x] docs/PHASE11_DEPLOYMENT_CHECKLIST.md
- [x] docs/PRODUCTION_OPERATIONS.md updated
- [x] docs/phase11_release_checklist.md (this file)
- [x] docs/PHASE11_CLOSURE_REPORT.md

## Tests
- [x] tests/test_phase11_production_ops.py — 37 tests
- [x] tests/conftest.py — rate limiter reset fixture
- [x] Full regression: 341 passed / 0 failed

## Historical Tags
- [x] v1.0-baseline unchanged
- [x] v1.1-foundation unchanged
- [x] v1.2-ingestion unchanged
- [x] v1.3-dense unchanged
- [x] v1.4-bm25 unchanged
- [x] v1.5-hybrid unchanged
- [x] v1.6-reranker unchanged
- [x] v1.7-grounding unchanged
- [x] v1.8-evaluation unchanged
- [x] v1.9-production unchanged
- [x] v1.10-production unchanged

## Release
- [ ] Commit: release(ops): Phase 11 production operations and keep-warm
- [ ] Tag: v1.11-production-ops
- [ ] Push branch
- [ ] Push tag
- [ ] Verify local tag
- [ ] Verify remote tag
