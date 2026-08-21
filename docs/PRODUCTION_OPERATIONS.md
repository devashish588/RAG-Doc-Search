# Production Operations Guide

## Architecture

```
Client → CORS → Rate Limiter → Request ID → FastAPI Router
                                              ├── /v1/ask    → Retrieval → LLM (circuit breaker) → Verification → Confidence → Response
                                              ├── /upload    → Ingestion Queue → Chroma + BM25
                                              ├── /healthz   → Liveness (no-op)
                                              ├── /readyz    → Chroma + BM25 + Disk checks
                                              └── /metrics   → Prometheus text exposition
```

## Resource Requirements

| Component         | Memory   | Notes                          |
|-------------------|----------|--------------------------------|
| Application base  | ~728 MB  | FastAPI + Chroma + embeddings  |
| Dense model       | ~50 MB   | BAAI/bge-small-en-v1.5         |
| BM25              | ~10 MB   | rank_bm25 in-memory            |
| TinyBERT reranker | ~786 MB  | ONNX cross-encoder             |
| MiniLM reranker   | ~1143 MB | Too heavy for 1 GB instances   |

**512 MB is NOT a supported configuration.** Peak memory with TinyBERT (~786 MB) exceeds it. MiniLM (~1143 MB) is completely out of reach.

| Tier     | Memory | Recommendation                        |
|----------|--------|---------------------------------------|
| Minimum  | 1 GB   | Dense/BM25/hybrid only                |
| Recommended | 2 GB | hybrid_rerank with TinyBERT           |
| Premium  | 4 GB   | MiniLM reranker or multi-user         |

## Recommended Render Plan

- **Starter** (1 GB) — Dense/BM25/hybrid modes
- **Standard** (2 GB) — hybrid_rerank with TinyBERT
- Free tier (512 MB) is **not recommended** — the application will OOM on ingestion or reranking.

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `production` or `development` |
| `LOG_LEVEL` | `INFO` | Python log level |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window per IP |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window size |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `3` | Failures before OPEN |
| `CIRCUIT_BREAKER_RECOVERY_SECONDS` | `30` | Recovery timeout |
| `MAX_UPLOAD_MB` | `25` | Upload size limit |
| `MAX_TOP_K` | `50` | Max dense/sparse top_k |
| `MAX_TOP_K_FINAL` | `20` | Max reranker top_k |
| `OPENROUTER_API_KEY` | (empty) | LLM API key |

## Health Probes

| Endpoint | Purpose | Weight | Response |
|----------|---------|--------|----------|
| `/health` | Legacy health | None | `{"status": "ok"}` |
| `/healthz` | Liveness | Extremely lightweight | `{"status": "ok"}` |
| `/readyz` | Readiness | Checks Chroma + BM25 + disk | `{"status": "ready", "checks": {...}}` |

`/healthz` must **never** load embeddings, reranker, or call OpenRouter.

## Metrics

Prometheus-compatible metrics at `/metrics`:

- `http_requests_total` — Total HTTP requests (labels: endpoint, method, status)
- `http_request_latency_seconds` — Request latency histogram
- `rag_retrieval_latency_seconds` — Retrieval latency (labels: retrieval_mode)
- `rag_generation_latency_seconds` — LLM generation latency
- `rag_reranker_latency_seconds` — Reranker latency
- `rag_verification_latency_seconds` — Citation verification latency
- `rag_confidence_score` — Confidence distribution (labels: level)
- `rag_abstentions_total` — Abstention counter
- `rag_ingestion_total` — Ingestion operations (labels: status)
- `rag_errors_total` — Error counter (labels: endpoint, code)

Labels are **bounded** — never use question text, document content, or user IDs.

## Logging

Structured JSON logging for every request (except `/healthz` and `/metrics`):

```json
{
  "timestamp": "2025-01-15T10:30:00+0000",
  "request_id": "abc123",
  "endpoint": "/v1/ask",
  "method": "POST",
  "status_code": 200,
  "duration_ms": 123.4,
  "retrieval_mode": "dense",
  "confidence_level": "high",
  "abstention": false
}
```

**Never logged:** API keys, Authorization headers, passwords, tokens, document content, raw user questions as labels.

## Rate Limiting

In-memory sliding-window per-IP. Returns HTTP 429 with `Retry-After` header.

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
Retry-After: 12
```

This is **single-instance protection only**. For multi-instance deployments, use a shared rate limiter (Redis, API gateway). Do not add Redis in Phase 9.

## Circuit Breaker

Wraps the OpenRouter LLM call only (not the entire `/v1/ask` endpoint).

| State | Behavior |
|-------|----------|
| CLOSED | Normal LLM calls |
| OPEN | Skip LLM, return fallback answer |
| HALF_OPEN | Allow one probe request |

Default: 3 failures → OPEN, 30 seconds → HALF_OPEN.

## Chroma Persistence

Chroma data is stored at `data/chroma_db/`. The Render Blueprint mounts a persistent disk at this path.

## BM25 Persistence

BM25 index is stored at `data/bm25_index/`. Rebuilt from Chroma on first access if missing.

## Ingestion

1. Upload file → validate extension + size
2. Save to `data/uploads/` with content hash
3. Duplicate detection by content hash
4. Background queue processes ingestion
5. Text extraction → chunking → embedding → Chroma + BM25 index

**Metadata safety:** All values passed to Chroma are converted via `_safe_value()` to ensure str/int/float/bool — no `WindowsPath` or `Path` objects.

## Security

- CORS: Configurable via `CORS_ORIGINS`. Production should **not** use `*`.
- Rate limiting: Per-IP sliding window
- Upload limits: Configurable max file size
- Error handling: Never exposes tracebacks, internal paths, API keys, or environment variables
- Secrets: `.env` is git-ignored; `.env.example` contains placeholders only

## CORS

```bash
# Development (allow all)
CORS_ORIGINS=*

# Production (specific origins)
CORS_ORIGINS=https://yourfrontend.pages.dev,https://yourdomain.com
```

## Docker

```bash
docker compose up --build
```

- Python 3.12 slim base
- Non-root user
- HEALTHCHECK against `/healthz`
- Data volume persisted at `./data`

## Deployment

1. Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY`
2. `docker compose up -d` or deploy to Render via Blueprint
3. Verify: `curl http://localhost:9826/healthz`
4. Verify: `curl http://localhost:9826/readyz`
5. Verify: `curl http://localhost:9826/metrics`
6. Upload a document via `/upload`
7. Query via `/v1/ask`

## Rollback

1. Revert to previous git tag: `git checkout v1.8-evaluation`
2. Rebuild and redeploy
3. Chroma data is backward-compatible within the same ChromaDB version

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM on startup | Insufficient memory | Upgrade to 1 GB+ plan |
| 429 errors | Rate limit exceeded | Increase `RATE_LIMIT_REQUESTS` or reduce traffic |
| Circuit OPEN | OpenRouter failures | Check API key, model availability |
| 503 on /readyz | Chroma/BM25 unhealthy | Check `data/` directory, disk space |
| Slow ingestion | Large documents | Reduce chunk size or document size |
