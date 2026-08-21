# Production Operations

## Production Deployment

The backend is deployed on **Render** as a web service with persistent disk.

The frontend is deployed on **GitHub Pages** and calls the Render backend via HTTPS.

### Render Configuration

| Setting | Value |
|---------|-------|
| Plan | `standard` (1 GB min) |
| Runtime | Python 3.12 |
| Build | `pip install -r requirements.txt` |
| Start | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health | `/healthz` |
| Disk | 1 GB at `/opt/render/project/src/data` |

### GitHub Pages Configuration

- Frontend served from GitHub Pages repository settings
- `frontend/config.js` auto-detects environment and sets API URL
- No build step required — static HTML/JS/CSS
- CORS must allow the GitHub Pages origin

## Render Cold Starts

Render suspends idle web services after a period of inactivity. When suspended:

1. The next request triggers a cold start (service restart)
2. Cold starts typically take 30–120 seconds depending on dependencies
3. The service must re-import FastEmbed, ChromaDB, and BM25 indexes
4. The first request after cold start may take significantly longer than subsequent requests

**Key facts:**
- 512 MB instances are **unsupported** — memory peak with reranker is ~786 MB
- 1 GB instances can run basic modes (dense, sparse, hybrid)
- 2 GB instances are recommended for hybrid_rerank with TinyBERT

## External Keep-Warm Health Check

An external GitHub Actions workflow pings `/healthz` every 5 minutes to reduce inactivity-related cold starts.

### Configuration

| Setting | Value |
|---------|-------|
| Provider | GitHub Actions |
| Workflow | `.github/workflows/render-keepalive.yml` |
| Schedule | `*/5 * * * *` (every 5 minutes) |
| Endpoint | `$RENDER_HEALTH_URL` (repository variable) |
| Timeout | 10 seconds |
| HTTP method | GET |

### Required Setup

1. Create a repository variable `RENDER_HEALTH_URL` in GitHub Settings → Secrets and variables → Actions → Variables
2. Set the value to `https://rag-doc-search-1.onrender.com/healthz`
3. The workflow is enabled by default after the repository variable exists

### Limitations

- **Not a guarantee**: The keep-warm cron reduces cold starts but does not eliminate them
- **Render plan dependent**: Effectiveness varies by Render service plan
- **Platform behavior**: GitHub Actions scheduling may have delays
- **Monitoring only**: The cron is for monitoring/keep-warm, not strict real-time

## Health Endpoints

| Endpoint | Method | Purpose | Rate Limited |
|----------|--------|---------|-------------|
| `/health` | GET | Full health (Chroma + BM25 + disk) | Yes |
| `/healthz` | GET | Lightweight liveness probe | No (exempt) |
| `/readyz` | GET | Readiness with storage details | No (exempt) |
| `/metrics` | GET | Prometheus metrics | Yes |

## Monitoring

### Prometheus Metrics

Available at `/metrics`:
- `http_requests_total` — total requests by method, endpoint, status
- `http_request_duration_seconds` — request latency histogram
- `retrieval_total` — retrievals by mode
- `retrieval_latency_seconds` — retrieval latency by mode
- `reranker_total` — reranker invocations
- `reranker_latency_seconds` — reranker latency
- `llm_total` — LLM calls
- `llm_latency_seconds` — LLM latency
- `llm_errors_total` — LLM errors by type
- `circuit_breaker_state` — current circuit breaker state (0=closed, 1=open, 2=half-open)

### Structured Logs

JSON format with:
- `request_id` — unique per request
- `timestamp` — ISO 8601
- `endpoint` — request path
- `method` — HTTP method
- `status` — HTTP status code
- `duration_ms` — request duration
- `level` — log level

No secrets, API keys, or query content are logged.

## Rate Limiting

| Setting | Value |
|---------|-------|
| Requests | 60 per window |
| Window | 60 seconds |
| Algorithm | Sliding window, per-IP |
| Storage | In-memory (single instance) |
| Health probes | Exempt from rate limiting |

Response headers:
- `X-RateLimit-Limit` — window limit
- `X-RateLimit-Remaining` — remaining requests
- `Retry-After` — seconds until next allowed request (429 only)

## Circuit Breaker

The OpenRouter LLM calls use a circuit breaker:

| State | Behavior |
|-------|----------|
| `CLOSED` | Normal operation, requests pass through |
| `OPEN` | All requests rejected immediately, fallback message returned |
| `HALF_OPEN` | One test request allowed through to verify recovery |

| Setting | Value |
|---------|-------|
| Failure threshold | 3 consecutive failures |
| Recovery timeout | 30 seconds |

## Rollback

1. **Render dashboard**: Deploys → select previous deploy → Rollback
2. **Git revert**: `git revert <commit>` + push triggers auto-deploy
3. **Manual redeploy**: Render dashboard → Manual Deploy → redeploy

## Incident Response

1. Check `/healthz` — is the service alive?
2. Check `/metrics` — is request volume normal?
3. Check Render dashboard — logs, memory usage, disk usage
4. Check circuit breaker state — is OpenRouter failing?
5. Check GitHub Actions logs — is keep-warm cron running?
6. If memory pressure: disable reranker (`RERANKER_ENABLED=false`)
7. If disk full: clear old uploads from `/data/uploads`
8. If OpenRouter down: circuit breaker returns safe fallback message
