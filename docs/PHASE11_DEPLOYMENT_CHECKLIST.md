# Phase 11 — Deployment Checklist

## Render Service Configuration

| Setting | Value |
|---------|-------|
| Service type | Web |
| Runtime | Python |
| Plan | `standard` (1 GB min, 2 GB recommended) |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health check | `/healthz` |
| Auto-deploy | `true` |
| Disk | 1 GB, mounted at `/opt/render/project/src/data` |

## Environment Variables

| Variable | Source | Value |
|----------|--------|-------|
| `ENVIRONMENT` | Config | `production` |
| `LOG_LEVEL` | Config | `INFO` |
| `OPENROUTER_API_KEY` | Secret (manual) | **DO NOT COMMIT** |
| `OPENROUTER_MODEL` | Config | `openai/gpt-4o-mini` |
| `EMBEDDING_BACKEND` | Config | `auto` |
| `EMBEDDING_WARMUP` | Config | `true` |
| `PYTHONUNBUFFERED` | Config | `1` |
| `CORS_ORIGINS` | Config | `https://<your-github-pages>.github.io,https://rag-doc-search-1.onrender.com` |
| `RATE_LIMIT_REQUESTS` | Config | `60` |
| `RATE_LIMIT_WINDOW_SECONDS` | Config | `60` |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | Config | `3` |
| `CIRCUIT_BREAKER_RECOVERY_SECONDS` | Config | `30` |
| `RERANKER_ENABLED` | Config | `false` (set `true` for 2 GB) |
| `RERANKER_MODEL` | Config | `ms-marco-TinyBERT-L-2-v2` |
| `CONFIDENCE_THRESHOLD` | Config | `0.50` |
| `MAX_UPLOAD_MB` | Config | `25` |
| `MAX_TOP_K` | Config | `50` |
| `MAX_TOP_K_FINAL` | Config | `20` |

> **Important**: In production, set `CORS_ORIGINS` to your actual GitHub Pages origin, not `*`.

## Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Full health (Chroma + BM25 + disk) |
| `/healthz` | Lightweight liveness probe (no DB checks) |
| `/readyz` | Readiness with storage details |
| `/metrics` | Prometheus metrics |

## Frontend API Endpoint

| Environment | URL |
|-------------|-----|
| GitHub Pages | `https://rag-doc-search-1.onrender.com` (auto-detected via `config.js`) |
| Render (same-origin) | Empty string (auto-detected via `config.js`) |
| Local | `http://localhost:9826` (auto-detected via `config.js`) |

## Memory Requirement

| Tier | Status |
|------|--------|
| 512 MB | **UNSUPPORTED** — will OOM |
| 1 GB | Minimum practical — basic modes only |
| 2 GB | Recommended — hybrid_rerank enabled |

## Docker Configuration

```bash
docker build -t rag-doc-search .
docker run -p 9826:9826 -e OPENROUTER_API_KEY=... rag-doc-search
```

Or via docker-compose:
```bash
docker compose up
```

Docker memory limit: 2 GB.

## Persistence Requirements

| Data | Location | Survives Redeploy |
|------|----------|-------------------|
| ChromaDB index | `/data/chroma_db` | Yes (disk mount) |
| BM25 index | `/data/bm25_index` | Yes (disk mount) |
| Uploads | `/data/uploads` | Yes (disk mount) |

## Rollback Procedure

1. In Render dashboard → Deploys → select previous successful deploy → "Rollback to this deploy"
2. Or: `git revert <commit>` and push — auto-deploy triggers
3. Or: tag the rollback commit and redeploy

## Smoke Test Procedure

```bash
# Local
curl http://localhost:9826/healthz
curl http://localhost:9826/readyz
curl -X POST http://localhost:9826/v1/ask -H "Content-Type: application/json" -d '{"question":"test","retrieval_mode":"dense"}'

# Production
curl https://rag-doc-search-1.onrender.com/healthz
curl -X POST https://rag-doc-search-1.onrender.com/v1/ask -H "Content-Type: application/json" -d '{"question":"test","retrieval_mode":"dense"}'
```

## Required Repository Variables (GitHub Actions)

| Variable | Purpose |
|----------|---------|
| `RENDER_HEALTH_URL` | Full URL to `/healthz` endpoint for keep-warm cron |
