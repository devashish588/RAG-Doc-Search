# Runtime Dependency Guide

## Prerequisites

- Python 3.12+
- 1 GB RAM minimum (2 GB recommended for hybrid_rerank)
- 100 MB disk for dependencies + embeddings model cache

## Installation

```bash
git clone https://github.com/devashish588/RAG-Doc-Search.git
cd RAG-Doc-Search

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Validation

Verify all critical dependencies are installed:

```bash
python -c "import chromadb, fastembed, numpy, rank_bm25; print('Runtime dependencies OK')"
```

Expected output:
```
Runtime dependencies OK
```

If you see `ModuleNotFoundError`, reinstall:
```bash
pip install -r requirements.txt
```

## Startup

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 9826
```

Or using the convenience runner:
```bash
python main.py
```

On startup, the application validates:
1. All critical runtime dependencies are importable
2. The configured embedding backend (FastEmbed) initializes successfully
3. ChromaDB vector store is accessible

If any validation fails, the startup logs will contain a clear error message identifying the missing package and how to install it.

## Health Endpoints

| Endpoint | Purpose | Expected |
|----------|---------|----------|
| `GET /healthz` | Liveness probe | `200 {"status": "ok"}` |
| `GET /readyz` | Readiness with dependency check | `200` when ready, `503` when not |
| `GET /metrics` | Prometheus metrics | `200` |

The `/readyz` endpoint includes a dependency health check:
```json
{
  "status": "ready",
  "checks": {
    "dependencies": {"status": "ok"},
    "chroma": {"status": "ok"},
    "bm25": {"status": "ok"},
    "disk": {"status": "ok"}
  }
}
```

If a critical dependency is missing, `/readyz` returns `503`:
```json
{
  "status": "not_ready",
  "checks": {
    "dependencies": {"status": "error", "missing": ["chromadb"]}
  }
}
```

## Critical Dependencies

| Package | PyPI | Purpose | Required |
|---------|------|---------|----------|
| chromadb | `chromadb` | Vector database | Yes |
| fastembed | `fastembed` | ONNX embeddings | Yes (production) |
| numpy | `numpy` | Array operations (BM25, dedup) | Yes |
| rank-bm25 | `rank-bm25` | BM25 sparse retrieval | Yes |
| langchain-chroma | `langchain-chroma` | ChromaDB LangChain integration | Yes |
| scikit-learn | `scikit-learn` | Hashing fallback embeddings | Yes |
| fastapi | `fastapi` | Web framework | Yes |
| uvicorn | `uvicorn` | ASGI server | Yes |
| flashrank | `flashrank` | Cross-encoder reranking | Optional |
| prometheus-client | `prometheus-client` | Metrics | Yes |

## Troubleshooting

### `ModuleNotFoundError: No module named 'chromadb'`

```bash
pip install -r requirements.txt
```

### `Could not import 'fastembed' Python package`

FastEmbed provides the ONNX-based embedding model. Install:
```bash
pip install fastembed
```

The first use downloads the model (~130 MB) to `data/fastembed_cache/`.

To use the development-only hashing fallback instead:
```bash
set EMBEDDING_BACKEND=hashing
```

> **Warning:** Hashing embeddings are for development/testing only. They produce lower retrieval quality than FastEmbed. Do not use in production.

### `ModuleNotFoundError: No module named 'rank_bm25'`

```bash
pip install rank-bm25
```

### `ModuleNotFoundError: No module named 'numpy'`

```bash
pip install numpy
```

### Ingestion fails silently

Check `/readyz` — if dependencies are not ready, ingestion will fail. Verify:
```bash
python -c "from backend.dependency_check import check_runtime_dependencies; r = check_runtime_dependencies(); print(r.status, r.missing_critical)"
```

## Docker

```bash
docker build -t hybridrag .
docker run -p 9826:9826 -e OPENROUTER_API_KEY=sk-or-v1-... hybridrag
```

Docker builds install all dependencies from `requirements.txt`. The container runs as non-root with a HEALTHCHECK.

## Memory Requirements

| Tier | Status |
|------|--------|
| 512 MB | **UNSUPPORTED** — will OOM |
| 1 GB | Minimum practical |
| 2 GB | Recommended (for hybrid_rerank) |
