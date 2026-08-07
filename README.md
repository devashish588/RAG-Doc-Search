# RAG Document Search Engine

Semantic document search using FastAPI, LangChain, fastembed (ONNX), and ChromaDB.

## Features

- Upload PDF, TXT, or Markdown documents.
- Process uploads in a background ingestion task.
- Extract text with LangChain loaders.
- Split documents into overlapping chunks.
- Embed chunks with `fastembed` (ONNX, `BAAI/bge-small-en-v1.5`) by default.
- Fall back to local hashing embeddings when offline.
- Store vectors and metadata in a persistent ChromaDB collection.
- Search by meaning instead of exact keywords.
- Retrieve diverse chunks with MMR, deduplicate, and add adjacent chunks so
  multi-page lists aren't cut off.
- Generate grounded answers with an LLM via OpenRouter (optional).
- View retrieved source chunks in the browser UI.

## Project Layout

```text
.
|-- frontend/              # standalone static UI (host separately)
|   |-- index.html         # page markup only
|   |-- styles.css         # all styling
|   |-- app.js             # all UI logic + API calls
|   `-- config.js          # API base URL (set for separate hosting)
|-- backend/               # FastAPI API only (host separately)
|   |-- main.py            # FastAPI app and API routes
|   |-- ingestion.py       # document loading, chunking, background indexing
|   |-- llm.py             # OpenRouter answer generation
|   |-- retrieval.py       # search response assembly
|   |-- schemas.py         # Pydantic request/response models
|   |-- settings.py        # paths and runtime settings
|   |-- vector_store.py    # embeddings + ChromaDB
|   `-- webapp.py          # serves frontend/ for local single-origin dev
|-- data/
|   |-- uploads/           # runtime uploads
|   `-- chroma_db/         # runtime Chroma database
|-- main.py                # convenience API runner
|-- requirements.txt
|-- .env                   # local secrets (git-ignored) — add OPENROUTER_API_KEY
```

The frontend and backend are fully decoupled:

- The frontend is a static site (`frontend/`) that talks to the API over HTTP.
- The backend is a FastAPI app exposing only the REST API (`/health`, `/upload`,
  `/documents`, `/search`).
- CORS is enabled, so the frontend can be hosted on a different origin than the
  backend.
- For local development the backend also serves the `frontend/` directory at the
  root, so `python main.py` still gives you a single-port app.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Optional: OpenRouter LLM answers

By default the app shows retrieved context only. To have an LLM write the
answer, add your OpenRouter key and (optionally) model to `.env`:

```text
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

`.env` is git-ignored, so the key never enters version control. If the key is
missing or the API call fails, the app silently falls back to the plain
context-based answer.

## Search pipeline

MMR retrieval pulls diverse chunks, which are then deduplicated by text
content and pruned by a `MIN_RELEVANCE_SCORE` cutoff. Adjacent chunks are pulled
in so multi-page lists aren't cut off at a chunk boundary.

## Run locally (single origin)

```bash
python main.py
```

Open the app:

- http://127.0.0.1:9826

API docs:

- http://127.0.0.1:9826/docs

## Run locally (frontend and backend separately)

Terminal 1 — backend:

```bash
python main.py
```

Terminal 2 — frontend with any static file server:

```bash
# from the frontend/ directory
python -m http.server 5500
```

Open http://127.0.0.1:5500. `frontend/config.js` auto-detects the environment:
on localhost it targets `http://localhost:9826`, on Render it uses same-origin,
and anywhere else it points at the canonical Render backend.

## API

Upload:

```http
POST /upload
```

Document status:

```http
GET /documents
GET /documents/{document_id}
DELETE /documents/{document_id}
```

Deletes a document's indexed chunks from ChromaDB, removes its uploaded file,
and clears its record.

Search:

```http
POST /search
Content-Type: application/json

{
  "query": "How does AI learn from examples?",
  "top_k": 5
}
```

## Deploying (frontend and backend separately)

### 1. Backend — Render (needs persistent disk for ChromaDB and uploads)

The backend keeps the Chroma vector database and uploaded files on disk
(`data/`), so it needs a long-running host with a persistent filesystem.
Serverless hosts (Vercel, Netlify Functions) are NOT suitable.

Deploy with Render:

1. Push this repo to GitHub (done).
2. In Render: **New → Blueprint** and pick the repo. `render.yaml` provisions:
   - the Web Service with `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - a **1 GB persistent disk** mounted at `data/` (survives redeploys)
3. In the service's **Environment** tab, set `OPENROUTER_API_KEY` to your key
   (it's marked as a secret in `render.yaml`).
4. Deploy. First request may be slow while fastembed downloads its model into
   the persistent disk (`data/fastembed_cache/`). fastembed's ONNX runtime is
   lightweight (~120 MB RAM), so the free 512 MB plan suffices. Set
   `EMBEDDING_BACKEND=hashing` to skip the model download entirely.

Your backend URL will look like `https://rag-doc-search-1.onrender.com`.

### 2. Frontend — GitHub Pages (static, no build)

Project Pages serve at `https://<user>.github.io/RAG-Doc-Search/`. The
`config.js` file auto-detects its host and points GitHub Pages at the canonical
Render backend, so no manual URL edit is needed.

1. Run the deploy script (PowerShell, from the repo root):
   ```powershell
   .\deploy-gh-pages.ps1
   ```
   This pushes the `frontend/` folder to a `gh-pages` branch.
2. In GitHub: **Settings → Pages → Build and deployment → Branch → `gh-pages` / `/root`** → Save.
3. Open `https://<user>.github.io/RAG-Doc-Search/`.

The backend already allows cross-origin requests (`allow_origins=["*"]`), so no
extra CORS setup is needed.

Notes:

- The first upload can take longer because the embedding model is loaded. If the
  `fastembed` model is not available locally, the app automatically
  falls back to a fully local hashing-based embedding backend so search still
  works offline.
- The embedding model downloads on first use (~130 MB ONNX, cached at
  `data/fastembed_cache/`). Set
  `EMBEDDING_BACKEND=hashing` to force the local no-download fallback.
- `CHROMA_COLLECTION`, `CHUNK_SIZE`, `MAX_UPLOAD_MB`, `OPENROUTER_MODEL`, and
  friends are configurable via environment variables (see `backend/settings.py`).
- The `/health` endpoint reports `answer_model` (null when OpenRouter is not
  configured).
- Generated uploads and Chroma files are runtime data and are ignored by Git.
