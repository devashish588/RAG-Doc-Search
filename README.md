# RAG Document Search Engine

Semantic document search using FastAPI, LangChain, HuggingFace embeddings, and ChromaDB.

## Features

- Upload PDF, TXT, or Markdown documents.
- Process uploads in a background ingestion task.
- Extract text with LangChain loaders.
- Split documents into overlapping chunks.
- Embed chunks with `sentence-transformers` (all-MiniLM-L6-v2) when available.
- Fall back to local hashing embeddings when offline.
- Store vectors and metadata in a persistent ChromaDB collection.
- Search by meaning instead of exact keywords.
- Rerank results with a Cross-Encoder so substantive chunks beat summaries.
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

## Reranker

Bi-encoder retrieval pulls 20 candidate chunks (configurable via
`RERANK_CANDIDATES`), then a Cross-Encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescored each `(query, chunk)` pair so
substantive answers rank above summaries. The top `top_k` survive, then a
`MIN_RELEVANCE_SCORE` cutoff drops noise. Disable with `RERANK_ENABLED=false`;
the reranker auto-disables if the model cannot load.

## Run locally (single origin)

```bash
python main.py
```

Open the app:

- http://127.0.0.1:9752

API docs:

- http://127.0.0.1:9752/docs

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

Open http://127.0.0.1:5500. Because `frontend/config.js` sets `API_BASE = ""`
by default, the UI will call the backend on the same origin and fail. For
separate local ports set `API_BASE` to `http://127.0.0.1:9752`.

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

### 1. Backend — needs persistent disk for ChromaDB and uploads

The backend keeps the Chroma vector database and uploaded files on local disk
(`data/`), so it needs a long-running host with a persistent filesystem.
Serverless (Vercel, Netlify Functions) is NOT suitable.

Recommended platforms:

- Render (Web Service), Railway, or Fly.io — deploy from the repo root with the
  start command `uvicorn main:app --host 0.0.0.0 --port $PORT` (see `Procfile`).
- Or any VPS (DigitalOcean, AWS EC2, Hetzner) running the same command.

Notes:

- The embedding model downloads on first use (~90 MB). Add `EMBEDDING_MODEL`,
  `EMBEDDING_BACKEND=hashing` to force the local no-download fallback, or bump
  the instance's memory if cold starts are slow.
- `CHROMA_COLLECTION`, `CHUNK_SIZE`, and `MAX_UPLOAD_MB` are configurable via
  environment variables (see `backend/settings.py`).
- If you ever redeploy from scratch, the Chroma data is wiped — keep `data/`
  backed up or attach persistent storage.

### 2. Frontend — static hosting, no build step

The `frontend/` folder is a plain static site. Host it on:

- Netlify Drop (drag the `frontend/` folder in), Vercel, Cloudflare Pages, or
  GitHub Pages.

Before deploying:

1. Open `frontend/config.js`.
2. Set `window.API_BASE` to your deployed backend URL, e.g.
   `"https://my-backend.onrender.com"` (no trailing slash).
3. Deploy the `frontend/` folder.

The backend already allows cross-origin requests, so no extra CORS setup is
needed.

Notes:

- The first upload can take longer because the embedding model is loaded. If the
  `sentence-transformers` model is not available locally, the app automatically
  falls back to a fully local hashing-based embedding backend so search still
  works offline.
- The embedding model downloads on first use (~90 MB). Set
  `EMBEDDING_BACKEND=hashing` to force the local no-download fallback.
- `CHROMA_COLLECTION`, `CHUNK_SIZE`, `MAX_UPLOAD_MB`, `OPENROUTER_MODEL`, and
  friends are configurable via environment variables (see `backend/settings.py`).
- The `/health` endpoint reports `answer_model` (null when OpenRouter is not
  configured).
- Generated uploads and Chroma files are runtime data and are ignored by Git.
