# RAG Document Search Engine

Semantic document search using FastAPI, LangChain, HuggingFace embeddings, and ChromaDB.

## Features

- Single-port browser app at `http://127.0.0.1:8000`.
- Upload PDF, TXT, or Markdown documents.
- Process uploads in a background ingestion task.
- Extract text with LangChain loaders.
- Split documents into overlapping chunks.
- Embed chunks with HuggingFace embeddings when available.
- Fall back to local hashing embeddings when offline.
- Store vectors and metadata in a persistent ChromaDB collection.
- Search by meaning instead of exact keywords.
- View retrieved source chunks in the browser UI.

## Project Layout

```text
.
|-- backend/
|   |-- main.py          # FastAPI app and API routes
|   |-- ingestion.py     # document loading, chunking, background indexing
|   |-- retrieval.py     # search response assembly
|   |-- schemas.py       # Pydantic request/response models
|   |-- settings.py      # paths and runtime settings
|   |-- vector_store.py  # embeddings + ChromaDB
|   `-- webapp.py        # single-port HTML UI
|-- data/
|   |-- uploads/         # runtime uploads
|   `-- chroma_db/       # runtime Chroma database
|-- main.py              # convenience API runner
`-- requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Open the app:

- http://127.0.0.1:8000

API docs:

- http://127.0.0.1:8000/docs

## API

Upload:

```http
POST /upload
```

Document status:

```http
GET /documents
GET /documents/{document_id}
```

Search:

```http
POST /search
Content-Type: application/json

{
  "query": "How does AI learn from examples?",
  "top_k": 5
}
```

## Notes

The first upload can take longer because the embedding model is loaded. If the
HuggingFace model is not available locally, the app automatically falls back to
a fully local hashing-based embedding backend so search still works offline.
Generated uploads and Chroma files are runtime data and are ignored by Git.
