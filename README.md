# RAG Document Search Engine

Semantic document search using FastAPI, LangChain, HuggingFace embeddings,
ChromaDB, and Streamlit.

## Features

- Upload PDF, TXT, or Markdown documents.
- Process uploads in a background ingestion task.
- Extract text with LangChain loaders.
- Split documents into overlapping chunks.
- Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`.
- Store vectors and metadata in a persistent ChromaDB collection.
- Search by meaning instead of exact keywords.
- View retrieved source chunks in Streamlit.

## Project Layout

```text
.
├── backend/
│   ├── main.py          # FastAPI app and API routes
│   ├── ingestion.py     # document loading, chunking, background indexing
│   ├── retrieval.py     # search response assembly
│   ├── schemas.py       # Pydantic request/response models
│   ├── settings.py      # paths and runtime settings
│   └── vector_store.py  # HuggingFace embeddings + ChromaDB
├── frontend/
│   └── app.py           # Streamlit UI
├── data/
│   ├── uploads/         # uploaded files
│   └── chroma_db/       # persistent Chroma database
├── main.py              # convenience API runner
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Start the FastAPI backend:

```bash
python main.py
```

Then open the full app at:

- http://127.0.0.1:8000

Optional API docs:

- http://127.0.0.1:8000/docs

The Streamlit frontend in `frontend/app.py` is still available for experiments,
but the main project now runs from one common port on FastAPI.

## API

### Upload a Document

```http
POST /upload
```

Returns immediately with a queued status while ingestion continues in the
background.

### Check Documents

```http
GET /documents
GET /documents/{document_id}
```

### Search

```http
POST /search
Content-Type: application/json

{
  "query": "How does AI learn from examples?",
  "top_k": 5
}
```

Example response:

```json
{
  "query": "How does AI learn from examples?",
  "answer": "Most relevant context:\n\n[1] notes.pdf: ...",
  "results": [
    {
      "text": "Machine learning models require labeled datasets...",
      "source": "notes.pdf",
      "page": 2,
      "score": 0.91,
      "metadata": {
        "source": "notes.pdf",
        "document_id": "..."
      }
    }
  ],
  "latency_ms": 42.7
}
```

## Notes

The first upload can take longer because the embedding model is loaded. If the
HuggingFace model is not available locally, the app automatically falls back to
a fully local hashing-based embedding backend so search still works offline.
Later searches use precomputed document embeddings stored in ChromaDB.
