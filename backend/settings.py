import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR  = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"

COLLECTION_NAME           = os.getenv("CHROMA_COLLECTION", "rag_documents")
EMBEDDING_MODEL           = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BACKEND         = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
HASHING_EMBEDDING_DIMS    = int(os.getenv("HASHING_EMBEDDING_DIMENSIONS", "1024"))

CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_UPLOAD_MB  = int(os.getenv("MAX_UPLOAD_MB", "50"))

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# OpenRouter answer-generation LLM (optional). Key should live in .env, which
# is git-ignored, or in the environment. Leave OPENROUTER_API_KEY empty to
# keep the plain context-based answer behaviour.
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Cross-Encoder reranker. Bi-encoder retrieval first pulls RERANK_CANDIDATES
# candidates, which are then rescored pairwise (query, chunk) and the top
# RERANK_CANDIDATES-capped slice is kept. Set RERANK_ENABLED=false to disable.
RERANK_ENABLED    = os.getenv("RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
RERANK_MODEL      = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
