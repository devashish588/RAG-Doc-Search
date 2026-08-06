import os
from pathlib import Path

# Pin native runtimes to a single thread: Render free has 1 vCPU, and ONNX
# thread arenas waste scarce memory on a 512 MB instance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
EMBEDDING_MODEL           = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_BACKEND         = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
EMBEDDING_WARMUP          = os.getenv("EMBEDDING_WARMUP", "false").strip().lower() in {"1", "true", "yes", "on"}
HASHING_EMBEDDING_DIMS    = int(os.getenv("HASHING_EMBEDDING_DIMENSIONS", "1024"))

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "2000"))  # Increased from 1000
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "400"))  # Increased from 200
MAX_UPLOAD_MB  = int(os.getenv("MAX_UPLOAD_MB", "50"))

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# OpenRouter answer-generation LLM (optional). Key should live in .env, which
# is git-ignored, or in the environment. Leave OPENROUTER_API_KEY empty to
# keep the plain context-based answer behaviour.
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
