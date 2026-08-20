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

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive").strip().lower()
NEAR_DUPLICATE_THRESHOLD = float(os.getenv("NEAR_DUPLICATE_THRESHOLD", "0.95"))
MAX_UPLOAD_MB  = int(os.getenv("MAX_UPLOAD_MB", "50"))

# Dense retrieval configuration
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "10"))
DENSE_FETCH_K = int(os.getenv("DENSE_FETCH_K", "20"))
DENSE_MMR_LAMBDA = float(os.getenv("DENSE_MMR_LAMBDA", "0.5"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))

# BM25 sparse retrieval configuration
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))
BM25_INDEX_VERSION = int(os.getenv("BM25_INDEX_VERSION", "1"))
BM25_INDEX_DIR = DATA_DIR / "bm25_index"

# RRF hybrid retrieval configuration
RRF_K = int(os.getenv("RRF_K", "60"))
RRF_DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT", "0.7"))
RRF_SPARSE_WEIGHT = float(os.getenv("RRF_SPARSE_WEIGHT", "0.3"))
RRF_TOP_K = int(os.getenv("RRF_TOP_K", "10"))

# Cross-encoder reranking configuration (Phase 6, opt-in by default)
# Reranker is DISABLED by default; enable explicitly via env / API mode "hybrid_rerank".
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
# Lightweight CPU cross-encoder (FlashRank / ONNX). Default is TinyBERT-L-2-v2:
# ~3 MB model, ~23 ms rerank, ~785 MB peak on this corpus, R@1=0.90 / MRR=0.95.
# ms-marco-MiniLM-L-12-v2 is stronger (R@1=0.94 / MRR=0.97) but ~28x slower
# (~648 ms rerank) and ~1141 MB peak - only viable on a larger instance, not the
# 512 MB free tier. Override via RERANKER_MODEL when latency/RAM budget allows.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2")
RERANKER_CANDIDATE_K = int(os.getenv("RERANKER_CANDIDATE_K", "20"))
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))

# Validate reranker config: candidate pool must be positive and >= final top-k.
if RERANKER_CANDIDATE_K <= 0:
    raise ValueError("RERANKER_CANDIDATE_K must be > 0")
if RERANKER_TOP_K <= 0:
    raise ValueError("RERANKER_TOP_K must be > 0")
if RERANKER_TOP_K > RERANKER_CANDIDATE_K:
    raise ValueError(
        f"RERANKER_TOP_K ({RERANKER_TOP_K}) must be <= RERANKER_CANDIDATE_K ({RERANKER_CANDIDATE_K})"
    )

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}

# OpenRouter answer-generation LLM (optional). Key should live in .env, which
# is git-ignored, or in the environment. Leave OPENROUTER_API_KEY empty to
# keep the plain context-based answer behaviour.
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
