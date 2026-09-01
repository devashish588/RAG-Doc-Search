import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

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

# ---------------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Comma-separated origins.  "*" means allow all (development default).
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").strip()

# ---------------------------------------------------------------------------
# Rate limiting (per-IP sliding window, in-memory)
# ---------------------------------------------------------------------------
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# ---------------------------------------------------------------------------
# Circuit breaker (OpenRouter)
# ---------------------------------------------------------------------------
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "3"))
CIRCUIT_BREAKER_RECOVERY_SECONDS = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_SECONDS", "30"))

# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "50"))
MAX_TOP_K_FINAL = int(os.getenv("MAX_TOP_K_FINAL", "20"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

COLLECTION_NAME           = os.getenv("CHROMA_COLLECTION", "rag_documents")
EMBEDDING_MODEL           = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_BACKEND         = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
EMBEDDING_WARMUP          = os.getenv("EMBEDDING_WARMUP", "false").strip().lower() in {"1", "true", "yes", "on"}
HASHING_EMBEDDING_DIMS    = int(os.getenv("HASHING_EMBEDDING_DIMENSIONS", "1024"))
FASTEMBED_BATCH_SIZE      = int(os.getenv("FASTEMBED_BATCH_SIZE", "64"))
EMBEDDING_BATCH_SIZE      = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive").strip().lower()
NEAR_DUPLICATE_THRESHOLD = float(os.getenv("NEAR_DUPLICATE_THRESHOLD", "0.95"))

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

# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------
# LLM_PROVIDER selects the backend: "openrouter" (default) or "groq".
# Unified LLM_* vars override provider-specific OPENROUTER_* / GROQ_* vars.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

# OpenRouter (legacy, kept for backward compatibility)
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Groq
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Unified LLM config (provider-agnostic). These are the values llm.py uses.
# If LLM_BASE_URL / LLM_MODEL are set, they win. Otherwise the provider
# default is used.
LLM_API_KEY = ""
LLM_BASE_URL = ""
LLM_MODEL = ""

if LLM_PROVIDER == "groq":
    LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip() or GROQ_API_KEY
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or GROQ_BASE_URL
    LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or GROQ_MODEL
else:  # openrouter (default)
    LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip() or OPENROUTER_API_KEY
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or OPENROUTER_BASE_URL
    LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or OPENROUTER_MODEL


# Citation verification & grounding (Phase 7)
VERIFIER_SUPPORT_THRESHOLD = float(os.getenv("VERIFIER_SUPPORT_THRESHOLD", "0.82"))

# Confidence estimation (Phase 7)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
CONFIDENCE_HIGH_THRESHOLD = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.75"))
CONFIDENCE_MEDIUM_THRESHOLD = float(os.getenv("CONFIDENCE_MEDIUM_THRESHOLD", "0.40"))

# Confidence component weights (sum to 1.0 for retrieval signals)
CONF_DENSE_WEIGHT = float(os.getenv("CONF_DENSE_WEIGHT", "0.4"))
CONF_BM25_WEIGHT = float(os.getenv("CONF_BM25_WEIGHT", "0.3"))
CONF_RRF_WEIGHT = float(os.getenv("CONF_RRF_WEIGHT", "0.2"))
CONF_RERANKER_WEIGHT = float(os.getenv("CONF_RERANKER_WEIGHT", "0.1"))
CONF_RETRIEVAL_WEIGHT = float(os.getenv("CONF_RETRIEVAL_WEIGHT", "0.5"))
CONF_GROUNDING_WEIGHT = float(os.getenv("CONF_GROUNDING_WEIGHT", "0.5"))


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
