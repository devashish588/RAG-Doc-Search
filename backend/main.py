import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib import error as _urlerror
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.api_errors import HTTPException as _HTTPException, register_error_handlers
from backend.api_v1 import router as v1_router
from backend.dependency_check import startup_validation
from backend.circuit_breaker import get_circuit_breaker
from backend.ingestion import (
    delete_document,
    find_active_duplicate,
    get_document_status,
    list_document_statuses,
    register_document,
)
from backend.ingestion_queue import start as start_ingestion_worker, stop as stop_ingestion_worker, submit as submit_job
from backend.llm import llm_available
from backend.middleware import RateLimitMiddleware
from backend.monitoring import (
    RequestMonitoringMiddleware,
    metrics_snapshot,
)
from backend.retrieval import run_search
from backend.reconciliation import health_check as reconciliation_health_check
from backend.schemas import DeleteResponse, DocumentStatus, HealthResponse, SearchRequest, SearchResponse, UploadResponse
from backend.storage_health import readiness_report
from backend.vector_store import get_embeddings
from backend.settings import (
    CHROMA_DIR,
    CORS_ORIGINS,
    EMBEDDING_BACKEND,
    EMBEDDING_WARMUP,
    MAX_UPLOAD_MB,
    OPENROUTER_MODEL,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
    ensure_runtime_dirs,
)
from backend.webapp import FRONTEND_DIR, build_index_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _warmup():
    try:
        get_vector_store()
        get_embeddings().embed_query("warmup query")
        log.info("Warmup complete: embeddings + vector store resident")
    except Exception as exc:
        log.error("Warmup failed: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_runtime_dirs()
    startup_validation()
    if EMBEDDING_WARMUP:
        _warmup()
    start_ingestion_worker()
    yield
    stop_ingestion_worker()


app = FastAPI(
    title="RAG Document Search",
    description="Semantic document search with FastAPI, LangChain, fastembed (ONNX), and ChromaDB.",
    version="1.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — configurable, production-safe
# ---------------------------------------------------------------------------
_cors_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
_allow_all = "*" in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _cors_origins,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Production middleware (order matters: outermost runs first) ---
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestMonitoringMiddleware)

# Register structured error handlers
register_error_handlers(app)

# Include v1 API router
app.include_router(v1_router)


# ---------------------------------------------------------------------------
# Health probes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_store=str(CHROMA_DIR),
        embedding_backend=EMBEDDING_BACKEND,
        answer_model=OPENROUTER_MODEL if llm_available() else None,
    )


@app.get("/healthz")
def healthz():
    """Liveness probe — extremely lightweight."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness probe — checks Chroma, BM25, disk."""
    report = readiness_report()
    code = 200 if report["status"] == "ready" else 503
    return Response(
        content=json.dumps(report),
        status_code=code,
        media_type="application/json",
    )


@app.get("/health/index", response_model=dict)
def index_health() -> dict:
    """Index reconciliation health check."""
    return reconciliation_health_check()


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

@app.get("/metrics")
def metrics():
    return PlainTextResponse(content=metrics_snapshot(), media_type="text/plain; version=0.0.4; charset=utf-8")


# ---------------------------------------------------------------------------
# Upload helper
# ---------------------------------------------------------------------------

async def _save_upload(upload: UploadFile, dest: Path) -> str:
    import hashlib

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    hasher = hashlib.sha256()
    with dest.open("wb") as fh:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
                )
            hasher.update(chunk)
            fh.write(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Legacy routes (backward compatible, delegate to same logic)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(build_index_html())


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A filename is required.")

    name = Path(file.filename).name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    ensure_runtime_dirs()
    doc_id = uuid4().hex
    dest = UPLOAD_DIR / f"{doc_id}_{name}"
    content_hash = await _save_upload(file, dest)

    duplicate = find_active_duplicate(content_hash)
    if duplicate:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document already being indexed: '{duplicate['filename']}'.",
        )

    register_document(document_id=doc_id, filename=name, stored_path=dest, content_hash=content_hash)
    if not submit_job(doc_id, dest, name):
        delete_document(doc_id)
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion queue is full. Please try again shortly.",
        )

    return UploadResponse(
        document_id=doc_id,
        filename=name,
        status="queued",
        message="Upload accepted. Ingestion running in the background.",
    )


@app.get("/documents", response_model=list[DocumentStatus])
def documents() -> list[DocumentStatus]:
    return list_document_statuses()


@app.get("/documents/{document_id}", response_model=DocumentStatus)
def document_status(document_id: str) -> DocumentStatus:
    record = get_document_status(document_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentStatus(**record)


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document_endpoint(document_id: str) -> DeleteResponse:
    result = delete_document(document_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DeleteResponse(**result)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        return run_search(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed.") from exc


# ---------------------------------------------------------------------------
# Global exception handler — never expose tracebacks or secrets
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return Response(
        content=json.dumps({"error": {"code": "INTERNAL_ERROR", "message": "An internal error occurred."}}),
        status_code=500,
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Local single-origin UI: serve the frontend/ directory at root.
# Registered last so all API routes above take priority. Safe to remove when
# the frontend is hosted separately.
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
