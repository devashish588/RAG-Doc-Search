"""API v1 router with structured contracts."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile, status
from pydantic import BaseModel, Field

from backend.ingestion import (
    delete_document,
    find_active_duplicate,
    get_document_status,
    list_document_statuses,
    register_document,
)
from backend.ingestion_queue import start as start_ingestion_worker, stop as stop_ingestion_worker, submit as submit_job
from backend.llm import llm_available
from backend.models import Chunk, Document, IngestionJob, canonical_chunk_id, canonical_job_id
from backend.retrieval import run_search
from backend.schemas import SearchRequest as LegacySearchRequest
from backend.vector_store import get_embeddings
from backend.settings import MAX_UPLOAD_MB, SUPPORTED_EXTENSIONS, UPLOAD_DIR, ensure_runtime_dirs
from backend.api_errors import (
    APIError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidFileTypeError,
    QueueFullError,
)
from backend.interfaces import RetrievalResult, RetrievalTrace


router = APIRouter(prefix="/v1")


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    job_id: str
    document_id: str
    status: str = "queued"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k_dense: int = Field(default=10, ge=1, le=50)
    top_k_sparse: int = Field(default=10, ge=1, le=50)
    top_k_fused: int = Field(default=20, ge=1, le=50)
    top_k_final: int = Field(default=5, ge=1, le=20)
    retrieval_mode: str = Field(default="hybrid", pattern="^(dense|sparse|hybrid)$")


class Citation(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    text_snippet: str


class AskResponse(BaseModel):
    answer: str
    status: str  # "answered" | "insufficient_context" | "failed"
    citations: list[Citation] = []
    confidence: float | None = None
    retrieval_trace: RetrievalTrace


class DocumentListResponse(BaseModel):
    documents: list[dict[str, Any]]


class DocumentDetailResponse(BaseModel):
    document: dict[str, Any]


class DeleteResponse(BaseModel):
    document_id: str
    filename: str
    deleted_chunks: int
    message: str


# ---------------------------------------------------------------------------
# Helper functions
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
                raise APIError(
                    code="FILE_TOO_LARGE",
                    message=f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                )
            hasher.update(chunk)
            fh.write(chunk)
    return hasher.hexdigest()


def _build_citations(results) -> list[Citation]:
    """Build citations from search results."""
    citations = []
    for i, r in enumerate(results):
        citations.append(Citation(
            chunk_id=r.metadata.get("document_id", "") + ":" + str(r.metadata.get("chunk", i)),
            source=r.source,
            page=r.page,
            text_snippet=r.text[:200],
        ))
    return citations


def _build_trace(results) -> RetrievalTrace:
    """Build retrieval trace from search results."""
    dense_results = []
    for i, r in enumerate(results):
        dense_results.append(RetrievalResult(
            chunk_id=r.metadata.get("document_id", "") + ":" + str(r.metadata.get("chunk", i)),
            score=r.score,
            rank=i + 1,
            source=r.source,
            content=r.text,
            metadata=r.metadata,
        ))
    return RetrievalTrace(dense=dense_results)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document_v1(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename:
        raise APIError(
            code="MISSING_FILENAME",
            message="A filename is required.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    name = Path(file.filename).name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise InvalidFileTypeError(name, sorted(SUPPORTED_EXTENSIONS))

    ensure_runtime_dirs()
    doc_id = canonical_job_id("").replace("job:", "")  # will be replaced by uuid
    from uuid import uuid4
    doc_id = uuid4().hex
    dest = UPLOAD_DIR / f"{doc_id}_{name}"
    content_hash = await _save_upload(file, dest)

    duplicate = find_active_duplicate(content_hash)
    if duplicate:
        dest.unlink(missing_ok=True)
        raise DuplicateDocumentError(duplicate["filename"])

    register_document(document_id=doc_id, filename=name, stored_path=dest, content_hash=content_hash)
    if not submit_job(doc_id, dest, name):
        delete_document(doc_id)
        dest.unlink(missing_ok=True)
        raise QueueFullError()

    job_id = canonical_job_id(doc_id)
    return IngestResponse(job_id=job_id, document_id=doc_id, status="queued")


@router.post("/ask", response_model=AskResponse)
async def ask_v1(request: AskRequest) -> AskResponse:
    # For Phase 1, only dense retrieval is active
    # Other parameters accepted for forward compatibility but not used
    legacy_req = LegacySearchRequest(
        query=request.question,
        top_k=request.top_k_dense,
        source=None,
    )
    try:
        search_res = run_search(legacy_req)
    except ValueError as exc:
        raise APIError(
            code="INVALID_QUERY",
            message=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc
    except Exception as exc:
        raise APIError(
            code="SEARCH_FAILED",
            message=f"Search failed: {exc}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    # Determine response status
    if not search_res.results:
        response_status = "insufficient_context"
    else:
        response_status = "answered"

    return AskResponse(
        answer=search_res.answer,
        status=response_status,
        citations=_build_citations(search_res.results),
        confidence=None,  # Phase 6 will implement
        retrieval_trace=_build_trace(search_res.results),
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents_v1() -> DocumentListResponse:
    return DocumentListResponse(documents=list_document_statuses())


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_v1(document_id: str) -> DocumentDetailResponse:
    record = get_document_status(document_id)
    if not record:
        raise DocumentNotFoundError(document_id)
    return DocumentDetailResponse(document=record)


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document_v1(document_id: str) -> DeleteResponse:
    result = delete_document(document_id)
    if not result:
        raise DocumentNotFoundError(document_id)
    return DeleteResponse(**result)