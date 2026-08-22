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
from backend.retrieval import run_search, run_search_dense, run_search_bm25, run_search_hybrid
from backend.reranker import get_reranker
from backend.schemas import SearchRequest as LegacySearchRequest
from backend.schemas import (
    Confidence,
    ConfidenceSignals,
    Citation as SchemaCitation,
    GroundingMetrics,
)
from backend.verifier import get_verifier
from backend.confidence import get_confidence_estimator
from backend.vector_store import get_embeddings
from backend.settings import (
    MAX_UPLOAD_MB,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
    ensure_runtime_dirs,
)
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
    retrieval_mode: str = Field(default="dense", pattern="^(dense|sparse|hybrid|hybrid_rerank)$")


class AskResponse(BaseModel):
    answer: str
    status: str  # "answered" | "insufficient_context" | "failed"
    citations: list[SchemaCitation] = []
    confidence: Confidence | None = None
    grounding_metrics: GroundingMetrics | None = None
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


def _build_citations_from_verification(verification_result: dict[str, Any]) -> list[SchemaCitation]:
    """Build citations from verification output."""
    citations = []
    for cite in verification_result.get("citations", []):
        citations.append(SchemaCitation(
            claim=cite.get("claim", ""),
            source=cite.get("source"),
            page=cite.get("page"),
            chunk_id=cite.get("chunk_id"),
            verdict=cite.get("verdict", "unsupported"),
            text_snippet=cite.get("claim", "")[:200],
        ))
    return citations


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
    # Dense, BM25, and Hybrid retrieval are active in Phase 5
    try:
        source_filter = getattr(request, "source", None)

        if request.retrieval_mode == "dense":
            # Dense only mode
            dense_results = run_search_dense(
                query=request.question,
                k=request.top_k_dense,
                source=source_filter,
            )
            bm25_results = []
            rrf_results = []
            reranker_results = []
            legacy_req = LegacySearchRequest(
                query=request.question,
                top_k=request.top_k_dense,
                source=source_filter,
            )
            search_res = run_search(legacy_req)
        elif request.retrieval_mode == "sparse":
            # BM25 only mode
            dense_results = []
            bm25_results = run_search_bm25(
                query=request.question,
                k=request.top_k_sparse,
                source=source_filter,
            )
            rrf_results = []
            reranker_results = []
            legacy_req = LegacySearchRequest(
                query=request.question,
                top_k=request.top_k_sparse,
                source=source_filter,
            )
            search_res = run_search(legacy_req)
        elif request.retrieval_mode == "hybrid_rerank":
            # Hybrid RRF + Cross-Encoder reranking
            dense_results = run_search_dense(
                query=request.question,
                k=request.top_k_dense,
                source=source_filter,
            )
            bm25_results = run_search_bm25(
                query=request.question,
                k=request.top_k_sparse,
                source=source_filter,
            )
            rrf_results = run_search_hybrid(
                query=request.question,
                k=request.top_k_fused,
                top_k_dense=request.top_k_dense,
                top_k_sparse=request.top_k_sparse,
                top_k_fused=request.top_k_fused,
                source=source_filter,
            )
            # Rerank the RRF candidate pool (top_k_fused) down to top_k_final.
            reranker = get_reranker()
            reranker_results = reranker.rerank(
                query=request.question,
                candidates=rrf_results,
                top_k=request.top_k_final,
            )
            # Answer generation uses the reranked context (existing behavior)
            legacy_req = LegacySearchRequest(
                query=request.question,
                top_k=request.top_k_final,
                source=source_filter,
            )
            search_res = run_search(legacy_req)
        else:  # hybrid mode
            # Hybrid RRF mode
            dense_results = run_search_dense(
                query=request.question,
                k=request.top_k_dense,
                source=source_filter,
            )
            bm25_results = run_search_bm25(
                query=request.question,
                k=request.top_k_sparse,
                source=source_filter,
            )
            rrf_results = run_search_hybrid(
                query=request.question,
                k=request.top_k_fused,
                top_k_dense=request.top_k_dense,
                top_k_sparse=request.top_k_sparse,
                top_k_fused=request.top_k_fused,
                source=source_filter,
            )
            reranker_results = []
            # Use dense for answer generation (existing behavior)
            legacy_req = LegacySearchRequest(
                query=request.question,
                top_k=request.top_k_dense,
                source=source_filter,
            )
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

    # Phase 7: Grounding, Verification, and Confidence Integration
    verifier = get_verifier()
    confidence_estimator = get_confidence_estimator()

    # Build evaluation context from search results (the evidence the LLM saw)
    eval_context = []
    for i, r in enumerate(search_res.results):
        chunk_id_str = str(r.metadata.get("document_id", "")) + ":" + str(r.metadata.get("chunk", i))
        eval_context.append(RetrievalResult(
            chunk_id=chunk_id_str,
            score=r.score,
            rank=i + 1,
            source=r.source,
            content=r.text,
            metadata=r.metadata,
        ))

    # Verify claims against evidence
    verification_output = verifier.verify(search_res.answer, eval_context)
    raw_citations = verification_output.get("citations", [])
    g_metrics = verification_output.get("metrics", {})

    # Estimate confidence from retrieval signals + grounding
    trace_dict = {
        "dense": dense_results,
        "bm25": bm25_results,
        "rrf": rrf_results,
        "reranker": reranker_results,
    }

    conf_output = confidence_estimator.estimate(
        dense_results=dense_results,
        bm25_results=bm25_results,
        rrf_results=rrf_results,
        reranker_results=reranker_results,
        grounding_ratio=g_metrics.get("grounding_ratio", 0.0),
        retrieval_mode=request.retrieval_mode,
    )

    schema_citations = _build_citations_from_verification(verification_output)

    conf_obj = Confidence(
        overall_score=conf_output.overall_score,
        level=conf_output.level,
        retrieval_confidence=conf_output.retrieval_confidence,
        grounding_confidence=conf_output.grounding_confidence,
        abstention_flag=conf_output.abstention_flag,
        signals=ConfidenceSignals(
            dense=conf_output.signals.dense,
            bm25=conf_output.signals.bm25,
            rrf=conf_output.signals.rrf,
            reranker=conf_output.signals.reranker,
            grounding=conf_output.signals.grounding,
        ),
    )

    grounding_metrics_obj = GroundingMetrics(
        total_claims=g_metrics.get("total_claims", 0),
        supported_claims=g_metrics.get("supported_claims", 0),
        unsupported_claims=g_metrics.get("unsupported_claims", 0),
        grounding_ratio=g_metrics.get("grounding_ratio", 0.0),
        citation_coverage=g_metrics.get("citation_coverage", 0.0),
        citation_accuracy=g_metrics.get("citation_accuracy", 0.0),
    )

    # Determine response status & abstention guardrail
    if not search_res.results or conf_output.abstention_flag:
        response_status = "insufficient_context"
        final_answer = "I don't have enough evidence in the provided documents to answer this reliably."
        final_citations = []
    else:
        response_status = "answered"
        final_answer = search_res.answer
        final_citations = schema_citations

    return AskResponse(
        answer=final_answer,
        status=response_status,
        citations=final_citations,
        confidence=conf_obj,
        grounding_metrics=grounding_metrics_obj,
        retrieval_trace=RetrievalTrace(
            dense=dense_results,
            bm25=bm25_results,
            rrf=rrf_results,
            reranker=reranker_results,
        ),
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