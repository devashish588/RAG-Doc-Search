from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


class DocumentStatus(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_indexed: int = 0
    total_chunks: int | None = None
    progress_pct: float | None = None
    message: str | None = None
    error: str | None = None
    uploaded_at: datetime


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=20)
    source: str | None = Field(default=None, description="Optional filename filter")


class SearchResult(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    answer: str
    results: list[SearchResult]
    latency_ms: float


class DeleteResponse(BaseModel):
    document_id: str
    filename: str
    deleted_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    embedding_backend: str
    answer_model: str | None = None


# Phase 7: Confidence & Grounding models

class ConfidenceSignals(BaseModel):
    dense: float | None = None
    bm25: float | None = None
    rrf: float | None = None
    reranker: float | None = None
    grounding: float | None = None


class Confidence(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=1.0)
    level: str  # "high" | "medium" | "low"
    retrieval_confidence: float = Field(..., ge=0.0, le=1.0)
    grounding_confidence: float = Field(..., ge=0.0, le=1.0)
    abstention_flag: bool
    signals: ConfidenceSignals


class Citation(BaseModel):
    claim: str = ""
    source: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    verdict: str = "supported"  # "supported" | "unsupported"
    text_snippet: str | None = None
    text_snippet: str = ""  # backward compat (Phase 3 tests)


class GroundingMetrics(BaseModel):
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    grounding_ratio: float = 0.0
    citation_coverage: float = 0.0
    citation_accuracy: float = 0.0
