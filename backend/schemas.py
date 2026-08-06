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
    message: str | None = None
    error: str | None = None
    uploaded_at: datetime


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
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
    reranker_model: str | None = None
