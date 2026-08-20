"""Canonical domain models for HybridRAG.

Provides Document, Chunk, and IngestionJob entities with stable identities.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class IngestionJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class Document(BaseModel):
    """Canonical document entity."""
    id: str = Field(default_factory=lambda: uuid4().hex)
    filename: str
    content_hash: str
    version: int = 1
    status: DocumentStatus = DocumentStatus.QUEUED
    source_type: str = "upload"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1


class Chunk(BaseModel):
    """Canonical chunk entity with stable identity."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    section: str | None = None
    page: int | None = None
    chunk_strategy: str = "recursive"
    character_count: int
    token_count: int | None = None
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        document_id: str,
        chunk_index: int,
        content: str,
        content_hash: str,
        section: str | None = None,
        page: int | None = None,
        chunk_strategy: str = "recursive",
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Chunk":
        return cls(
            id=canonical_chunk_id(document_id, chunk_index),
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            section=section,
            page=page,
            chunk_strategy=chunk_strategy,
            character_count=len(content),
            token_count=token_count,
            content_hash=content_hash,
            metadata=metadata or {},
        )

    def to_langchain_document(self):
        """Convert to LangChain Document for vector store indexing."""
        from langchain_core.documents import Document
        meta = dict(self.metadata)
        meta.update(
            document_id=self.document_id,
            chunk=self.chunk_index,
            chunk_strategy=self.chunk_strategy,
            content_hash=self.content_hash,
            character_count=self.character_count,
        )
        if self.token_count is not None:
            meta["token_count"] = self.token_count
        if self.section is not None:
            meta["section"] = self.section
        if self.page is not None:
            meta["page"] = self.page
        return Document(page_content=self.content, metadata=meta)


class IngestionJob(BaseModel):
    """Canonical ingestion job tracking."""
    id: str = Field(default_factory=lambda: uuid4().hex)
    document_id: str
    status: IngestionJobStatus = IngestionJobStatus.QUEUED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    configuration_snapshot: dict[str, Any] = Field(default_factory=dict)

    def mark_running(self) -> None:
        self.status = IngestionJobStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_complete(self) -> None:
        self.status = IngestionJobStatus.COMPLETE
        self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, error_code: str, error_message: str) -> None:
        self.status = IngestionJobStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_code = error_code
        self.error_message = error_message


# ---------------------------------------------------------------------------
# Canonical ID generation
# ---------------------------------------------------------------------------

def canonical_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate stable canonical chunk ID.

    Format: {document_id}:{chunk_index}

    This ID is used across dense retrieval, future sparse retrieval,
    citations, and evaluation.
    """
    return f"{document_id}:{chunk_index}"


def parse_chunk_id(chunk_id: str) -> tuple[str, int] | None:
    """Parse canonical chunk ID into (document_id, chunk_index).

    Returns None if the ID doesn't match the canonical format.
    """
    parts = chunk_id.rsplit(":", 1)
    if len(parts) != 2:
        return None
    doc_id, idx_str = parts
    try:
        return doc_id, int(idx_str)
    except ValueError:
        return None


def canonical_job_id(document_id: str) -> str:
    """Generate ingestion job ID (currently 1:1 with document)."""
    return f"job:{document_id}"