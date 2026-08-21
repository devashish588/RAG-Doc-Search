"""Ingestion V2 pipeline with canonical architecture.

Implements the target ingestion architecture:
Raw File → Format Detection → Document Loader → Normalized Document
    → Metadata Preservation → Chunking Strategy → Exact Deduplication
    → Near-Duplicate Detection → Canonical Chunks → Embedding / Vector Index
"""
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.bm25_index import get_bm25_index
from backend.chunking import get_chunker
from backend.deduplication import Deduplicator, compute_document_hash, deduplicate_chunks
from backend.loaders import get_loader, supported_extensions as loader_extensions
from backend.models import Chunk, Document as ModelDocument, IngestionJob, DocumentStatus as ModelDocStatus
from langchain_core.documents import Document
from backend.normalizer import get_normalizer
from backend.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKING_STRATEGY,
    NEAR_DUPLICATE_THRESHOLD,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
)
from backend.vector_store import add_documents, delete_by_document, get_embeddings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ingestion configuration snapshot
# ---------------------------------------------------------------------------

def get_ingestion_config_snapshot() -> dict[str, Any]:
    """Capture current ingestion configuration for job snapshots."""
    return {
        "chunking_strategy": CHUNKING_STRATEGY,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "embedding_backend": os.getenv("EMBEDDING_BACKEND", "auto"),
    }


# ---------------------------------------------------------------------------
# Structured logging events
# ---------------------------------------------------------------------------

INGESTION_EVENTS = {
    "document_received": "document_received",
    "document_loaded": "document_loaded",
    "document_normalized": "document_normalized",
    "chunking_started": "chunking_started",
    "chunking_completed": "chunking_completed",
    "deduplication_started": "deduplication_started",
    "deduplication_completed": "deduplication_completed",
    "embedding_started": "embedding_started",
    "indexing_completed": "indexing_completed",
    "ingestion_completed": "ingestion_completed",
    "ingestion_failed": "ingestion_failed",
}


def log_event(event: str, document_id: str, **details: Any) -> None:
    """Log structured ingestion event."""
    log.info(
        "%s: document_id=%s %s",
        event,
        document_id,
        " ".join(f"{k}={v}" for k, v in details.items()),
    )


# ---------------------------------------------------------------------------
# Document loading and normalization
# ---------------------------------------------------------------------------

def load_document(path: Path) -> tuple[list, dict[str, Any]]:
    """Load document and return (langchain_documents, metadata)."""
    loader = get_loader(path)
    docs = loader.load(path)

    # Extract document-level metadata
    metadata = {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "source_type": "upload",
        "loader": loader.__class__.__name__,
    }

    # Aggregate metadata from loaded documents
    if docs:
        first_meta = docs[0].metadata
        if "page_count" in first_meta:
            metadata["page_count"] = first_meta["page_count"]
        if "sections" in first_meta:
            metadata["sections"] = first_meta["sections"]

    return docs, metadata


def normalize_documents(documents: list, normalizer_name: str = "standard") -> list:
    """Normalize loaded documents."""
    normalizer = get_normalizer(normalizer_name)
    normalized = []

    for doc in documents:
        normalized_text = normalizer.normalize(doc.page_content)
        normalized_doc = Document(
            page_content=normalized_text,
            metadata=dict(doc.metadata),
        )
        normalized.append(normalized_doc)

    return normalized


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_document_v2(
    document_id: str,
    stored_path: Path,
    filename: str,
    ingestion_job: IngestionJob | None = None,
) -> dict[str, Any]:
    """Execute the full ingestion pipeline for a document.

    Returns a dict with results: chunks_indexed, status, error, etc.
    """
    log_event(INGESTION_EVENTS["document_received"], document_id, filename=filename)

    try:
        # Mark job as running
        if ingestion_job:
            ingestion_job.mark_running()
            ingestion_job.configuration_snapshot = get_ingestion_config_snapshot()

        # Load document
        log_event(INGESTION_EVENTS["document_loaded"], document_id)
        langchain_docs, doc_metadata = load_document(stored_path)
        if not langchain_docs:
            raise ValueError("No content extracted from document")

        # Normalize
        log_event(INGESTION_EVENTS["document_normalized"], document_id)
        normalized_docs = normalize_documents(langchain_docs)

        # Chunking
        log_event(INGESTION_EVENTS["chunking_started"], document_id, strategy=CHUNKING_STRATEGY)
        chunker = get_chunker(CHUNKING_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP)
        chunks = chunker.chunk(normalized_docs, document_id, filename)
        log_event(INGESTION_EVENTS["chunking_completed"], document_id, chunk_count=len(chunks))

        from backend.ingestion import _update
        _update(
            document_id,
            status="indexing",
            chunks_indexed=len(chunks),
            message=f"Chunked into {len(chunks)} chunks. Indexing...",
            error=None,
        )

        if not chunks:
            raise ValueError("No chunks generated from document")

        # Deduplication
        log_event(INGESTION_EVENTS["deduplication_started"], document_id)
        deduplicator = Deduplicator(near_duplicate_threshold=NEAR_DUPLICATE_THRESHOLD)

        # Get embeddings function for near-duplicate detection
        embeddings = get_embeddings()

        def embed_fn(text: str) -> list[float]:
            emb = embeddings.embed_query(text)
            import numpy as np
            arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else arr

        kept_chunks, dup_results = deduplicate_chunks(
            chunks,
            near_duplicate_threshold=NEAR_DUPLICATE_THRESHOLD,
            embedder_fn=embed_fn,
        )

        dup_count = sum(1 for r in dup_results if r.is_duplicate)
        log_event(
            INGESTION_EVENTS["deduplication_completed"],
            document_id,
            kept=len(kept_chunks),
            duplicates=dup_count,
        )

        if not kept_chunks:
            raise ValueError("All chunks were duplicates")

        # Prepare for indexing
        langchain_chunks = []
        chunk_ids = []

        for chunk in kept_chunks:
            lc_doc = chunk.to_langchain_document()
            langchain_chunks.append(lc_doc)
            chunk_ids.append(chunk.id)

        # Embed and index (dense)
        log_event(INGESTION_EVENTS["embedding_started"], document_id, chunk_count=len(langchain_chunks))
        count = add_documents(langchain_chunks, chunk_ids)
        log_event(INGESTION_EVENTS["indexing_completed"], document_id, indexed=count)

        # Index to BM25 (sparse)
        log_event(INGESTION_EVENTS["indexing_completed"], document_id, indexed=count, index="bm25")
        try:
            bm25_index = get_bm25_index()
            bm25_index.build(kept_chunks)
            bm25_index.save()
            log.info("BM25 index built and saved for document %s", document_id)
        except Exception as exc:
            log.warning("Failed to build BM25 index for document %s: %s", document_id, exc)

        # Mark job complete
        if ingestion_job:
            ingestion_job.mark_complete()

        log_event(INGESTION_EVENTS["ingestion_completed"], document_id, chunks_indexed=count)

        return {
            "status": "complete",
            "chunks_indexed": count,
            "chunks_generated": len(chunks),
            "duplicates_removed": dup_count,
            "message": f"Indexed {count} chunks.",
            "error": None,
        }

    except Exception as exc:
        log.exception("Ingestion failed for %s", document_id)
        log_event(INGESTION_EVENTS["ingestion_failed"], document_id, error=str(exc))

        if ingestion_job:
            ingestion_job.mark_failed("INGESTION_ERROR", str(exc))

        return {
            "status": "failed",
            "chunks_indexed": 0,
            "message": "Ingestion failed.",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Compatibility wrapper for existing queue worker
# ---------------------------------------------------------------------------

def ingest_document_legacy(
    document_id: str,
    stored_path: Path,
    filename: str,
) -> None:
    """Legacy compatibility wrapper for existing ingestion queue.

    This function maintains the same signature as the old ingest_document
    but uses the new V2 pipeline internally.
    """
    from backend.ingestion import _update

    try:
        _update(document_id, status="processing", message="Ingesting with V2 pipeline...", error=None)
        result = ingest_document_v2(document_id, stored_path, filename)

        if result["status"] == "complete":
            _update(
                document_id,
                status="complete",
                chunks_indexed=result["chunks_indexed"],
                message=result["message"],
                error=None,
            )
        else:
            _update(
                document_id,
                status="failed",
                message="Ingestion failed.",
                error=result["error"],
            )
    except Exception as exc:
        log.exception("Legacy ingestion failed for %s", document_id)
        _update(document_id, status="failed", message="Ingestion failed.", error=str(exc))


# ---------------------------------------------------------------------------
# Document hashing for deduplication
# ---------------------------------------------------------------------------

async def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Helper for queue worker
# ---------------------------------------------------------------------------

def submit_ingestion_job(
    document_id: str,
    stored_path: Path,
    filename: str,
) -> bool:
    """Submit ingestion job to queue (uses existing queue)."""
    from backend.ingestion_queue import submit
    return submit(document_id, stored_path, filename)