"""Deduplication utilities for ingestion pipeline.

Provides exact and near-duplicate detection for documents and chunks.
"""
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.models import Chunk

log = logging.getLogger(__name__)


@dataclass
class DuplicateResult:
    """Result of duplicate detection."""
    is_duplicate: bool
    duplicate_type: str | None = None  # "exact_document", "exact_chunk", "near_chunk"
    existing_id: str | None = None
    similarity: float | None = None
    details: dict[str, Any] | None = None


def compute_document_hash(content: bytes) -> str:
    """Compute SHA-256 hash of document content."""
    return hashlib.sha256(content).hexdigest()


def compute_chunk_hash(content: str) -> str:
    """Compute SHA-256 hash of chunk content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class Deduplicator:
    """Handles exact and near-duplicate detection for chunks."""

    def __init__(
        self,
        near_duplicate_threshold: float = 0.95,
        enable_near_duplicate: bool = True,
    ):
        self.near_duplicate_threshold = near_duplicate_threshold
        self.enable_near_duplicate = enable_near_duplicate
        self._seen_chunk_hashes: set[str] = set()
        self._chunk_embeddings: dict[str, np.ndarray] = {}

    def check_exact_document(self, content_hash: str, existing_hashes: set[str]) -> DuplicateResult:
        """Check for exact document duplicate."""
        if content_hash in existing_hashes:
            return DuplicateResult(
                is_duplicate=True,
                duplicate_type="exact_document",
                existing_id=content_hash,
                details={"hash": content_hash},
            )
        return DuplicateResult(is_duplicate=False)

    def check_exact_chunk(self, chunk: Chunk) -> DuplicateResult:
        """Check for exact chunk duplicate using content hash."""
        if chunk.content_hash in self._seen_chunk_hashes:
            return DuplicateResult(
                is_duplicate=True,
                duplicate_type="exact_chunk",
                existing_id=chunk.content_hash,
                details={"content_hash": chunk.content_hash, "chunk_id": chunk.id},
            )
        return DuplicateResult(is_duplicate=False)

    def check_near_duplicate(
        self,
        chunk: Chunk,
        embedder_fn=None,
    ) -> DuplicateResult:
        """Check for near-duplicate chunks using embedding similarity."""
        if not self.enable_near_duplicate:
            return DuplicateResult(is_duplicate=False)

        if embedder_fn is None:
            return DuplicateResult(is_duplicate=False)

        try:
            # Get embedding for this chunk
            embedding = embedder_fn(chunk.content)

            # Compare against existing embeddings
            for existing_id, existing_emb in self._chunk_embeddings.items():
                # Cosine similarity (assuming normalized embeddings)
                similarity = float(np.dot(embedding, existing_emb))
                if similarity >= self.near_duplicate_threshold:
                    return DuplicateResult(
                        is_duplicate=True,
                        duplicate_type="near_chunk",
                        existing_id=existing_id,
                        similarity=similarity,
                        details={
                            "chunk_id": chunk.id,
                            "existing_chunk_id": existing_id,
                            "threshold": self.near_duplicate_threshold,
                        },
                    )
        except Exception as exc:
            log.warning("Near-duplicate check failed for chunk %s: %s", chunk.id, exc)

        return DuplicateResult(is_duplicate=False)

    def register_chunk(self, chunk: Chunk, embedding: np.ndarray | None = None) -> None:
        """Register a chunk as seen."""
        self._seen_chunk_hashes.add(chunk.content_hash)
        if embedding is not None and self.enable_near_duplicate:
            self._chunk_embeddings[chunk.id] = embedding

    def clear(self) -> None:
        """Clear deduplication state."""
        self._seen_chunk_hashes.clear()
        self._chunk_embeddings.clear()


def deduplicate_chunks(
    chunks: list[Chunk],
    near_duplicate_threshold: float = 0.95,
    embedder_fn=None,
) -> tuple[list[Chunk], list[DuplicateResult]]:
    """Deduplicate a list of chunks.

    Returns:
        - deduplicated chunks (kept)
        - list of duplicate detection results for all chunks
    """
    deduplicator = Deduplicator(near_duplicate_threshold=near_duplicate_threshold)
    kept = []
    results = []

    for chunk in chunks:
        # Check exact chunk duplicate
        exact_result = deduplicator.check_exact_chunk(chunk)
        if exact_result.is_duplicate:
            results.append(exact_result)
            log.info("Skipping exact duplicate chunk: %s", chunk.id)
            continue

        # Check near duplicate
        near_result = deduplicator.check_near_duplicate(chunk, embedder_fn)
        if near_result.is_duplicate:
            results.append(near_result)
            log.info(
                "Skipping near duplicate chunk: %s (similar to %s, sim=%.4f)",
                chunk.id,
                near_result.existing_id,
                near_result.similarity,
            )
            continue

        # Keep chunk
        kept.append(chunk)
        results.append(DuplicateResult(is_duplicate=False))
        deduplicator.register_chunk(chunk)

    return kept, results