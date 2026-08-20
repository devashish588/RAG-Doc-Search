"""Chunking strategies for document ingestion.

Implements the BaseChunker interface with fixed, recursive, and semantic strategies.
"""
import hashlib
import logging
import re
from abc import ABC
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.interfaces import BaseChunker
from backend.models import Chunk

log = logging.getLogger(__name__)


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def estimate_token_count(text: str) -> int:
    """Rough token count estimation (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


def extract_section_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Extract section info from document metadata."""
    # Check for explicit section
    if "section" in metadata:
        return metadata["section"]

    # Check for Markdown sections
    if "sections" in metadata and metadata["sections"]:
        # Use the last (deepest) section
        return metadata["sections"][-1]

    # Check for header-like content in metadata
    if "header" in metadata:
        return metadata["header"]

    return None


class BaseChunkerImpl(BaseChunker):
    """Base implementation with common chunking utilities."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy_name = "base"

    def _create_chunk(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        section: str | None = None,
        page: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """Create a canonical Chunk with all required fields."""
        content_hash = compute_content_hash(content)
        return Chunk.create(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            content_hash=content_hash,
            section=section,
            page=page,
            chunk_strategy=self.strategy_name,
            token_count=estimate_token_count(content),
            metadata=metadata or {},
        )

    def _get_page_from_metadata(self, metadata: dict[str, Any]) -> int | None:
        """Extract page number from metadata."""
        page = metadata.get("page")
        if isinstance(page, int):
            return page
        if isinstance(page, str) and page.isdigit():
            return int(page)
        return None


class FixedChunker(BaseChunkerImpl):
    """Fixed-size chunker with deterministic boundaries."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.strategy_name = "fixed"

    def chunk(
        self,
        documents: list[Document],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        """Split documents into fixed-size chunks with overlap."""
        chunks = []
        chunk_index = 0

        for doc in documents:
            text = doc.page_content
            meta = dict(doc.metadata)

            # Extract section info
            section = extract_section_from_metadata(meta)
            page = self._get_page_from_metadata(meta)

            # Fixed-size chunking with overlap
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunk_meta = dict(meta)
                    chunk_meta.update(source=filename)

                    chunk = self._create_chunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=chunk_text,
                        section=section,
                        page=page,
                        metadata=chunk_meta,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                if end >= len(text):
                    break

                # Move start by (chunk_size - overlap)
                start += self.chunk_size - self.chunk_overlap

        return chunks


class RecursiveChunker(BaseChunkerImpl):
    """Recursive character-based chunker (preserves existing behavior)."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.strategy_name = "recursive"

    def chunk(
        self,
        documents: list[Document],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        """Split documents using recursive character splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

        chunks = []
        chunk_index = 0

        for doc in documents:
            text = doc.page_content
            meta = dict(doc.metadata)

            section = extract_section_from_metadata(meta)
            page = self._get_page_from_metadata(meta)

            # Split using recursive splitter
            split_docs = splitter.split_documents([doc])

            for split_doc in split_docs:
                chunk_text = split_doc.page_content.strip()
                if not chunk_text:
                    continue

                chunk_meta = dict(split_doc.metadata)
                chunk_meta.update(source=filename)

                chunk = self._create_chunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=chunk_text,
                    section=section,
                    page=page,
                    metadata=chunk_meta,
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks


class SemanticChunker(BaseChunkerImpl):
    """Semantic chunker that attempts to preserve semantic boundaries.

    Uses sentence embeddings to find natural break points.
    Falls back to recursive chunking if embeddings unavailable.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int | None = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.strategy_name = "semantic"
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size or (chunk_size * 2)
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load sentence transformer for semantic chunking."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use a small, fast model
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                log.info("Loaded sentence transformer for semantic chunking")
            except ImportError:
                log.warning("sentence-transformers not available, falling back to recursive")
                self._embedder = False
            except Exception as exc:
                log.warning("Failed to load sentence transformer: %s, falling back to recursive", exc)
                self._embedder = False
        return self._embedder

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be enhanced with nltk/spacy
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarities(self, sentences: list[str]) -> list[float]:
        """Compute cosine similarities between adjacent sentences."""
        embedder = self._get_embedder()
        if not embedder:
            return []

        try:
            embeddings = embedder.encode(sentences, normalize_embeddings=True)
            similarities = []
            for i in range(len(embeddings) - 1):
                # Cosine similarity (embeddings are normalized)
                sim = float(embeddings[i] @ embeddings[i + 1])
                similarities.append(sim)
            return similarities
        except Exception as exc:
            log.warning("Failed to compute similarities: %s", exc)
            return []

    def _find_break_points(
        self,
        sentences: list[str],
        similarities: list[float],
    ) -> list[int]:
        """Find semantic break points based on similarity drops."""
        if not similarities:
            # Fallback: split roughly by chunk_size
            break_points = []
            current_len = 0
            for i, sent in enumerate(sentences):
                current_len += len(sent)
                if current_len >= self.chunk_size:
                    break_points.append(i)
                    current_len = len(sent)
            return break_points

        # Find significant similarity drops
        break_points = []
        threshold = 0.7  # Configurable threshold for semantic boundary

        for i, sim in enumerate(similarities):
            if sim < threshold:
                break_points.append(i + 1)

        # Ensure chunks are within size bounds
        final_breaks = []
        last_break = 0

        for bp in break_points:
            chunk_text = " ".join(sentences[last_break:bp])
            if len(chunk_text) >= self.min_chunk_size:
                if len(chunk_text) <= self.max_chunk_size:
                    final_breaks.append(bp)
                    last_break = bp
                else:
                    # Chunk too large, force break at reasonable point
                    mid = last_break + (bp - last_break) // 2
                    if mid > last_break:
                        final_breaks.append(mid)
                        last_break = mid

        # Add remaining as final chunk
        if last_break < len(sentences):
            chunk_text = " ".join(sentences[last_break:])
            if len(chunk_text) >= self.min_chunk_size:
                final_breaks.append(len(sentences))

        return final_breaks

    def chunk(
        self,
        documents: list[Document],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        """Split documents using semantic boundaries."""
        chunks = []
        chunk_index = 0

        for doc in documents:
            text = doc.page_content
            meta = dict(doc.metadata)

            section = extract_section_from_metadata(meta)
            page = self._get_page_from_metadata(meta)

            # Split into sentences
            sentences = self._split_sentences(text)
            if not sentences:
                continue

            # Compute similarities
            similarities = self._compute_similarities(sentences)

            # Find semantic break points
            break_points = self._find_break_points(sentences, similarities)

            if not break_points:
                # Fallback to recursive
                log.info("Semantic chunking found no boundaries, falling back to recursive")
                recursive = RecursiveChunker(self.chunk_size, self.chunk_overlap)
                return recursive.chunk(documents, document_id, filename)

            # Create chunks from break points
            last_break = 0
            for bp in break_points:
                chunk_text = " ".join(sentences[last_break:bp]).strip()
                if not chunk_text:
                    last_break = bp
                    continue

                chunk_meta = dict(meta)
                chunk_meta.update(source=filename)

                chunk = self._create_chunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=chunk_text,
                    section=section,
                    page=page,
                    metadata=chunk_meta,
                )
                chunks.append(chunk)
                chunk_index += 1
                last_break = bp

        return chunks


# ---------------------------------------------------------------------------
# Chunker factory
# ---------------------------------------------------------------------------

_CHUNKER_REGISTRY: dict[str, type[BaseChunkerImpl]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def get_chunker(
    strategy: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> BaseChunkerImpl:
    """Get chunker instance by strategy name."""
    strategy = strategy.lower()
    if strategy not in _CHUNKER_REGISTRY:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Available: {list(_CHUNKER_REGISTRY)}")

    chunker_class = _CHUNKER_REGISTRY[strategy]
    return chunker_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def available_strategies() -> list[str]:
    """Return list of available chunking strategies."""
    return list(_CHUNKER_REGISTRY.keys())