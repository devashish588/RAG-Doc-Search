"""Service interfaces (ABCs) for HybridRAG components.

These define contracts for future implementations without implementing
the algorithms themselves. Phase 1 only establishes the interfaces;
concrete implementations remain in their existing modules.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel

from backend.models import Chunk


class BaseLoader(ABC):
    """Load raw documents from various sources."""

    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """Load and return a list of LangChain Documents."""
        ...

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return set of supported file extensions (e.g., {'.pdf', '.txt'})."""
        ...


class BaseNormalizer(ABC):
    """Normalize/clean raw document text before chunking."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Return normalized text."""
        ...


class BaseChunker(ABC):
    """Split documents into chunks with stable identities."""

    @abstractmethod
    def chunk(
        self,
        documents: list[Document],
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        """Split documents into canonical Chunks with stable IDs."""
        ...


class RetrievalResult(BaseModel):
    """Canonical retrieval result structure."""
    chunk_id: str
    score: float
    rank: int
    source: str
    content: str
    metadata: dict[str, Any] = {}


class RetrievalTrace(BaseModel):
    """Structured trace of retrieval stages."""
    dense: list[RetrievalResult] = []
    bm25: list[RetrievalResult] = []
    rrf: list[RetrievalResult] = []
    reranker: list[RetrievalResult] = []


class BaseDenseRetriever(ABC):
    """Dense vector retrieval interface."""

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 10,
        source: str | None = None,
    ) -> list[RetrievalResult]:
        """Return top-k dense retrieval results with scores."""
        ...


class BaseSparseRetriever(ABC):
    """Sparse (e.g., BM25) retrieval interface. Not implemented in Phase 1."""

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 10,
        source: str | None = None,
    ) -> list[RetrievalResult]:
        """Return top-k sparse retrieval results with scores."""
        ...

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Add chunks to the sparse index."""
        ...

    @abstractmethod
    def delete(self, document_id: str) -> None:
        """Remove all chunks for a document from the sparse index."""
        ...


class BaseReranker(ABC):
    """Reranking interface. Not implemented in Phase 1."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank candidates and return top-k."""
        ...


class BaseGenerator(ABC):
    """Answer generation interface."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
    ) -> str:
        """Generate an answer from retrieved context."""
        ...


class BaseVerifier(ABC):
    """Answer verification/grounding interface. Not implemented in Phase 1."""

    @abstractmethod
    def verify(
        self,
        answer: str,
        context: list[RetrievalResult],
    ) -> dict[str, Any]:
        """Return verification metrics (faithfulness, citation accuracy, etc.)."""
        ...


class BaseIndexHealth(ABC):
    """Index reconciliation and health checking."""

    @abstractmethod
    def check(self) -> dict[str, Any]:
        """Return health report with detected issues."""
        ...