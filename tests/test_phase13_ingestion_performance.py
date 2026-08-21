"""Phase 13 — Ingestion Performance, Progress Tracking & Frontend Polling Tests."""
import json
import pytest
from unittest.mock import MagicMock, patch

from backend.schemas import DocumentStatus
from backend.settings import EMBEDDING_BATCH_SIZE, FASTEMBED_BATCH_SIZE
from backend.vector_store import add_documents
from langchain_core.documents import Document


class TestBatchSizeDefaults:
    def test_fastembed_batch_size_is_64(self):
        assert FASTEMBED_BATCH_SIZE == 64

    def test_embedding_batch_size_is_64(self):
        assert EMBEDDING_BATCH_SIZE == 64


class TestProgressCallback:
    def test_progress_callback_called_per_batch(self):
        docs = [Document(page_content=f"Chunk {i}") for i in range(10)]
        ids = [f"id_{i}" for i in range(10)]

        callback_calls = []

        def on_progress(added: int, total: int):
            callback_calls.append((added, total))

        with patch("backend.vector_store.get_vector_store") as mock_get_vs:
            mock_vs = MagicMock()
            mock_get_vs.return_value = mock_vs

            with patch("backend.vector_store.EMBEDDING_BATCH_SIZE", 4):
                added_count = add_documents(docs, ids, progress_callback=on_progress)
                assert added_count == 10
                # 10 docs with batch_size 4 => 3 batches (4, 8, 10)
                assert len(callback_calls) == 3
                assert callback_calls[0] == (4, 10)
                assert callback_calls[1] == (8, 10)
                assert callback_calls[2] == (10, 10)


class TestDocumentStatusSchema:
    def test_document_status_optional_progress_fields(self):
        doc = DocumentStatus(
            document_id="doc-123",
            filename="test.pdf",
            status="indexing",
            chunks_indexed=50,
            total_chunks=100,
            progress_pct=50.0,
            message="Indexing 50/100 chunks...",
            uploaded_at="2026-08-22T00:00:00Z",
        )
        data = doc.model_dump()
        assert data["total_chunks"] == 100
        assert data["progress_pct"] == 50.0
        assert data["chunks_indexed"] == 50

    def test_document_status_backwards_compatibility(self):
        doc = DocumentStatus(
            document_id="doc-456",
            filename="test.txt",
            status="complete",
            uploaded_at="2026-08-22T00:00:00Z",
        )
        data = doc.model_dump()
        assert data["total_chunks"] is None
        assert data["progress_pct"] is None
        assert data["chunks_indexed"] == 0
