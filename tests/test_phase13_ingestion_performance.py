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


class TestPhase13CompletionAndSearchability:
    def test_ingestion_reaches_complete_and_persists(self, tmp_path):
        from backend.ingestion import register_document, get_document_status
        from backend.ingestion_v2 import ingest_document_v2
        from backend.vector_store import get_vector_store

        test_file = tmp_path / "test_doc.txt"
        test_file.write_text("The HybridRAG system uses ChromaDB for vector storage and semantic document search.", encoding="utf-8")
        doc_id = "test_completion_123"

        register_document(doc_id, "test_doc.txt", test_file)

        res = ingest_document_v2(doc_id, test_file, "test_doc.txt")
        assert res["status"] == "complete"
        assert res["chunks_indexed"] > 0

        # Verify global registry record
        status = get_document_status(doc_id)
        assert status is not None
        assert status["status"] == "complete"
        assert status["chunks_indexed"] == res["chunks_indexed"]
        assert status["total_chunks"] == res["chunks_indexed"]
        assert status["progress_pct"] == 100.0
        assert status["error"] is None

    def test_no_false_100_percent_progress(self, tmp_path):
        from backend.ingestion import register_document, get_document_status
        from backend.ingestion_v2 import ingest_document_v2

        test_file = tmp_path / "progress_doc.txt"
        test_file.write_text("Chunk one content for progress test.\nChunk two content for progress test.", encoding="utf-8")
        doc_id = "test_progress_cap"

        register_document(doc_id, "progress_doc.txt", test_file)

        recorded_statuses = []

        with patch("backend.ingestion._update", side_effect=lambda did, **k: recorded_statuses.append(dict(k))):
            ingest_document_v2(doc_id, test_file, "progress_doc.txt")

        # Check indexing statuses during vector progress
        indexing_records = [r for r in recorded_statuses if r.get("status") == "indexing"]
        for r in indexing_records:
            assert r.get("progress_pct") <= 99.0, f"False 100% progress emitted during indexing: {r}"

        # Final record must be complete with 100%
        final_record = recorded_statuses[-1]
        assert final_record.get("status") == "complete"
        assert final_record.get("progress_pct") == 100.0

    def test_document_is_searchable_after_completion(self, tmp_path):
        from backend.ingestion import register_document
        from backend.ingestion_v2 import ingest_document_v2
        from backend.retrieval import run_search
        from backend.schemas import SearchRequest

        test_file = tmp_path / "searchable_doc.txt"
        test_file.write_text("Syllabus for System Discipline course covering architectural patterns and reliability.", encoding="utf-8")
        doc_id = "test_searchable_789"

        register_document(doc_id, "searchable_doc.txt", test_file)
        res = ingest_document_v2(doc_id, test_file, "searchable_doc.txt")
        assert res["status"] == "complete"

        # Dense search query
        response = run_search(SearchRequest(query="System Discipline syllabus architectural patterns", top_k=5))
        assert len(response.results) > 0
        matching = [r for r in response.results if "System Discipline" in r.text]
        assert len(matching) > 0

    def test_bm25_failure_marks_failed(self, tmp_path):
        from backend.ingestion import register_document, get_document_status
        from backend.ingestion_v2 import ingest_document_v2

        test_file = tmp_path / "bm25_fail_doc.txt"
        test_file.write_text("Test content for BM25 failure simulation.", encoding="utf-8")
        doc_id = "test_bm25_fail"

        register_document(doc_id, "bm25_fail_doc.txt", test_file)

        with patch("backend.ingestion_v2.get_bm25_index", side_effect=RuntimeError("BM25 index lock failure")):
            res = ingest_document_v2(doc_id, test_file, "bm25_fail_doc.txt")
            assert res["status"] == "failed"

        status = get_document_status(doc_id)
        assert status is not None
        assert status["status"] == "failed"
        assert "BM25" in str(status.get("error"))
