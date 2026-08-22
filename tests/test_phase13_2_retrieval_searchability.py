"""Phase 13.2 — Retrieval & Searchability Integration & Diagnostic Tests."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.ingestion import register_document, get_document_status, list_document_statuses
from backend.ingestion_v2 import ingest_document_v2
from backend.vector_store import get_vector_store
from backend.bm25_index import get_bm25_index
from backend.retrieval_dense import get_dense_retriever
from backend.retrieval_bm25 import get_bm25_retriever
from backend.retrieval_hybrid import get_hybrid_retriever
from backend.retrieval import run_search
from backend.schemas import SearchRequest
from backend.api_v1 import ask_v1, AskRequest


class TestRetrievalSearchabilityEndToEnd:
    @pytest.fixture
    def unique_doc(self, tmp_path):
        doc_file = tmp_path / "quantum_computing_spec.txt"
        doc_file.write_text(
            "Quantum Computing Specification 2026.\n"
            "Superconducting qubits utilize Josephson junctions for quantum gate operations.\n"
            "Surface code error correction is deployed to maintain fault-tolerant logical qubits.",
            encoding="utf-8"
        )
        doc_id = f"test_quantum_{tmp_path.name}"
        filename = "quantum_computing_spec.txt"
        register_document(doc_id, filename, doc_file)
        res = ingest_document_v2(doc_id, doc_file, filename)
        assert res["status"] == "complete"
        return doc_id, filename, doc_file

    def test_ingestion_reaches_complete(self, unique_doc):
        doc_id, filename, _ = unique_doc
        status = get_document_status(doc_id)
        assert status is not None
        assert status["status"] == "complete"

    def test_final_progress_is_100(self, unique_doc):
        doc_id, _, _ = unique_doc
        status = get_document_status(doc_id)
        assert status["progress_pct"] == 100.0

    def test_no_false_100_percent_progress(self, tmp_path):
        doc_file = tmp_path / "progress_cap_check.txt"
        doc_file.write_text("Line 1 of content.\nLine 2 of content.", encoding="utf-8")
        doc_id = "cap_check_123"
        register_document(doc_id, "progress_cap_check.txt", doc_file)

        recorded = []
        with patch("backend.ingestion._update", side_effect=lambda did, **k: recorded.append(dict(k))):
            ingest_document_v2(doc_id, doc_file, "progress_cap_check.txt")

        indexing_records = [r for r in recorded if r.get("status") == "indexing"]
        for r in indexing_records:
            assert r.get("progress_pct") <= 99.0

        assert recorded[-1].get("status") == "complete"
        assert recorded[-1].get("progress_pct") == 100.0

    def test_failed_ingestion_reaches_failed_state(self, tmp_path):
        doc_file = tmp_path / "corrupt.txt"
        doc_file.write_text("Some text", encoding="utf-8")
        doc_id = "fail_test_1"
        register_document(doc_id, "corrupt.txt", doc_file)

        with patch("backend.ingestion_v2.add_documents", side_effect=RuntimeError("Vector Store Write Failed")):
            res = ingest_document_v2(doc_id, doc_file, "corrupt.txt")
            assert res["status"] == "failed"

        status = get_document_status(doc_id)
        assert status["status"] == "failed"

    def test_document_status_persists_after_completion(self, unique_doc):
        doc_id, _, _ = unique_doc
        status = get_document_status(doc_id)
        assert status["status"] == "complete"
        all_statuses = list_document_statuses()
        found = [s for s in all_statuses if s["document_id"] == doc_id]
        assert len(found) == 1
        assert found[0]["status"] == "complete"

    def test_complete_requires_vector_store_success(self, unique_doc):
        doc_id, _, _ = unique_doc
        vs = get_vector_store()
        col = getattr(vs, "_collection", None)
        assert col is not None
        get_res = col.get(where={"document_id": doc_id})
        assert len(get_res.get("ids", [])) > 0

    def test_expected_non_duplicate_chunk_ids_exist(self, unique_doc):
        doc_id, _, _ = unique_doc
        vs = get_vector_store()
        col = getattr(vs, "_collection", None)
        get_res = col.get(where={"document_id": doc_id})
        ids = get_res.get("ids", [])
        assert f"{doc_id}:0" in ids

    def test_vector_store_failure_marks_failed(self, tmp_path):
        doc_file = tmp_path / "vs_fail.txt"
        doc_file.write_text("Text", encoding="utf-8")
        doc_id = "vs_fail_id"
        register_document(doc_id, "vs_fail.txt", doc_file)
        with patch("backend.ingestion_v2.add_documents", side_effect=RuntimeError("Chroma Disk Full")):
            res = ingest_document_v2(doc_id, doc_file, "vs_fail.txt")
            assert res["status"] == "failed"

    def test_bm25_update_failure_marks_failed(self, tmp_path):
        doc_file = tmp_path / "bm25_fail.txt"
        doc_file.write_text("Text", encoding="utf-8")
        doc_id = "bm25_fail_id"
        register_document(doc_id, "bm25_fail.txt", doc_file)
        with patch("backend.ingestion_v2.get_bm25_index", side_effect=RuntimeError("BM25 Pickling Error")):
            res = ingest_document_v2(doc_id, doc_file, "bm25_fail.txt")
            assert res["status"] == "failed"

    def test_bm25_index_contains_expected_chunks(self, unique_doc):
        doc_id, _, _ = unique_doc
        bm25 = get_bm25_index()
        assert f"{doc_id}:0" in bm25._chunk_ids

    def test_document_is_searchable_after_completion(self, unique_doc):
        doc_id, _, _ = unique_doc
        search_res = run_search(SearchRequest(query="Josephson junctions superconducting qubits", top_k=5))
        assert len(search_res.results) > 0
        matching = [r for r in search_res.results if "Josephson junctions" in r.text]
        assert len(matching) > 0

    @pytest.mark.anyio
    async def test_v1_ask_can_retrieve_new_document(self, unique_doc):
        doc_id, filename, _ = unique_doc
        req = AskRequest(
            question="What do superconducting qubits utilize for quantum gate operations?",
            retrieval_mode="hybrid_rerank",
            top_k_dense=5,
            top_k_sparse=5,
            top_k_fused=10,
            top_k_final=5,
        )
        res = await ask_v1(req)
        assert res.status in ("answered", "insufficient_context")
        # Ensure retrieval trace contains results for unique document
        all_retrieved = (res.retrieval_trace.dense or []) + (res.retrieval_trace.bm25 or []) + (res.retrieval_trace.rrf or [])
        target_retrieved = [r for r in all_retrieved if r.metadata.get("document_id") == doc_id or filename in r.source]
        assert len(target_retrieved) > 0

    @pytest.mark.anyio
    async def test_negative_query_abstention(self, unique_doc):
        doc_id, filename, _ = unique_doc
        req = AskRequest(
            question="What is the recipe for baking chocolate lava cake?",
            retrieval_mode="dense",
            top_k_dense=5,
        )
        res = await ask_v1(req)
        assert res.confidence.abstention_flag is True
        assert res.status == "insufficient_context"
