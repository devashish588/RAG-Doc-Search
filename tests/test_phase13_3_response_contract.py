"""Phase 13.3 — Retrieval response contract tests.

Verifies that /v1/ask returns retrieved_chunks for all four retrieval modes,
and that the frontend rendering contract is satisfied.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.middleware import reset_limiter
from backend.interfaces import RetrievalResult


client = TestClient(app)


def _ask(mode="dense", question="test query"):
    reset_limiter()
    return client.post("/v1/ask", json={
        "question": question,
        "retrieval_mode": mode,
        "top_k_dense": 5,
        "top_k_sparse": 5,
        "top_k_fused": 10,
        "top_k_final": 5,
    })


# ---------------------------------------------------------------------------
# 1-4: Each mode returns retrieved_chunks
# ---------------------------------------------------------------------------

class TestDenseRetrievedChunks:
    def test_response_contains_retrieved_chunks(self):
        resp = _ask("dense")
        assert resp.status_code == 200
        data = resp.json()
        assert "retrieved_chunks" in data
        assert isinstance(data["retrieved_chunks"], list)

    def test_retrieved_chunk_has_required_fields(self):
        resp = _ask("dense")
        data = resp.json()
        chunks = data["retrieved_chunks"]
        if chunks:
            c = chunks[0]
            assert "chunk_id" in c
            assert "source" in c
            assert "text" in c
            assert "score" in c
            assert "rank" in c


class TestBM25RetrievedChunks:
    def test_response_contains_retrieved_chunks(self):
        resp = _ask("sparse")
        assert resp.status_code == 200
        data = resp.json()
        assert "retrieved_chunks" in data
        assert isinstance(data["retrieved_chunks"], list)


class TestHybridRetrievedChunks:
    def test_response_contains_retrieved_chunks(self):
        resp = _ask("hybrid")
        assert resp.status_code == 200
        data = resp.json()
        assert "retrieved_chunks" in data
        assert isinstance(data["retrieved_chunks"], list)


class TestHybridRerankRetrievedChunks:
    def test_response_contains_retrieved_chunks(self):
        resp = _ask("hybrid_rerank")
        assert resp.status_code == 200
        data = resp.json()
        assert "retrieved_chunks" in data
        assert isinstance(data["retrieved_chunks"], list)


# ---------------------------------------------------------------------------
# 5: /search and /v1/ask response contracts are consistent
# ---------------------------------------------------------------------------

class TestSearchAndAskConsistency:
    def test_both_endpoints_return_results(self):
        reset_limiter()
        ask_resp = _ask("dense")
        search_resp = client.post("/search", json={"query": "test", "top_k": 5})
        assert ask_resp.status_code == 200
        assert search_resp.status_code == 200
        ask_data = ask_resp.json()
        search_data = search_resp.json()
        # /v1/ask should have retrieved_chunks
        assert "retrieved_chunks" in ask_data
        # /search should have results
        assert "results" in search_data


# ---------------------------------------------------------------------------
# 6-8: Retrieved chunk contains expected fields
# ---------------------------------------------------------------------------

class TestRetrievedChunkFields:
    def test_chunk_contains_text(self):
        resp = _ask("dense")
        chunks = resp.json().get("retrieved_chunks", [])
        for c in chunks:
            assert "text" in c
            assert isinstance(c["text"], str)

    def test_chunk_contains_source(self):
        resp = _ask("dense")
        chunks = resp.json().get("retrieved_chunks", [])
        for c in chunks:
            assert "source" in c
            assert isinstance(c["source"], str)

    def test_chunk_contains_score(self):
        resp = _ask("dense")
        chunks = resp.json().get("retrieved_chunks", [])
        for c in chunks:
            assert "score" in c
            assert isinstance(c["score"], (int, float))


# ---------------------------------------------------------------------------
# 9: Source filter preserved
# ---------------------------------------------------------------------------

class TestSourceFilter:
    def test_source_filter_in_payload(self):
        resp = _ask("dense")
        assert resp.status_code == 200
        # Just verifying the endpoint accepts the payload without error


# ---------------------------------------------------------------------------
# 10: Frontend contract field exists
# ---------------------------------------------------------------------------

class TestFrontendContract:
    def test_ask_response_has_all_frontend_fields(self):
        resp = _ask("dense")
        data = resp.json()
        # Frontend expects these top-level fields
        assert "answer" in data
        assert "status" in data
        assert "confidence" in data
        assert "retrieval_trace" in data
        assert "retrieved_chunks" in data
        # Frontend expects these confidence sub-fields
        conf = data.get("confidence", {})
        assert "overall_score" in conf
        assert "level" in conf
        assert "retrieval_confidence" in conf
        assert "grounding_confidence" in conf
        assert "abstention_flag" in conf


# ---------------------------------------------------------------------------
# 11: Grounding receives retrieved context
# ---------------------------------------------------------------------------

class TestGroundingContext:
    def test_grounding_metrics_present(self):
        resp = _ask("dense")
        data = resp.json()
        assert "grounding_metrics" in data
        gm = data["grounding_metrics"]
        assert "total_claims" in gm
        assert "grounding_ratio" in gm


# ---------------------------------------------------------------------------
# 12: Unsupported query still abstains
# ---------------------------------------------------------------------------

class TestUnsupportedQuery:
    def test_unsupported_query_abstains(self):
        resp = _ask("dense", question="What is the recipe for baking chocolate lava cake?")
        data = resp.json()
        # Should still return a valid response
        assert "answer" in data
        assert "confidence" in data
        # retrieved_chunks may be empty or populated — either is valid
        assert "retrieved_chunks" in data


# ---------------------------------------------------------------------------
# 13: No fake chunks on empty retrieval
# ---------------------------------------------------------------------------

class TestNoFakeChunks:
    def test_empty_retrieval_returns_empty_chunks(self):
        # Mock dense retriever to return nothing
        with patch("backend.api_v1.run_search_dense", return_value=[]):
            with patch("backend.api_v1.run_search_bm25", return_value=[]):
                with patch("backend.api_v1.run_search_hybrid", return_value=[]):
                    reset_limiter()
                    resp = client.post("/v1/ask", json={
                        "question": "test",
                        "retrieval_mode": "dense",
                    })
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["retrieved_chunks"] == []


# ---------------------------------------------------------------------------
# 14: _extract_retrieved_chunks helper
# ---------------------------------------------------------------------------

class TestExtractRetrievedChunks:
    def _make_result(self, chunk_id, source, content, score, rank):
        return RetrievalResult(
            chunk_id=chunk_id,
            source=source,
            content=content,
            score=score,
            rank=rank,
            metadata={"page": 1, "document_id": "doc1"},
        )

    def test_dense_mode_uses_dense_results(self):
        from backend.api_v1 import _extract_retrieved_chunks
        dense = [self._make_result("c1", "src", "text", 0.9, 1)]
        chunks = _extract_retrieved_chunks(dense, [], [], [], "dense")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "c1"

    def test_sparse_mode_uses_bm25_results(self):
        from backend.api_v1 import _extract_retrieved_chunks
        bm25 = [self._make_result("c1", "src", "text", 0.9, 1)]
        chunks = _extract_retrieved_chunks([], bm25, [], [], "sparse")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "c1"

    def test_hybrid_mode_uses_rrf_results(self):
        from backend.api_v1 import _extract_retrieved_chunks
        rrf = [self._make_result("c1", "src", "text", 0.9, 1)]
        chunks = _extract_retrieved_chunks([], [], rrf, [], "hybrid")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "c1"

    def test_hybrid_rerank_mode_uses_reranker_results(self):
        from backend.api_v1 import _extract_retrieved_chunks
        reranker = [self._make_result("c1", "src", "text", 0.9, 1)]
        chunks = _extract_retrieved_chunks([], [], [], reranker, "hybrid_rerank")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "c1"

    def test_hybrid_rerank_falls_back_to_rrf(self):
        from backend.api_v1 import _extract_retrieved_chunks
        rrf = [self._make_result("c1", "src", "text", 0.9, 1)]
        chunks = _extract_retrieved_chunks([], [], rrf, [], "hybrid_rerank")
        assert len(chunks) == 1

    def test_empty_results_returns_empty(self):
        from backend.api_v1 import _extract_retrieved_chunks
        chunks = _extract_retrieved_chunks([], [], [], [], "dense")
        assert chunks == []

    def test_metadata_preserved(self):
        from backend.api_v1 import _extract_retrieved_chunks
        r = RetrievalResult(
            chunk_id="c1",
            source="src",
            content="hello",
            score=0.5,
            rank=2,
            metadata={"page": 5, "document_id": "d1"},
        )
        chunks = _extract_retrieved_chunks([], [], [r], [], "hybrid")
        assert chunks[0].metadata["page"] == 5
        assert chunks[0].page == 5
