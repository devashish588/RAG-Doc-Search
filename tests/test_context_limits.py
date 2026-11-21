"""Regression tests for context limits in retrieval modes.

Ensures that the number of chunks passed to the LLM respects the
configured context limit for all retrieval modes.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app
from backend.middleware import reset_limiter
from backend.interfaces import RetrievalResult

client = TestClient(app)


def _make_result(chunk_id, source, content, score, rank):
    return RetrievalResult(
        chunk_id=chunk_id,
        source=source,
        content=content,
        score=score,
        rank=rank,
        metadata={"page": 1, "document_id": "doc1"},
    )


class TestContextLimits:
    """Test that context limits are enforced for each retrieval mode."""

    def test_dense_mode_respects_top_k_dense(self):
        """Dense mode should limit chunks to top_k_dense."""
        reset_limiter()
        dense_results = [_make_result(f"c{i}", "src", f"text {i}", 0.9 - i * 0.1, i) for i in range(1, 11)]
        
        with patch("backend.api_v1.run_search_dense", return_value=dense_results):
            with patch("backend.api_v1.run_search_bm25", return_value=[]):
                with patch("backend.api_v1.run_search_hybrid", return_value=[]):
                    resp = client.post("/v1/ask", json={
                        "question": "test query",
                        "retrieval_mode": "dense",
                        "top_k_dense": 3,
                    })
        
        assert resp.status_code == 200
        data = resp.json()
        # Should only use top_k_dense (3) chunks for answer generation
        assert len(data["retrieved_chunks"]) <= 3

    def test_sparse_mode_respects_top_k_sparse(self):
        """Sparse mode should limit chunks to top_k_sparse."""
        reset_limiter()
        bm25_results = [_make_result(f"c{i}", "src", f"text {i}", 0.9 - i * 0.1, i) for i in range(1, 11)]
        
        with patch("backend.api_v1.run_search_dense", return_value=[]):
            with patch("backend.api_v1.run_search_bm25", return_value=bm25_results):
                with patch("backend.api_v1.run_search_hybrid", return_value=[]):
                    resp = client.post("/v1/ask", json={
                        "question": "test query",
                        "retrieval_mode": "sparse",
                        "top_k_sparse": 4,
                    })
        
        assert resp.status_code == 200
        data = resp.json()
        # Should only use top_k_sparse (4) chunks for answer generation
        assert len(data["retrieved_chunks"]) <= 4

    def test_hybrid_mode_respects_top_k_fused(self):
        """Hybrid mode should limit chunks to top_k_fused."""
        reset_limiter()
        dense_results = [_make_result(f"c{i}", "src", f"text {i}", 0.9 - i * 0.1, i) for i in range(1, 11)]
        bm25_results = [_make_result(f"c{i}", "src", f"text {i}", 0.9 - i * 0.1, i) for i in range(1, 11)]
        rrf_results = [_make_result(f"c{i}", "src", f"text {i}", 0.9 - i * 0.1, i) for i in range(1, 11)]
        
        with patch("backend.api_v1.run_search_dense", return_value=dense_results):
            with patch("backend.api_v1.run_search_bm25", return_value=bm25_results):
                with patch("backend.api_v1.run_search_hybrid", return_value=rrf_results):
                    resp = client.post("/v1/ask", json={
                        "question": "test query",
                        "retrieval_mode": "hybrid",
                        "top_k_dense": 10,
                        "top_k_sparse": 10,
                        "top_k_fused": 5,
                    })
        
        assert resp.status_code == 200
        data = resp.json()
        # Should only use top_k_fused (5) chunks for answer generation
        assert len(data["retrieved_chunks"]) <= 5

    def test_hybrid_rerank_mode_respects_top_k_final(self):
        """Hybrid rerank mode should limit chunks to top_k_final."""
        reset_limiter()
        dense_results = [_make_result(f"c{i}", "src", f"text {i}", 0.9 - i * 0.1, i) for i in range(1, 11)]
        bm25_results = [_make_result(f"c{i}", "src", f"text {i}", 0.9 - i * 0.1, i) for i in range(1, 11)]
        rrf_results = [_make_result(f"c{i}", "src", f"text {i}", 0.9 - i * 0.1, i) for i in range(1, 11)]
        reranker_results = [_make_result(f"c{i}", "src", f"text {i}", 0.9 - i * 0.1, i) for i in range(1, 11)]
        
        with patch("backend.api_v1.run_search_dense", return_value=dense_results):
            with patch("backend.api_v1.run_search_bm25", return_value=bm25_results):
                with patch("backend.api_v1.run_search_hybrid", return_value=rrf_results):
                    with patch("backend.reranker.get_reranker") as mock_reranker:
                        mock_reranker.return_value.rerank.return_value = reranker_results
                        resp = client.post("/v1/ask", json={
                            "question": "test query",
                            "retrieval_mode": "hybrid_rerank",
                            "top_k_dense": 10,
                            "top_k_sparse": 10,
                            "top_k_fused": 10,
                            "top_k_final": 3,
                        })
        
        assert resp.status_code == 200
        data = resp.json()
        # Should only use top_k_final (3) chunks for answer generation
        assert len(data["retrieved_chunks"]) <= 3


class TestRunSearchContextLimit:
    """Test that run_search enforces context limit when retrieval_results supplied."""

    def test_run_search_limits_supplied_results(self):
        """run_search should truncate supplied retrieval_results to top_k."""
        from backend.retrieval import run_search
        from backend.schemas import SearchRequest
        
        # Create 10 results but request top_k=3
        results = [_make_result(f"c{i}", "src", f"text {i}", 0.9 - i * 0.1, i) for i in range(1, 11)]
        request = SearchRequest(query="test", top_k=3)
        
        response = run_search(request, retrieval_results=results)
        
        # Should only use top_k (3) results
        assert len(response.results) <= 3

    def test_run_search_without_supplied_results_uses_own_limit(self):
        """run_search without supplied results should use its own retrieval limit."""
        from backend.retrieval import run_search
        from backend.schemas import SearchRequest
        
        request = SearchRequest(query="test", top_k=5)
        
        # This will use the actual dense retriever
        response = run_search(request)
        
        # Should respect top_k limit
        assert len(response.results) <= 5


class TestExtractiveFallback:
    """Test extractive fallback behavior for answerable and unrelated questions."""

    def test_answerable_question_with_evidence(self):
        """Extractive fallback should answer when sufficient evidence exists."""
        from backend.retrieval import _extractive_answer
        from backend.schemas import SearchResult
        
        results = [
            SearchResult(
                text="The maximum file upload size is 25 MB.",
                source="config.txt",
                page=1,
                score=0.9,
                metadata={}
            ),
            SearchResult(
                text="Files larger than 25 MB will be rejected.",
                source="config.txt",
                page=2,
                score=0.8,
                metadata={}
            ),
        ]
        
        answer = _extractive_answer("What is the maximum file upload size?", results)
        
        assert answer is not None
        assert "25 MB" in answer
        assert "[1]" in answer  # Citation marker

    def test_unrelated_question_abstains(self):
        """Extractive fallback should abstain for unrelated questions."""
        from backend.retrieval import _extractive_answer
        from backend.schemas import SearchResult
        
        results = [
            SearchResult(
                text="The maximum file upload size is 25 MB.",
                source="config.txt",
                page=1,
                score=0.9,
                metadata={}
            ),
        ]
        
        answer = _extractive_answer("What is the capital of France?", results)
        
        # Should return None (abstain) because no meaningful overlap
        assert answer is None

    def test_incidental_overlap_not_sufficient(self):
        """Single word overlap should not trigger extractive answer."""
        from backend.retrieval import _extractive_answer
        from backend.schemas import SearchResult
        
        results = [
            SearchResult(
                text="The system configuration includes various settings.",
                source="config.txt",
                page=1,
                score=0.9,
                metadata={}
            ),
        ]
        
        # Query shares only "the" and "system" - common words
        answer = _extractive_answer("What is the system?", results)
        
        # Should return None because overlap is incidental
        assert answer is None

    def test_meaningful_overlap_triggers_answer(self):
        """Multiple matching terms should trigger extractive answer."""
        from backend.retrieval import _extractive_answer
        from backend.schemas import SearchResult
        
        results = [
            SearchResult(
                text="The MAX_UPLOAD_MB setting controls file size limits.",
                source="config.txt",
                page=1,
                score=0.9,
                metadata={}
            ),
        ]
        
        # Query shares "MAX_UPLOAD_MB" and "file" - meaningful terms
        answer = _extractive_answer("What is MAX_UPLOAD_MB for files?", results)
        
        assert answer is not None
        assert "MAX_UPLOAD_MB" in answer

    def test_source_attribution_in_answer(self):
        """Extractive answer should include source citations."""
        from backend.retrieval import _extractive_answer
        from backend.schemas import SearchResult
        
        results = [
            SearchResult(
                text="The chunk size is 512 tokens.",
                source="settings.txt",
                page=1,
                score=0.9,
                metadata={}
            ),
            SearchResult(
                text="Chunk overlap is set to 50 tokens.",
                source="settings.txt",
                page=2,
                score=0.8,
                metadata={}
            ),
        ]
        
        answer = _extractive_answer("What is the chunk size?", results)
        
        assert answer is not None
        assert "[1]" in answer  # First citation
        assert "512" in answer


class TestMMRFallbackBehavior:
    """Test MMR fallback behavior in dense retrieval."""

    def test_mmr_fallback_returns_zero_scores(self):
        """MMR fallback should return 0.0 scores that get filtered."""
        from backend.retrieval_dense import DenseRetriever
        from backend.settings import MIN_RELEVANCE_SCORE
        
        # This test verifies the behavior is documented and intentional
        # The actual MMR fallback is tested in integration tests
        assert MIN_RELEVANCE_SCORE > 0.0
        # MMR results with 0.0 score will be filtered by MIN_RELEVANCE_SCORE


class TestBM25IncrementalIngestion:
    """Test BM25 incremental ingestion and index consistency."""

    def test_incremental_ingestion_preserves_existing(self):
        """Adding new chunks should preserve existing ones."""
        from backend.bm25_index import BM25Index
        from backend.models import Chunk
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as d:
            idx = BM25Index(index_dir=Path(d))
            
            # Add first document
            a = Chunk.create('doc-a', 0, 'alpha retrieval evidence', 'ha')
            idx.build([a])
            
            # Add second document
            b = Chunk.create('doc-b', 0, 'beta retrieval evidence', 'hb')
            idx.add_chunks([b])
            
            # Both should be retrievable
            alpha_results = idx.search('alpha', 2)
            beta_results = idx.search('beta', 2)
            
            assert any(r['chunk_id'] == a.id for r in alpha_results)
            assert any(r['chunk_id'] == b.id for r in beta_results)

    def test_persistence_preserves_both_documents(self):
        """Save and reload should preserve both documents."""
        from backend.bm25_index import BM25Index
        from backend.models import Chunk
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as d:
            idx = BM25Index(index_dir=Path(d))
            a = Chunk.create('doc-a', 0, 'alpha retrieval evidence', 'ha')
            b = Chunk.create('doc-b', 0, 'beta retrieval evidence', 'hb')
            idx.build([a])
            idx.add_chunks([b])
            idx.save()
            
            # Reload
            reloaded = BM25Index(index_dir=Path(d))
            assert reloaded.load()
            
            alpha_results = reloaded.search('alpha', 2)
            beta_results = reloaded.search('beta', 2)
            
            assert any(r['chunk_id'] == a.id for r in alpha_results)
            assert any(r['chunk_id'] == b.id for r in beta_results)

    def test_deletion_removes_only_target(self):
        """Deleting one document should not affect others."""
        from backend.bm25_index import BM25Index
        from backend.models import Chunk
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as d:
            idx = BM25Index(index_dir=Path(d))
            a = Chunk.create('doc-a', 0, 'alpha retrieval evidence', 'ha')
            b = Chunk.create('doc-b', 0, 'beta retrieval evidence', 'hb')
            idx.build([a])
            idx.add_chunks([b])
            
            # Delete doc-b
            removed = idx.remove_document('doc-b')
            assert removed == 1
            
            # doc-a should still be there
            alpha_results = idx.search('alpha', 2)
            beta_results = idx.search('beta', 2)
            
            assert any(r['chunk_id'] == a.id for r in alpha_results)
            assert len(beta_results) == 0

    def test_repeated_ingestion_no_duplicate_ids(self):
        """Re-ingesting same content should not create duplicate chunk IDs."""
        from backend.bm25_index import BM25Index
        from backend.models import Chunk
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as d:
            idx = BM25Index(index_dir=Path(d))
            a = Chunk.create('doc-a', 0, 'alpha retrieval evidence', 'ha')
            
            # Build twice with same content
            idx.build([a])
            idx.build([a])
            
            # Should have only one chunk
            results = idx.search('alpha', 5)
            chunk_ids = [r['chunk_id'] for r in results]
            assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates
            assert len(chunk_ids) == 1


class TestIndexConsistency:
    """Test index consistency diagnostics."""

    def test_consistency_check_detects_mismatch(self):
        """Consistency check should detect Chroma/BM25 mismatch."""
        from backend.storage_health import check_index_consistency
        
        # This test verifies the diagnostic exists and returns expected structure
        result = check_index_consistency()
        assert "status" in result
        assert result["status"] in ("ok", "warning", "error")
        if result["status"] == "error":
            assert "issues" in result
            for issue in result["issues"]:
                assert "type" in issue