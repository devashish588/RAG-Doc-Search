"""Phase 6 tests: Cross-Encoder reranking."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.interfaces import RetrievalResult
from backend.main import app
from backend.reranker import CrossEncoderReranker, get_reranker, reset_reranker
from backend.schemas import SearchResponse

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeRanker:
    """Deterministic fake cross-encoder: scores by passage index (reverses order
    of equal-ordered input so we can verify reranking actually reorders)."""

    def __init__(self, score_map=None):
        # score_map: chunk_id -> score; if None, score = position-based
        self.score_map = score_map or {}

    def rerank(self, request):
        passages = request.passages
        out = []
        for i, p in enumerate(passages):
            if p["id"] in self.score_map:
                score = self.score_map[p["id"]]
            else:
                score = float(len(passages) - i)
            out.append({"id": p["id"], "score": score})
        return out


def _cand(chunk_id, rrf_rank, rrf_score=0.1, in_dense=True, in_bm25=False):
    return RetrievalResult(
        chunk_id=chunk_id,
        score=rrf_score,
        rank=rrf_rank,
        source="s.txt",
        content=f"content-{chunk_id}",
        metadata={
            "dense_rank": rrf_rank if in_dense else None,
            "sparse_rank": rrf_rank if in_bm25 else None,
            "in_dense": in_dense,
            "in_bm25": in_bm25,
            "rrf_score": rrf_score,
        },
    )


# ---------------------------------------------------------------------------
# MODEL / LOADING
# ---------------------------------------------------------------------------

def test_reranker_injection_ready():
    rr = CrossEncoderReranker(_ranker=FakeRanker(), top_k=2)
    assert rr.available is True
    assert rr.status == "ready"


def test_reranker_lazy_singleton_reuse():
    reset_reranker()
    a = get_reranker()
    b = get_reranker()
    assert a is b
    reset_reranker()


def test_reranker_init_failure_fallback(monkeypatch):
    """If the cross-encoder model fails to load, reranking must not crash;
    it must fall back to the RRF pool with reranker_status='failed'."""
    import flashrank

    class BoomRanker:
        def __init__(self, *a, **k):
            raise RuntimeError("model load failed")

    monkeypatch.setattr(flashrank, "Ranker", BoomRanker)
    rr = CrossEncoderReranker()
    cands = [_cand("c1", 1), _cand("c2", 2)]
    out, info = rr.rerank_with_timing("q", cands, top_k=2)
    assert info["status"] == "fallback"
    assert len(out) == 2
    assert out[0].metadata["reranker_status"] == "failed"
    assert out[0].metadata["original_rrf_rank"] == 1


# ---------------------------------------------------------------------------
# RERANKING
# ---------------------------------------------------------------------------

def test_reranker_ranking_order():
    # Fake scores: c3 highest, c1 lowest -> reversed from RRF order
    sm = {"c1": 0.1, "c2": 0.5, "c3": 0.9}
    rr = CrossEncoderReranker(_ranker=FakeRanker(sm), top_k=3)
    cands = [_cand("c1", 1), _cand("c2", 2), _cand("c3", 3)]
    out, info = rr.rerank_with_timing("q", cands)
    assert [r.chunk_id for r in out] == ["c3", "c2", "c1"]
    assert info["status"] == "ok"


def test_reranker_top_k_limits_output():
    rr = CrossEncoderReranker(_ranker=FakeRanker(), top_k=2)
    cands = [_cand(f"c{i}", i + 1) for i in range(5)]
    out = rr.rerank("q", cands)
    assert len(out) == 2


def test_reranker_deterministic():
    rr = CrossEncoderReranker(_ranker=FakeRanker({"c1": 0.1, "c2": 0.9}), top_k=2)
    cands = [_cand("c1", 1), _cand("c2", 2)]
    o1 = rr.rerank("q", cands)
    o2 = rr.rerank("q", cands)
    assert [r.chunk_id for r in o1] == [r.chunk_id for r in o2]


def test_reranker_scoring_failure_fallback():
    class BoomRanker:
        def rerank(self, request):
            raise RuntimeError("scoring exploded")

    rr = CrossEncoderReranker(_ranker=BoomRanker(), top_k=2)
    cands = [_cand("c1", 1), _cand("c2", 2)]
    out, info = rr.rerank_with_timing("q", cands)
    assert info["status"] == "fallback"
    assert all(r.metadata["reranker_status"] == "failed" for r in out)
    # RRF candidates returned (not fabricated)
    assert {r.chunk_id for r in out} == {"c1", "c2"}


# ---------------------------------------------------------------------------
# PROVENANCE
# ---------------------------------------------------------------------------

def test_reranker_preserves_rrf_provenance():
    rr = CrossEncoderReranker(_ranker=FakeRanker({"c1": 0.9, "c2": 0.1}), top_k=2)
    c1 = _cand("c1", rrf_rank=1, rrf_score=0.22, in_dense=True, in_bm25=False)
    c2 = _cand("c2", rrf_rank=2, rrf_score=0.11, in_dense=False, in_bm25=True)
    out, info = rr.rerank_with_timing("q", [c1, c2])
    top = out[0]
    assert top.chunk_id == "c1"
    assert top.metadata["original_rrf_rank"] == 1
    assert top.metadata["original_rrf_score"] == 0.22
    assert top.metadata["reranker_score"] == 0.9
    assert top.metadata["final_rank"] == 1
    assert top.metadata["in_dense"] is True
    assert top.metadata["in_bm25"] is False
    assert "rrf_score" in top.metadata


# ---------------------------------------------------------------------------
# EMPTY / SHORT INPUT
# ---------------------------------------------------------------------------

def test_reranker_empty_candidates():
    rr = CrossEncoderReranker(_ranker=FakeRanker())
    out, info = rr.rerank_with_timing("q", [])
    assert out == []
    assert info["status"] == "ok"


def test_reranker_fewer_than_topk():
    rr = CrossEncoderReranker(_ranker=FakeRanker(), top_k=10)
    cands = [_cand("c1", 1)]
    out = rr.rerank("q", cands, top_k=10)
    assert len(out) == 1


def test_reranker_candidate_cap():
    rr = CrossEncoderReranker(_ranker=FakeRanker(), candidate_k=3, top_k=2)
    cands = [_cand(f"c{i}", i + 1) for i in range(10)]
    out = rr.rerank("q", cands, top_k=2)
    # capped at candidate_k=3 before reranking -> top_k=2 returned, all from first 3
    assert len(out) == 2
    assert {r.chunk_id for r in out} <= {f"c{i}" for i in range(3)}


# ---------------------------------------------------------------------------
# CONFIG VALIDATION
# ---------------------------------------------------------------------------

def test_reranker_invalid_topk_raises():
    with pytest.raises(ValueError):
        CrossEncoderReranker(candidate_k=5, top_k=10)


# ---------------------------------------------------------------------------
# API MODES
# ---------------------------------------------------------------------------

def _mock_rr(chunk_id):
    return RetrievalResult(
        chunk_id=chunk_id, score=0.9, rank=1, source="s.txt",
        content="rc", metadata={"reranker_score": 0.9, "original_rrf_rank": 1,
                                "final_rank": 1, "reranker_status": "ok"},
    )


def test_api_default_is_dense():
    with patch("backend.api_v1.run_search_dense") as m_d, \
         patch("backend.api_v1.run_search_bm25") as m_b, \
         patch("backend.api_v1.run_search_hybrid") as m_h, \
         patch("backend.api_v1.run_search") as m_run, \
         patch("backend.api_v1.get_reranker") as m_re:
        m_d.return_value = [_cand("d1", 1)]
        m_b.return_value = [_cand("b1", 1)]
        m_h.return_value = [_cand("h1", 1)]
        m_re.return_value = CrossEncoderReranker(_ranker=FakeRanker())
        m_run.return_value = SearchResponse(query="q", answer="a", results=[], latency_ms=1.0)
        r = client.post("/v1/ask", json={"question": "q"})
        assert r.status_code == 200
        t = r.json()["retrieval_trace"]
        assert len(t["dense"]) == 1
        assert t["bm25"] == []
        assert t["rrf"] == []
        assert t["reranker"] == []
        m_h.assert_not_called()
        m_re.assert_not_called()


def test_api_hybrid_mode():
    with patch("backend.api_v1.run_search_dense") as m_d, \
         patch("backend.api_v1.run_search_bm25") as m_b, \
         patch("backend.api_v1.run_search_hybrid") as m_h, \
         patch("backend.api_v1.run_search") as m_run, \
         patch("backend.api_v1.get_reranker") as m_re:
        m_d.return_value = [_cand("d1", 1)]
        m_b.return_value = [_cand("b1", 1)]
        m_h.return_value = [_cand("h1", 1)]
        m_re.return_value = CrossEncoderReranker(_ranker=FakeRanker())
        m_run.return_value = SearchResponse(query="q", answer="a", results=[], latency_ms=1.0)
        r = client.post("/v1/ask", json={"question": "q", "retrieval_mode": "hybrid"})
        assert r.status_code == 200
        t = r.json()["retrieval_trace"]
        assert len(t["dense"]) == 1 and len(t["bm25"]) == 1 and len(t["rrf"]) == 1
        assert t["reranker"] == []
        m_re.assert_not_called()


def test_api_sparse_mode():
    with patch("backend.api_v1.run_search_dense") as m_d, \
         patch("backend.api_v1.run_search_bm25") as m_b, \
         patch("backend.api_v1.run_search_hybrid") as m_h, \
         patch("backend.api_v1.run_search") as m_run, \
         patch("backend.api_v1.get_reranker") as m_re:
        m_d.return_value = [_cand("d1", 1)]
        m_b.return_value = [_cand("b1", 1)]
        m_h.return_value = [_cand("h1", 1)]
        m_re.return_value = CrossEncoderReranker(_ranker=FakeRanker())
        m_run.return_value = SearchResponse(query="q", answer="a", results=[], latency_ms=1.0)
        r = client.post("/v1/ask", json={"question": "q", "retrieval_mode": "sparse"})
        assert r.status_code == 200
        t = r.json()["retrieval_trace"]
        assert t["dense"] == [] and len(t["bm25"]) == 1 and t["rrf"] == []
        assert t["reranker"] == []
        m_re.assert_not_called()


def test_api_hybrid_rerank_mode():
    fake_reranker = CrossEncoderReranker(_ranker=FakeRanker({"h1": 0.9}))
    with patch("backend.api_v1.run_search_dense") as m_d, \
         patch("backend.api_v1.run_search_bm25") as m_b, \
         patch("backend.api_v1.run_search_hybrid") as m_h, \
         patch("backend.api_v1.run_search") as m_run, \
         patch("backend.api_v1.get_reranker") as m_re:
        m_d.return_value = [_cand("d1", 1)]
        m_b.return_value = [_cand("b1", 1)]
        m_h.return_value = [_cand("h1", 1)]
        m_re.return_value = fake_reranker
        m_run.return_value = SearchResponse(query="q", answer="a", results=[], latency_ms=1.0)
        r = client.post("/v1/ask", json={"question": "q", "retrieval_mode": "hybrid_rerank", "top_k_final": 3})
        assert r.status_code == 200
        t = r.json()["retrieval_trace"]
        assert len(t["dense"]) == 1
        assert len(t["bm25"]) == 1
        assert len(t["rrf"]) == 1
        assert len(t["reranker"]) == 1
        assert t["reranker"][0]["metadata"]["reranker_status"] == "ok"


def test_api_hybrid_rerank_fallback_trace():
    """If the reranker fails, the reranker trace still appears (marked failed)."""

    class BoomRanker:
        def rerank(self, request):
            raise RuntimeError("boom")

    fake_reranker = CrossEncoderReranker(_ranker=BoomRanker())
    with patch("backend.api_v1.run_search_dense") as m_d, \
         patch("backend.api_v1.run_search_bm25") as m_b, \
         patch("backend.api_v1.run_search_hybrid") as m_h, \
         patch("backend.api_v1.run_search") as m_run, \
         patch("backend.api_v1.get_reranker") as m_re:
        m_d.return_value = [_cand("d1", 1)]
        m_b.return_value = [_cand("b1", 1)]
        m_h.return_value = [_cand("h1", 1)]
        m_re.return_value = fake_reranker
        m_run.return_value = SearchResponse(query="q", answer="a", results=[], latency_ms=1.0)
        r = client.post("/v1/ask", json={"question": "q", "retrieval_mode": "hybrid_rerank"})
        assert r.status_code == 200
        t = r.json()["retrieval_trace"]
        assert len(t["reranker"]) == 1
        assert t["reranker"][0]["metadata"]["reranker_status"] == "failed"
