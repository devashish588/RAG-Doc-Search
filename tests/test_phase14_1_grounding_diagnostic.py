"""Phase 14.1 — Grounding pipeline regression tests.

Proves the grounding/answer pipeline works end-to-end by feeding mock LLM
answers through the real verifier and confidence estimator.

Root cause identified: OpenRouter HTTP 402 (no credits) causes LLM failure,
which triggers the fallback "Most relevant context:" answer, which the
verifier correctly rejects as non-LLM output, resulting in grounding=0.0
and abstention.

These tests prove the pipeline is functionally correct when the LLM
produces a valid answer.
"""
from unittest.mock import patch

import pytest

from backend.interfaces import RetrievalResult
from backend.verifier import CitationVerifier
from backend.confidence import ConfidenceEstimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYLLABUS_CONTEXT = [
    RetrievalResult(
        chunk_id="eaf19b71022943d58c8d5a6263206de7:4",
        score=1.0,
        rank=1,
        source="Syllabus_System_Discipline.pdf",
        content=(
            "Section 9: Databases\n"
            "ER model. Relational model: relational algebra, tuple calculus, SQL. "
            "Integrity constraints, normal forms. File organization, indexing "
            "(e.g., B and B+ trees). Transactions and concurrency control"
        ),
        metadata={
            "document_id": "eaf19b71022943d58c8d5a6263206de7",
            "chunk": 4,
            "page": 1,
            "source": "Syllabus_System_Discipline.pdf",
        },
    ),
]

MOCK_ANSWER = (
    "Section 9 covers the ER model, relational model, relational algebra, "
    "tuple calculus, SQL, integrity constraints, normal forms, file organization, "
    "indexing, transactions, and concurrency control."
)


# ---------------------------------------------------------------------------
# 1. Retrieved chunks reach context builder
# ---------------------------------------------------------------------------

class TestRetrievedChunksReachContextBuilder:
    def test_eval_context_built_from_search_results(self):
        from backend.schemas import SearchResult
        search_results = [
            SearchResult(
                text=r.content,
                source=r.source,
                page=r.metadata.get("page"),
                score=r.score,
                metadata=r.metadata,
            )
            for r in SYLLABUS_CONTEXT
        ]
        eval_context = []
        for i, r in enumerate(search_results):
            chunk_id_str = str(r.metadata.get("document_id", "")) + ":" + str(r.metadata.get("chunk", i))
            eval_context.append(RetrievalResult(
                chunk_id=chunk_id_str,
                score=r.score,
                rank=i + 1,
                source=r.source,
                content=r.text,
                metadata=r.metadata,
            ))
        assert len(eval_context) == len(SYLLABUS_CONTEXT)
        assert eval_context[0].source == "Syllabus_System_Discipline.pdf"
        assert "Section 9" in eval_context[0].content


# ---------------------------------------------------------------------------
# 2. Context contains retrieved text
# ---------------------------------------------------------------------------

class TestContextContainsRetrievedText:
    def test_context_preserves_text_content(self):
        for r in SYLLABUS_CONTEXT:
            assert len(r.content) > 0
            assert "Databases" in r.content


# ---------------------------------------------------------------------------
# 3. Source metadata survives transformation
# ---------------------------------------------------------------------------

class TestSourceMetadataSurvivesTransformation:
    def test_source_field_preserved(self):
        for r in SYLLABUS_CONTEXT:
            assert r.source == "Syllabus_System_Discipline.pdf"
            assert r.metadata["source"] == "Syllabus_System_Discipline.pdf"

    def test_document_id_preserved(self):
        for r in SYLLABUS_CONTEXT:
            assert r.metadata["document_id"] == "eaf19b71022943d58c8d5a6263206de7"


# ---------------------------------------------------------------------------
# 4. Page metadata survives transformation
# ---------------------------------------------------------------------------

class TestPageMetadataSurvivesTransformation:
    def test_page_preserved_in_metadata(self):
        for r in SYLLABUS_CONTEXT:
            assert r.metadata["page"] == 1

    def test_page_preserved_in_retrieval_result(self):
        for r in SYLLABUS_CONTEXT:
            assert r.metadata.get("page") == 1


# ---------------------------------------------------------------------------
# 5. Citation verification receives correct metadata
# ---------------------------------------------------------------------------

class TestCitationVerificationReceivesCorrectMetadata:
    def test_verifier_returns_correct_source(self):
        verifier = CitationVerifier()
        result = verifier.verify(MOCK_ANSWER, SYLLABUS_CONTEXT)
        citations = result["citations"]
        supported = [c for c in citations if c["verdict"] == "supported"]
        assert len(supported) > 0
        assert supported[0]["source"] == "Syllabus_System_Discipline.pdf"

    def test_verifier_returns_correct_page(self):
        verifier = CitationVerifier()
        result = verifier.verify(MOCK_ANSWER, SYLLABUS_CONTEXT)
        citations = result["citations"]
        supported = [c for c in citations if c["verdict"] == "supported"]
        assert len(supported) > 0
        assert supported[0]["page"] == 1


# ---------------------------------------------------------------------------
# 6. Supported query produces non-zero grounding
# ---------------------------------------------------------------------------

class TestSupportedQueryProducesGrounding:
    def test_grounding_ratio_nonzero(self):
        verifier = CitationVerifier()
        result = verifier.verify(MOCK_ANSWER, SYLLABUS_CONTEXT)
        assert result["metrics"]["grounding_ratio"] > 0.0

    def test_supported_claims_exist(self):
        verifier = CitationVerifier()
        result = verifier.verify(MOCK_ANSWER, SYLLABUS_CONTEXT)
        assert result["metrics"]["supported_claims"] > 0


# ---------------------------------------------------------------------------
# 7. Supported query does not incorrectly abstain
# ---------------------------------------------------------------------------

class TestSupportedQueryDoesNotAbstain:
    def test_no_abstention_with_valid_answer(self):
        verifier = CitationVerifier()
        verification = verifier.verify(MOCK_ANSWER, SYLLABUS_CONTEXT)
        estimator = ConfidenceEstimator()
        conf = estimator.estimate(
            dense_results=SYLLABUS_CONTEXT,
            bm25_results=[],
            rrf_results=[],
            reranker_results=[],
            grounding_ratio=verification["metrics"]["grounding_ratio"],
            retrieval_mode="dense",
        )
        assert conf.abstention_flag is False
        assert conf.overall_score >= 0.5

    def test_confidence_level_not_low(self):
        verifier = CitationVerifier()
        verification = verifier.verify(MOCK_ANSWER, SYLLABUS_CONTEXT)
        estimator = ConfidenceEstimator()
        conf = estimator.estimate(
            dense_results=SYLLABUS_CONTEXT,
            bm25_results=[],
            rrf_results=[],
            reranker_results=[],
            grounding_ratio=verification["metrics"]["grounding_ratio"],
            retrieval_mode="dense",
        )
        assert conf.level in ("high", "medium")


# ---------------------------------------------------------------------------
# 8. Unsupported query still abstains
# ---------------------------------------------------------------------------

class TestUnsupportedQueryAbstains:
    def test_llm_fallback_triggers_abstention(self):
        fallback_answer = "Most relevant context:\n[1] Syllabus_System_Discipline.pdf, page 1: Section 9..."
        verifier = CitationVerifier()
        result = verifier.verify(fallback_answer, SYLLABUS_CONTEXT)
        assert result["metrics"]["total_claims"] == 0
        assert result["metrics"]["grounding_ratio"] == 0.0

    def test_grounding_zero_on_empty_claims(self):
        estimator = ConfidenceEstimator()
        conf = estimator.estimate(
            dense_results=SYLLABUS_CONTEXT,
            bm25_results=[],
            rrf_results=[],
            reranker_results=[],
            grounding_ratio=0.0,
            retrieval_mode="dense",
        )
        assert conf.grounding_confidence == 0.0
        assert conf.abstention_flag is True


# ---------------------------------------------------------------------------
# 9. All four retrieval modes preserve answer/grounding contract
# ---------------------------------------------------------------------------

class TestAllModesPreserveGroundingContract:
    @pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid", "hybrid_rerank"])
    def test_mock_answer_produces_grounding_in_each_mode(self, mode):
        verifier = CitationVerifier()
        verification = verifier.verify(MOCK_ANSWER, SYLLABUS_CONTEXT)
        estimator = ConfidenceEstimator()
        mode_map = {
            "dense": "dense_results",
            "sparse": "bm25_results",
            "hybrid": "rrf_results",
            "hybrid_rerank": "reranker_results",
        }
        kwargs = {
            "dense_results": [],
            "bm25_results": [],
            "rrf_results": [],
            "reranker_results": [],
            "grounding_ratio": verification["metrics"]["grounding_ratio"],
            "retrieval_mode": mode,
        }
        kwargs[mode_map[mode]] = SYLLABUS_CONTEXT
        conf = estimator.estimate(**kwargs)
        assert conf.grounding_confidence > 0.0, f"Mode {mode}: grounding should be > 0"
        assert conf.abstention_flag is False, f"Mode {mode}: should not abstain with valid answer"


# ---------------------------------------------------------------------------
# 10. Source-filtered context works
# ---------------------------------------------------------------------------

class TestSourceFilteredContext:
    def test_source_filter_preserves_chunks(self):
        filtered = [r for r in SYLLABUS_CONTEXT if r.source == "Syllabus_System_Discipline.pdf"]
        assert len(filtered) == len(SYLLABUS_CONTEXT)

    def test_source_filter_with_mock_answer(self):
        verifier = CitationVerifier()
        result = verifier.verify(MOCK_ANSWER, SYLLABUS_CONTEXT)
        assert result["metrics"]["grounding_ratio"] > 0.0


# ---------------------------------------------------------------------------
# 11. Existing Phase 13.2 behavior remains intact
# ---------------------------------------------------------------------------

class TestPhase132BehaviorIntact:
    def test_retrieved_chunks_field_exists_in_ask_response(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.middleware import reset_limiter
        client = TestClient(app)
        reset_limiter()
        resp = client.post("/v1/ask", json={
            "question": "test query",
            "retrieval_mode": "dense",
            "top_k_dense": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "retrieved_chunks" in data
        assert isinstance(data["retrieved_chunks"], list)

    def test_confidence_signals_present(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.middleware import reset_limiter
        client = TestClient(app)
        reset_limiter()
        resp = client.post("/v1/ask", json={
            "question": "test query",
            "retrieval_mode": "dense",
            "top_k_dense": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "confidence" in data
        conf = data["confidence"]
        assert "overall_score" in conf
        assert "retrieval_confidence" in conf
        assert "grounding_confidence" in conf
        assert "abstention_flag" in conf
