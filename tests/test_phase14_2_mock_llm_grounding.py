"""Phase 14.2 — Deterministic mock LLM test.

Proves the grounding pipeline works end-to-end by feeding a mock LLM answer
through the real verifier and confidence estimator.

This test proves whether the failure is:
A. Groq API/provider failure → mock bypasses API, answer flows through
B. LLM response parsing failure → mock provides parsed content
C. prompt construction failure → mock bypasses prompt
D. citation generation failure → verifier generates citations from mock answer
E. citation verification failure → verifier verifies mock answer against context
F. confidence calculation failure → estimator computes confidence from verification
G. abstention logic failure → abstention only triggers on low grounding
"""
import json
from unittest.mock import patch, MagicMock

import pytest

from backend.interfaces import RetrievalResult
from backend.verifier import CitationVerifier
from backend.confidence import ConfidenceEstimator
from backend.schemas import SearchResult


# ---------------------------------------------------------------------------
# Fixture: OS syllabus chunks (production-realistic)
# ---------------------------------------------------------------------------

SYLLABUS_CHUNKS = [
    RetrievalResult(
        chunk_id="eaf19b71022943d58c8d5a6263206de7:4",
        score=1.0,
        rank=1,
        source="Syllabus_System_Discipline.pdf",
        content=(
            "Section 8: Operating System\n"
            "System calls, processes, threads, inter-process communication, "
            "concurrency and synchronization. Deadlock. CPU and I/O scheduling. "
            "Memory management and virtual memory. File systems.\n"
            "Section 9: Databases\n"
            "ER model. Relational model: relational algebra, tuple calculus, SQL. "
            "Integrity constraints, normal forms. File organization, indexing "
            "(e.g., B and B+ trees). Transactions and concurrency control\n"
            "Section 10: Computer Networks\n"
            "Concept of layering: OSI and TCP/IP Protocol Stacks."
        ),
        metadata={
            "document_id": "eaf19b71022943d58c8d5a6263206de7",
            "chunk": 4,
            "page": 1,
            "source": "Syllabus_System_Discipline.pdf",
        },
    ),
]

MOCK_LLM_ANSWER = (
    "The syllabus covers three main sections. Section 8: Operating System "
    "includes system calls, processes, threads, inter-process communication, "
    "concurrency, deadlock, CPU scheduling, memory management, and file systems. "
    "Section 9: Databases covers the ER model, relational model, relational "
    "algebra, tuple calculus, SQL, integrity constraints, normal forms, file "
    "organization, indexing, transactions, and concurrency control. "
    "Section 10: Computer Networks covers OSI and TCP/IP protocol stacks."
)


# ---------------------------------------------------------------------------
# 1. Mock LLM answer produces grounding > 0
# ---------------------------------------------------------------------------

class TestMockLLMAnswerGrounds:
    def test_verifier_produces_supported_claims(self):
        verifier = CitationVerifier()
        result = verifier.verify(MOCK_LLM_ANSWER, SYLLABUS_CHUNKS)
        assert result["metrics"]["total_claims"] > 0
        assert result["metrics"]["supported_claims"] > 0
        assert result["metrics"]["grounding_ratio"] > 0.0

    def test_verifier_produces_citations(self):
        verifier = CitationVerifier()
        result = verifier.verify(MOCK_LLM_ANSWER, SYLLABUS_CHUNKS)
        supported = [c for c in result["citations"] if c["verdict"] == "supported"]
        assert len(supported) > 0
        assert supported[0]["source"] == "Syllabus_System_Discipline.pdf"


# ---------------------------------------------------------------------------
# 2. Mock LLM answer does not abstain
# ---------------------------------------------------------------------------

class TestMockLLMAnswerNoAbstain:
    def test_confidence_above_threshold(self):
        verifier = CitationVerifier()
        verification = verifier.verify(MOCK_LLM_ANSWER, SYLLABUS_CHUNKS)
        estimator = ConfidenceEstimator()
        conf = estimator.estimate(
            dense_results=SYLLABUS_CHUNKS,
            bm25_results=[],
            rrf_results=[],
            reranker_results=[],
            grounding_ratio=verification["metrics"]["grounding_ratio"],
            retrieval_mode="dense",
        )
        assert conf.abstention_flag is False
        assert conf.overall_score >= 0.5
        assert conf.level in ("high", "medium")


# ---------------------------------------------------------------------------
# 3. Full integration: mock LLM → verifier → confidence → response
# ---------------------------------------------------------------------------

class TestFullMockPipeline:
    def test_end_to_end_with_mock_llm(self):
        """Simulate the complete /v1/ask flow with a mocked LLM."""
        # Step 1: Retrieval (already done, we have chunks)
        search_results = [
            SearchResult(
                text=r.content,
                source=r.source,
                page=r.metadata.get("page"),
                score=r.score,
                metadata=r.metadata,
            )
            for r in SYLLABUS_CHUNKS
        ]

        # Step 2: LLM generates answer (mocked)
        llm_answer = MOCK_LLM_ANSWER
        assert llm_answer is not None
        assert len(llm_answer) > 0

        # Step 3: Build eval_context (same as api_v1.py)
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

        # Step 4: Verify claims
        verifier = CitationVerifier()
        verification = verifier.verify(llm_answer, eval_context)
        assert verification["metrics"]["grounding_ratio"] > 0.0

        # Step 5: Estimate confidence
        estimator = ConfidenceEstimator()
        conf = estimator.estimate(
            dense_results=SYLLABUS_CHUNKS,
            bm25_results=[],
            rrf_results=[],
            reranker_results=[],
            grounding_ratio=verification["metrics"]["grounding_ratio"],
            retrieval_mode="dense",
        )

        # Step 6: Response contract
        if not search_results or conf.abstention_flag:
            status = "insufficient_context"
        else:
            status = "answered"

        assert status == "answered"
        assert conf.abstention_flag is False
        assert conf.grounding_confidence > 0.0


# ---------------------------------------------------------------------------
# 4. Fallback text still abstains (proves verifier detection works)
# ---------------------------------------------------------------------------

class TestFallbackTextAbstains:
    def test_most_relevant_context_abstains(self):
        fallback = "Most relevant context:\n[1] Syllabus_System_Discipline.pdf, page 1: Section 9..."
        verifier = CitationVerifier()
        result = verifier.verify(fallback, SYLLABUS_CHUNKS)
        assert result["metrics"]["total_claims"] == 0
        assert result["metrics"]["grounding_ratio"] == 0.0

    def test_llm_failure_produces_none_abstains(self):
        estimator = ConfidenceEstimator()
        conf = estimator.estimate(
            dense_results=SYLLABUS_CHUNKS,
            bm25_results=[],
            rrf_results=[],
            reranker_results=[],
            grounding_ratio=0.0,
            retrieval_mode="dense",
        )
        assert conf.abstention_flag is True


# ---------------------------------------------------------------------------
# 5. Mock generate_answer integration test
# ---------------------------------------------------------------------------

class TestMockGenerateAnswerIntegration:
    def test_generate_answer_returns_content_when_api_succeeds(self):
        from backend.circuit_breaker import reset_circuit_breaker
        reset_circuit_breaker()

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": MOCK_LLM_ANSWER}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_breaker = MagicMock()
        mock_breaker.allow_request.return_value = True

        with patch("backend.llm.request.urlopen", return_value=mock_response), \
             patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
            import importlib
            import backend.llm as llm
            importlib.reload(llm)
            answer = llm.generate_answer("What is section 9?", [
                SearchResult(text="Section 9 covers databases.", source="test.pdf",
                             page=1, score=0.9, metadata={"document_id": "d1", "chunk": 0})
            ])
            assert answer is not None
            assert "Section 9" in answer

    def test_generate_answer_returns_none_on_http_error(self):
        from urllib import error as urlerror
        mock_breaker = MagicMock()
        mock_breaker.allow_request.return_value = True

        with patch("backend.llm.request.urlopen",
                    side_effect=urlerror.HTTPError(
                        "https://api.groq.com/openai/v1/chat/completions",
                        403, "Forbidden", {}, None
                    )), \
             patch("backend.llm.get_circuit_breaker", return_value=mock_breaker), \
             patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)
            # Patch the reloaded module's breaker reference
            with patch.object(llm, "get_circuit_breaker", return_value=mock_breaker):
                answer = llm.generate_answer("test", [
                    SearchResult(text="test", source="test.pdf", page=1, score=0.9, metadata={})
                ])
                assert answer is None
                mock_breaker.record_failure.assert_called_once()
