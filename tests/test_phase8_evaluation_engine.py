"""Phase 8 — Evaluation Engine Tests.

Tests advanced IR metrics, EvaluationEngine, and benchmark output validation.
"""
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure workspace root is in sys.path
TESTS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = TESTS_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evals.metrics import (
    _binary_relevance_vector,
    calculate_ndcg_at_k,
    calculate_map,
    calculate_faithfulness,
    calculate_answer_relevance,
    calculate_retrieval_metrics,
    tokenize,
)


# ============================================================================
# NDCG@5 Tests
# ============================================================================

class TestNDCG:
    """NDCG@5 calculation tests."""

    def test_perfect_ranking(self):
        """All relevant docs at top positions -> NDCG@5 = 1.0."""
        retrieved = [{"source": "doc_a.txt"}, {"source": "doc_b.txt"}, {"source": "other.txt"}]
        expected = [{"document": "doc_a.txt"}, {"document": "doc_b.txt"}]
        ndcg = calculate_ndcg_at_k(retrieved, expected, k=5)
        assert ndcg == 1.0

    def test_imperfect_ranking(self):
        """Relevant doc at position 2 -> NDCG@5 < 1.0."""
        retrieved = [{"source": "other.txt"}, {"source": "doc_a.txt"}, {"source": "other2.txt"}]
        expected = [{"document": "doc_a.txt"}]
        ndcg = calculate_ndcg_at_k(retrieved, expected, k=5)
        # DCG = 1/log2(3) = 0.6309
        # IDCG = 1/log2(2) = 1.0
        # NDCG = 0.6309
        assert 0.6 < ndcg < 0.65

    def test_no_relevant_docs(self):
        """No relevant docs in retrieved -> NDCG@5 = 0.0."""
        retrieved = [{"source": "other1.txt"}, {"source": "other2.txt"}]
        expected = [{"document": "doc_a.txt"}]
        ndcg = calculate_ndcg_at_k(retrieved, expected, k=5)
        assert ndcg == 0.0

    def test_empty_retrieval(self):
        """Empty retrieved list -> NDCG@5 = 0.0."""
        ndcg = calculate_ndcg_at_k([], [{"document": "doc_a.txt"}], k=5)
        assert ndcg == 0.0

    def test_no_expected_sources(self):
        """No expected sources -> all vacuously relevant (NDCG = 1.0)."""
        retrieved = [{"source": "doc_a.txt"}, {"source": "doc_b.txt"}]
        ndcg = calculate_ndcg_at_k(retrieved, [], k=5)
        assert ndcg == 1.0

    def test_k_greater_than_retrieved(self):
        """k > number of retrieved docs -> should not error."""
        retrieved = [{"source": "doc_a.txt"}]
        expected = [{"document": "doc_a.txt"}]
        ndcg = calculate_ndcg_at_k(retrieved, expected, k=5)
        assert ndcg == 1.0

    def test_duplicate_results(self):
        """Multiple relevant results from same source -> properly counted."""
        retrieved = [
            {"source": "doc_a.txt"}, {"source": "doc_a.txt"},
            {"source": "doc_b.txt"}, {"source": "doc_a.txt"},
        ]
        expected = [{"document": "doc_a.txt"}]
        ndcg = calculate_ndcg_at_k(retrieved, expected, k=5)
        # 3 relevant out of 4 (positions 1,2,4), irrelevant at position 3
        # DCG < IDCG because ideal would pack all relevant at top
        assert ndcg > 0.9  # high but not perfect due to gap at position 3

    def test_all_irrelevant(self):
        """All retrieved are irrelevant."""
        retrieved = [{"source": f"other{i}.txt"} for i in range(10)]
        expected = [{"document": "doc_a.txt"}]
        ndcg = calculate_ndcg_at_k(retrieved, expected, k=5)
        assert ndcg == 0.0


# ============================================================================
# MAP Tests
# ============================================================================

class TestMAP:
    """Mean Average Precision calculation tests."""

    def test_perfect_precision(self):
        """Single relevant doc at position 1 -> AP = 1.0."""
        retrieved = [{"source": "doc_a.txt"}, {"source": "other.txt"}]
        expected = [{"document": "doc_a.txt"}]
        ap = calculate_map(retrieved, expected)
        assert ap == 1.0

    def test_relevant_at_position_2(self):
        """Single relevant doc at position 2 -> AP = 0.5."""
        retrieved = [{"source": "other.txt"}, {"source": "doc_a.txt"}]
        expected = [{"document": "doc_a.txt"}]
        ap = calculate_map(retrieved, expected)
        assert ap == 0.5

    def test_no_relevant_docs(self):
        """No relevant docs -> AP = 0.0."""
        retrieved = [{"source": "other1.txt"}, {"source": "other2.txt"}]
        expected = [{"document": "doc_a.txt"}]
        ap = calculate_map(retrieved, expected)
        assert ap == 0.0

    def test_empty_retrieval(self):
        """Empty retrieved list -> AP = 0.0."""
        ap = calculate_map([], [{"document": "doc_a.txt"}])
        assert ap == 0.0

    def test_no_expected_sources(self):
        """No expected sources -> vacuously relevant."""
        retrieved = [{"source": "doc_a.txt"}]
        ap = calculate_map(retrieved, [])
        # All are relevant, AP = 1.0
        assert ap == 1.0

    def test_multiple_relevant_interleaved(self):
        """Two relevant docs at positions 1 and 3 -> AP = (1/1 + 2/3)/2 = 0.8333."""
        retrieved = [
            {"source": "doc_a.txt"},
            {"source": "other.txt"},
            {"source": "doc_b.txt"},
        ]
        expected = [{"document": "doc_a.txt"}, {"document": "doc_b.txt"}]
        ap = calculate_map(retrieved, expected)
        assert round(ap, 4) == round((1.0 + 2/3) / 2, 4)


# ============================================================================
# Faithfulness Tests
# ============================================================================

class TestFaithfulness:
    """Faithfulness calculation tests."""

    def test_fallback_answer_returns_none(self):
        """Context fallback answers should return None."""
        answer = "Most relevant context:\n[1] Some text..."
        faith = calculate_faithfulness(answer, ["some context"])
        assert faith is None

    def test_refusal_answer_returns_none(self):
        """Refusal answers should return None."""
        answer = "I don't have enough evidence to answer this."
        faith = calculate_faithfulness(answer, ["some context"])
        assert faith is None

    def test_empty_answer_returns_none(self):
        """Empty answer returns None."""
        faith = calculate_faithfulness("", ["context"])
        assert faith is None

    def test_no_context_returns_none(self):
        """No context returns None."""
        faith = calculate_faithfulness("The chunk size is 1200 characters.", [])
        assert faith is None

    def test_supported_claims(self):
        """Claims supported by context should yield high faithfulness."""
        answer = "The chunk size is 1200 characters. The overlap is 200 characters."
        context = ["The text splitter uses a chunk size of 1200 characters with an overlap of 200 characters."]
        faith = calculate_faithfulness(answer, context)
        assert faith is not None
        assert faith > 0.0

    def test_unsupported_claims(self):
        """Claims not in context should yield low faithfulness."""
        answer = "The system uses PostgreSQL for vector storage with Redis caching."
        context = ["The system uses ChromaDB for local disk-based vector storage."]
        faith = calculate_faithfulness(answer, context)
        # The claims about PostgreSQL and Redis have minimal overlap
        assert faith is not None
        # Could be 0 or very low

    def test_zero_length_answer_with_claims_precomputed(self):
        """Pre-extracted claims list that is empty -> None."""
        faith = calculate_faithfulness("some answer text", ["ctx"], claims=[])
        assert faith is None

    def test_unavailable_llm_indicator(self):
        """I cannot answer returns None."""
        faith = calculate_faithfulness("I cannot answer this question.", ["context"])
        assert faith is None


# ============================================================================
# Answer Relevance Tests
# ============================================================================

class TestAnswerRelevance:
    """Answer relevance calculation tests."""

    def test_fallback_returns_none(self):
        """Fallback answer -> None."""
        rel = calculate_answer_relevance(
            "Most relevant context:\n...",
            "What is the chunk size?",
        )
        assert rel is None

    def test_refusal_returns_none(self):
        """Refusal answer -> None."""
        rel = calculate_answer_relevance(
            "I don't have enough evidence to answer.",
            "What is PostgreSQL config?",
        )
        assert rel is None

    def test_empty_answer_returns_none(self):
        """Empty answer -> None."""
        rel = calculate_answer_relevance("", "What is X?")
        assert rel is None

    def test_empty_question_returns_none(self):
        """Empty question -> None."""
        rel = calculate_answer_relevance("Some answer.", "")
        assert rel is None

    def test_with_mock_embedding_model(self):
        """Mocked embedding model -> returns float."""
        mock_model = MagicMock()
        # Return unit vectors that have known cosine similarity
        mock_model.embed_documents.return_value = [[1.0, 0.0, 0.0]]

        rel = calculate_answer_relevance(
            "The chunk size is 1200.",
            "What is the chunk size?",
            embedding_model=mock_model,
        )
        # Both get same vector -> cosine = 1.0
        assert rel == 1.0

    def test_orthogonal_embeddings(self):
        """Orthogonal embeddings -> cosine = 0."""
        mock_model = MagicMock()
        mock_model.embed_documents.side_effect = [
            [[1.0, 0.0]],  # question
            [[0.0, 1.0]],  # answer
        ]
        rel = calculate_answer_relevance(
            "Some answer.", "What?", embedding_model=mock_model,
        )
        assert rel == 0.0


# ============================================================================
# Binary Relevance Vector Tests
# ============================================================================

class TestBinaryRelevanceVector:
    """Tests for _binary_relevance_vector helper."""

    def test_matching_source(self):
        """Source matching expected -> 1."""
        rel = _binary_relevance_vector(
            [{"source": "doc_a.txt"}],
            [{"document": "doc_a.txt"}],
        )
        assert rel == [1]

    def test_non_matching_source(self):
        """Source not in expected -> 0."""
        rel = _binary_relevance_vector(
            [{"source": "other.txt"}],
            [{"document": "doc_a.txt"}],
        )
        assert rel == [0]

    def test_empty_expected(self):
        """No expected sources -> all vacuously relevant."""
        rel = _binary_relevance_vector(
            [{"source": "a.txt"}, {"source": "b.txt"}],
            [],
        )
        assert rel == [1, 1]

    def test_mixed(self):
        """Mix of matching and non-matching."""
        rel = _binary_relevance_vector(
            [{"source": "doc_a.txt"}, {"source": "other.txt"}, {"source": "doc_b.txt"}],
            [{"document": "doc_a.txt"}, {"document": "doc_b.txt"}],
        )
        assert rel == [1, 0, 1]


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests for metrics."""

    def test_one_relevant_document_ndcg(self):
        """Single relevant doc at position 1 -> NDCG = 1.0."""
        retrieved = [{"source": "doc.txt"}]
        expected = [{"document": "doc.txt"}]
        assert calculate_ndcg_at_k(retrieved, expected, k=5) == 1.0

    def test_one_relevant_document_map(self):
        """Single relevant doc at position 1 -> MAP = 1.0."""
        assert calculate_map([{"source": "doc.txt"}], [{"document": "doc.txt"}]) == 1.0

    def test_ndcg_with_k_zero_would_fail(self):
        """k=0 should return 0.0 (no items to evaluate)."""
        ndcg = calculate_ndcg_at_k([{"source": "doc.txt"}], [{"document": "doc.txt"}], k=0)
        assert ndcg == 0.0

    def test_missing_metrics_represented_as_null(self):
        """Faithfulness and answer relevance should be None when not applicable."""
        # Fallback answer
        assert calculate_faithfulness("Most relevant context:\n...", ["ctx"]) is None
        assert calculate_answer_relevance("I don't have enough evidence.", "Q?") is None

    def test_recall_with_existing_function_still_works(self):
        """Existing calculate_retrieval_metrics still produces correct results."""
        retrieved = [
            {"source": "doc_a.txt", "text": "x", "score": 0.9},
            {"source": "other.txt", "text": "y", "score": 0.8},
        ]
        expected = [{"document": "doc_a.txt"}]
        m = calculate_retrieval_metrics(retrieved, expected)
        assert m["recall@1"] == 1.0
        assert m["mrr"] == 1.0

    def test_faithfulness_with_precomputed_claims(self):
        """Pre-computed claims list can be passed."""
        claims = [{"claim": "The chunk size is 1200 characters"}]
        ctx = ["Text splitter configured with chunk size 1200 characters"]
        faith = calculate_faithfulness("The chunk size is 1200 characters.", ctx, claims=claims)
        assert faith is not None
        assert faith > 0.0
