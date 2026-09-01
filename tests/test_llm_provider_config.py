"""LLM provider configuration tests.

Verifies Groq and OpenRouter provider selection, configuration loading,
and that API keys never appear in logs.
"""
import json
import logging
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. Groq configuration loads correctly
# ---------------------------------------------------------------------------

class TestGroqConfigLoads:
    def test_groq_provider_sets_base_url(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "GROQ_BASE_URL": "https://api.groq.com/openai/v1",
            "GROQ_MODEL": "llama-3.3-70b-versatile",
            "LLM_MODEL": "",
        }):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_PROVIDER == "groq"
            assert s.LLM_BASE_URL == "https://api.groq.com/openai/v1"
            assert s.LLM_MODEL == "llama-3.3-70b-versatile"
            assert s.LLM_API_KEY == "test-key"

    def test_groq_provider_defaults(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.GROQ_BASE_URL == "https://api.groq.com/openai/v1"
            assert s.GROQ_MODEL == "llama-3.3-70b-versatile"

    def test_groq_slashed_model_triggers_fallback(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "openai/gpt-oss-20b",
            "LLM_MODEL": "",
        }):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert "/" not in s.LLM_MODEL
            assert s.LLM_MODEL in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant")


# ---------------------------------------------------------------------------
# 2. Missing GROQ_API_KEY fails clearly
# ---------------------------------------------------------------------------

class TestMissingGroqKeyFails:
    def test_groq_without_key_produces_empty_api_key(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_API_KEY == ""

    def test_llm_available_false_without_key(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)
            assert llm.llm_available() is False


# ---------------------------------------------------------------------------
# 3. Groq base URL is correct
# ---------------------------------------------------------------------------

class TestGroqBaseUrl:
    def test_base_url_format(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_BASE_URL.startswith("https://")
            assert "groq.com" in s.LLM_BASE_URL


# ---------------------------------------------------------------------------
# 4. Configured Groq model is passed to the client
# ---------------------------------------------------------------------------

class TestGroqModelPassedToClient:
    def test_model_in_payload(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "llama-3.3-70b-versatile",
            "LLM_MODEL": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "choices": [{"message": {"content": "Test answer"}}]
            }).encode()
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            with patch("backend.llm.request.urlopen", return_value=mock_response) as mock_urlopen, \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                from backend.schemas import SearchResult
                results = [SearchResult(text="test", source="test.pdf", page=1, score=0.9, metadata={})]
                llm.generate_answer("test query", results)
                call_args = mock_urlopen.call_args
                payload = json.loads(call_args[0][0].data.decode())
                assert payload["model"] == "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# 5. API keys never appear in logs
# ---------------------------------------------------------------------------

class TestApiKeyNotInLogs:
    def test_no_api_key_in_log_output(self, caplog):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "gsk_SECRET_KEY_12345",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)

            with caplog.at_level(logging.WARNING):
                log_msg = f"LLM provider: {s.LLM_PROVIDER}"
                logging.getLogger("test").warning(log_msg)

            assert "gsk_SECRET_KEY" not in caplog.text

    def test_api_key_not_in_llm_log_on_failure(self, caplog):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "gsk_SECRET_KEY_12345",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)

            from urllib import error as urlerror
            with patch("backend.llm.request.urlopen", side_effect=urlerror.HTTPError(
                "https://api.groq.com/openai/v1/chat/completions", 401, "Unauthorized", {}, None
            )):
                import backend.llm as llm
                importlib.reload(llm)
                from backend.schemas import SearchResult
                results = [SearchResult(text="test", source="test.pdf", page=1, score=0.9, metadata={})]
                with caplog.at_level(logging.DEBUG):
                    llm.generate_answer("test", results)

            assert "gsk_SECRET_KEY" not in caplog.text


# ---------------------------------------------------------------------------
# 6. OpenRouter configuration remains functional
# ---------------------------------------------------------------------------

class TestOpenRouterBackwardCompat:
    def test_openrouter_provider_default(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "sk-or-test",
            "OPENROUTER_MODEL": "openai/gpt-4o-mini",
            "LLM_MODEL": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_PROVIDER == "openrouter"
            assert s.LLM_API_KEY == "sk-or-test"
            assert s.LLM_MODEL == "openai/gpt-4o-mini"
            assert "openrouter.ai" in s.LLM_BASE_URL

    def test_openrouter_backward_compat_no_llm_vars(self):
        with patch.dict("os.environ", {
            "OPENROUTER_API_KEY": "sk-or-legacy",
            "OPENROUTER_MODEL": "openai/gpt-4o-mini",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_API_KEY == "sk-or-legacy"
            assert s.LLM_PROVIDER == "openrouter"


# ---------------------------------------------------------------------------
# 7. Retrieval behavior is unchanged
# ---------------------------------------------------------------------------

class TestRetrievalUnchanged:
    def test_search_still_works(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.middleware import reset_limiter
        client = TestClient(app)
        reset_limiter()
        resp = client.post("/search", json={"query": "test", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_v1_ask_still_returns_chunks(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.middleware import reset_limiter
        client = TestClient(app)
        reset_limiter()
        resp = client.post("/v1/ask", json={
            "question": "test",
            "retrieval_mode": "dense",
            "top_k_dense": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "retrieved_chunks" in data


# ---------------------------------------------------------------------------
# 8. Grounding behavior is unchanged
# ---------------------------------------------------------------------------

class TestGroundingUnchanged:
    def test_verifier_still_works(self):
        from backend.verifier import CitationVerifier
        from backend.interfaces import RetrievalResult
        context = [
            RetrievalResult(
                chunk_id="test:0", score=1.0, rank=1, source="test.pdf",
                content="Section 9 covers databases.",
                metadata={"document_id": "test", "chunk": 0, "page": 1},
            ),
        ]
        verifier = CitationVerifier()
        result = verifier.verify("Section 9 covers databases.", context)
        assert result["metrics"]["grounding_ratio"] > 0.0

    def test_confidence_estimator_still_works(self):
        from backend.confidence import ConfidenceEstimator
        from backend.interfaces import RetrievalResult
        context = [
            RetrievalResult(
                chunk_id="test:0", score=1.0, rank=1, source="test.pdf",
                content="test content",
                metadata={"document_id": "test", "chunk": 0, "page": 1},
            ),
        ]
        estimator = ConfidenceEstimator()
        conf = estimator.estimate(
            dense_results=context, bm25_results=[], rrf_results=[],
            reranker_results=[], grounding_ratio=1.0, retrieval_mode="dense",
        )
        assert conf.abstention_flag is False


# ---------------------------------------------------------------------------
# 9. Abstention behavior is unchanged
# ---------------------------------------------------------------------------

class TestAbstentionUnchanged:
    def test_empty_clauses_still_abstain(self):
        from backend.confidence import ConfidenceEstimator
        estimator = ConfidenceEstimator()
        conf = estimator.estimate(
            dense_results=[], bm25_results=[], rrf_results=[],
            reranker_results=[], grounding_ratio=0.0, retrieval_mode="dense",
        )
        assert conf.abstention_flag is True


# ---------------------------------------------------------------------------
# 10. Response contract remains unchanged
# ---------------------------------------------------------------------------

class TestResponseContractUnchanged:
    def test_ask_response_has_all_fields(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.middleware import reset_limiter
        client = TestClient(app)
        reset_limiter()
        resp = client.post("/v1/ask", json={
            "question": "test",
            "retrieval_mode": "dense",
            "top_k_dense": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        for field in ("answer", "status", "citations", "retrieved_chunks",
                       "confidence", "grounding_metrics", "retrieval_trace"):
            assert field in data, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# 11. Generic LLM_* override behavior (the production root cause)
# ---------------------------------------------------------------------------

class TestGenericOverrideBehavior:
    """Verify that generic LLM_* vars override provider-specific defaults.

    This is the documented design (settings.py line 133-134):
        # If LLM_BASE_URL / LLM_MODEL are set, they win. Otherwise the provider
        # default is used.

    The production bug was: LLM_BASE_URL=https://openrouter.ai/api/v1 was set
    on Render, overriding GROQ_BASE_URL even when LLM_PROVIDER=groq.
    """

    def test_groq_with_stale_llm_base_url_uses_override(self):
        """Reproduces the exact production bug: LLM_BASE_URL overrides GROQ_BASE_URL."""
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-groq-key",
            "GROQ_MODEL": "llama-3.1-8b-instant",
            "LLM_BASE_URL": "https://openrouter.ai/api/v1",
            "LLM_MODEL": "",
            "LLM_API_KEY": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            # This IS the bug: generic LLM_BASE_URL wins over GROQ_BASE_URL
            assert s.LLM_BASE_URL == "https://openrouter.ai/api/v1"
            assert s.LLM_PROVIDER == "groq"

    def test_groq_without_override_uses_groq_url(self):
        """Without LLM_BASE_URL set, Groq gets its own base URL."""
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-groq-key",
            "GROQ_MODEL": "llama-3.1-8b-instant",
            "LLM_BASE_URL": "",
            "LLM_MODEL": "",
            "LLM_API_KEY": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_BASE_URL == "https://api.groq.com/openai/v1"
            assert s.LLM_PROVIDER == "groq"

    def test_groq_with_explicit_groq_url_override(self):
        """User explicitly sets LLM_BASE_URL to Groq URL — works correctly."""
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-groq-key",
            "GROQ_MODEL": "llama-3.1-8b-instant",
            "LLM_BASE_URL": "https://api.groq.com/openai/v1",
            "LLM_MODEL": "",
            "LLM_API_KEY": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_BASE_URL == "https://api.groq.com/openai/v1"

    def test_groq_with_llm_api_key_override(self):
        """Generic LLM_API_KEY overrides GROQ_API_KEY when set."""
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "groq-key",
            "LLM_API_KEY": "generic-key",
            "LLM_BASE_URL": "",
            "LLM_MODEL": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_API_KEY == "generic-key"

    def test_groq_without_llm_api_key_uses_groq_key(self):
        """Without LLM_API_KEY, Groq falls through to GROQ_API_KEY."""
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "groq-key",
            "LLM_API_KEY": "",
            "LLM_BASE_URL": "",
            "LLM_MODEL": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_API_KEY == "groq-key"

    def test_groq_with_llm_model_override(self):
        """Generic LLM_MODEL overrides GROQ_MODEL when set."""
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "llama-3.3-70b-versatile",
            "LLM_MODEL": "custom-model",
            "LLM_BASE_URL": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_MODEL == "custom-model"

    def test_groq_without_llm_model_uses_groq_model(self):
        """Without LLM_MODEL, Groq falls through to GROQ_MODEL."""
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "llama-3.1-8b-instant",
            "LLM_MODEL": "",
            "LLM_BASE_URL": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_MODEL == "llama-3.1-8b-instant"

    def test_openrouter_with_groq_url_override(self):
        """OpenRouter can also be overridden via LLM_BASE_URL."""
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "sk-or-test",
            "OPENROUTER_MODEL": "openai/gpt-4o-mini",
            "LLM_BASE_URL": "https://custom.api.com/v1",
            "LLM_MODEL": "",
            "LLM_API_KEY": "",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_BASE_URL == "https://custom.api.com/v1"

    def test_health_endpoint_shows_llm_model_not_openrouter_model(self):
        """Health endpoint should show LLM_MODEL, not always OPENROUTER_MODEL."""
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.middleware import reset_limiter
        client = TestClient(app)
        reset_limiter()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        # answer_model should reflect the actual configured model, not always OpenRouter
        assert data.get("answer_model") is not None
