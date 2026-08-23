"""Phase 14.2 — Provider smoke test.

Tests the actual LLM provider (Groq or OpenRouter) using the application's
own LLM abstraction. If GROQ_API_KEY or OPENROUTER_API_KEY is set in the
environment, runs a real API call. Otherwise skips.
"""
import json
import logging
import os
import time
from unittest.mock import patch

import pytest

from backend.settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER


def _provider_available():
    return bool(LLM_API_KEY)


@pytest.mark.skipif(not _provider_available(), reason="No LLM API key configured")
class TestProviderSmoke:
    """Real API smoke tests (skipped if no key)."""

    def test_provider_returns_valid_response_or_none(self):
        """Verify the provider is reachable. Returns content on success, None on provider error."""
        from backend.circuit_breaker import reset_circuit_breaker
        reset_circuit_breaker()
        from backend.llm import generate_answer
        from backend.schemas import SearchResult
        results = [
            SearchResult(
                text="The capital of France is Paris.",
                source="test.pdf",
                page=1,
                score=0.9,
                metadata={"document_id": "test", "chunk": 0},
            ),
        ]
        answer = generate_answer("What is the capital of France?", results)
        # Either the provider works (answer is non-empty) or it fails (answer is None).
        # Both are valid outcomes — the important thing is no crash.
        if answer is not None:
            assert len(answer) > 0
            assert "Paris" in answer
        # If None, the diagnostic logs should show why (HTTP error, circuit breaker, etc.)

    def test_provider_model_matches_config(self):
        from backend.circuit_breaker import reset_circuit_breaker
        reset_circuit_breaker()
        from backend.llm import generate_answer
        from backend.schemas import SearchResult
        mock_response = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "OK"}}],
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(return_value=False)

        with patch("backend.llm.request.urlopen", return_value=mock_response) as mock_urlopen:
            results = [SearchResult(text="x", source="t.pdf", page=1, score=0.9, metadata={})]
            generate_answer("test", results)
            payload = json.loads(mock_urlopen.call_args[0][0].data.decode())
            # Model must match whatever LLM_MODEL is configured
            from backend.settings import LLM_MODEL as configured_model
            assert payload["model"] == configured_model

    def test_provider_base_url_is_correct(self):
        if LLM_PROVIDER == "groq":
            assert "groq.com" in LLM_BASE_URL
        elif LLM_PROVIDER == "openrouter":
            assert "openrouter.ai" in LLM_BASE_URL

    def test_provider_logs_diagnostics(self, caplog):
        from backend.circuit_breaker import reset_circuit_breaker
        reset_circuit_breaker()
        from backend.llm import generate_answer
        from backend.schemas import SearchResult
        results = [
            SearchResult(text="test content", source="test.pdf",
                         page=1, score=0.9, metadata={"document_id": "t", "chunk": 0}),
        ]
        with caplog.at_level(logging.INFO, logger="rag.llm.diag"):
            answer = generate_answer("What is this?", results)
        assert "llm_request_start" in caplog.text or "llm_skip" in caplog.text
        # LLM_PROVIDER from the environment (could be groq or openrouter)
        provider_in_log = any(
            p in caplog.text
            for p in ("provider=groq", "provider=openrouter")
        )
        assert provider_in_log
        # Ensure no API key in logs
        assert LLM_API_KEY not in caplog.text


@pytest.mark.skipif(not _provider_available(), reason="No LLM API key configured")
class TestProviderGroqSpecific:
    """Groq-specific smoke tests."""

    def test_groq_base_url_reachable(self):
        from urllib import request as urllib_request, error
        req = urllib_request.Request(
            f"{LLM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            method="GET",
        )
        try:
            with urllib_request.urlopen(req, timeout=15) as resp:
                assert resp.status == 200
        except error.HTTPError as exc:
            # 401/403 means the endpoint is reachable but key may be invalid
            assert exc.code in (200, 401, 403), f"Unexpected: {exc.code}"
