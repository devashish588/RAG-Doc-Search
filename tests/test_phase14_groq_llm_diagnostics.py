"""Phase 14.0 — Groq LLM Diagnostic and Error Observability Tests."""

import io
import json
import logging
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from backend.circuit_breaker import get_circuit_breaker
from backend.llm import generate_answer, llm_available
from backend.schemas import SearchResult


@pytest.fixture
def mock_search_results():
    return [
        SearchResult(
            text="Operating Systems cover process management, memory management, and file systems.",
            source="Syllabus_System_Discipline.pdf",
            page=1,
            score=0.95,
            metadata={"document_id": "doc123", "chunk": 0},
        )
    ]


class TestGroqLLMDiagnostics:
    """Test suite for Groq LLM diagnostics, privacy, and error classification."""

    def test_llm_available_check(self):
        with patch("backend.llm.LLM_API_KEY", "gsk_test_key_123"):
            assert llm_available() is True
        with patch("backend.llm.LLM_API_KEY", ""):
            assert llm_available() is False

    def test_groq_successful_response_and_privacy(self, mock_search_results, caplog):
        get_circuit_breaker().reset()
        mock_response_bytes = json.dumps({
            "choices": [{"message": {"content": "GROQ_OK: Operating System topics include process and memory management."}}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 20},
        }).encode("utf-8")

        mock_resp = io.BytesIO(mock_response_bytes)
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda self, *args: None

        test_key = "gsk_SECRET_KEY_NEVER_LOG"
        with patch("backend.llm.LLM_API_KEY", test_key), \
             patch("backend.llm.LLM_PROVIDER", "groq"), \
             patch("backend.llm.LLM_MODEL", "llama-3.3-70b-versatile"), \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             caplog.at_level(logging.DEBUG):

            ans = generate_answer("Reply with exactly: GROQ_OK", mock_search_results)

            assert ans is not None
            assert "GROQ_OK" in ans
            assert test_key not in caplog.text

    def test_groq_http_404_model_not_found_classification(self, mock_search_results, caplog):
        get_circuit_breaker().reset()
        err_body = json.dumps({"error": {"message": "The model `openai/gpt-oss-20b` does not exist", "type": "invalid_request_error"}}).encode("utf-8")
        mock_fp = MagicMock()
        mock_fp.read.return_value = err_body
        http_err = HTTPError("https://api.groq.com/openai/v1/chat/completions", 404, "Not Found", {}, mock_fp)

        with patch("backend.llm.LLM_PROVIDER", "groq"), \
             patch("backend.llm.LLM_MODEL", "openai/gpt-oss-20b"), \
             patch("urllib.request.urlopen", side_effect=http_err), \
             caplog.at_level(logging.INFO):

            ans = generate_answer("What topics are covered?", mock_search_results)

            assert ans is None
            assert "category=model_not_found" in caplog.text or "http=404" in caplog.text
            assert "openai/gpt-oss-20b" in caplog.text

    def test_groq_http_401_invalid_api_key(self, mock_search_results, caplog):
        get_circuit_breaker().reset()
        err_body = json.dumps({"error": {"message": "Invalid API Key", "type": "invalid_request_error"}}).encode("utf-8")
        mock_fp = MagicMock()
        mock_fp.read.return_value = err_body
        http_err = HTTPError("https://api.groq.com/openai/v1/chat/completions", 401, "Unauthorized", {}, mock_fp)

        with patch("backend.llm.LLM_PROVIDER", "groq"), \
             patch("urllib.request.urlopen", side_effect=http_err), \
             caplog.at_level(logging.INFO):

            ans = generate_answer("Test question", mock_search_results)

            assert ans is None
            assert "category=invalid_api_key" in caplog.text or "http=401" in caplog.text

    def test_groq_url_error_connection_failure(self, mock_search_results, caplog):
        get_circuit_breaker().reset()
        url_err = URLError("Connection refused")

        with patch("backend.llm.LLM_PROVIDER", "groq"), \
             patch("urllib.request.urlopen", side_effect=url_err), \
             caplog.at_level(logging.INFO):

            ans = generate_answer("Test question", mock_search_results)

            assert ans is None
            assert "llm_request_error" in caplog.text
            assert "connection_or_timeout" in caplog.text
