"""Phase 14.2+ — LLM provider error categorization and model validation tests."""
import json
import logging
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. Error categorization — HTTP errors
# ---------------------------------------------------------------------------

class TestLLMErrorCategorization:
    """Verify _classify_http_error and generate_answer logs correct reasons."""

    @pytest.mark.parametrize("code,expected_reason", [
        (400, "invalid_request"),
        (401, "invalid_api_key"),
        (402, "payment_required"),
        (403, "network_restriction_or_banned"),
        (404, "model_not_found"),
        (429, "rate_limited"),
        (500, "provider_error"),
        (502, "provider_error"),
        (503, "provider_error"),
        (418, "http_error"),  # unknown code
    ])
    def test_http_error_reason(self, code, expected_reason):
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)
            from backend.llm import _classify_http_error
            assert _classify_http_error(code) == expected_reason

    def test_httperror_404_logs_model_not_found(self, caplog):
        from urllib import error as urlerror
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            with patch("backend.llm.request.urlopen",
                        side_effect=urlerror.HTTPError(
                            "https://api.groq.com/openai/v1/chat/completions",
                            404, "Not Found", {}, None
                        )), \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                from backend.schemas import SearchResult
                with caplog.at_level(logging.ERROR, logger="rag.llm.diag"):
                    llm.generate_answer("test", [
                        SearchResult(text="test", source="t.pdf", page=1, score=0.9, metadata={})
                    ])

        assert "reason=model_not_found" in caplog.text

    def test_httperror_401_logs_invalid_api_key(self, caplog):
        from urllib import error as urlerror
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            with patch("backend.llm.request.urlopen",
                        side_effect=urlerror.HTTPError(
                            "https://api.groq.com/openai/v1/chat/completions",
                            401, "Unauthorized", {}, None
                        )), \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                from backend.schemas import SearchResult
                with caplog.at_level(logging.ERROR, logger="rag.llm.diag"):
                    llm.generate_answer("test", [
                        SearchResult(text="test", source="t.pdf", page=1, score=0.9, metadata={})
                    ])

        assert "reason=invalid_api_key" in caplog.text

    def test_httperror_403_logs_network_restriction(self, caplog):
        from urllib import error as urlerror
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            with patch("backend.llm.request.urlopen",
                        side_effect=urlerror.HTTPError(
                            "https://api.groq.com/openai/v1/chat/completions",
                            403, "Forbidden", {}, None
                        )), \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                from backend.schemas import SearchResult
                with caplog.at_level(logging.ERROR, logger="rag.llm.diag"):
                    llm.generate_answer("test", [
                        SearchResult(text="test", source="t.pdf", page=1, score=0.9, metadata={})
                    ])

        assert "reason=network_restriction_or_banned" in caplog.text

    def test_urlerror_timeout_logs_timeout(self, caplog):
        from urllib import error as urlerror
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            timeout_exc = urlerror.URLError(TimeoutError("timed out"))
            with patch("backend.llm.request.urlopen", side_effect=timeout_exc), \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                from backend.schemas import SearchResult
                with caplog.at_level(logging.ERROR, logger="rag.llm.diag"):
                    llm.generate_answer("test", [
                        SearchResult(text="test", source="t.pdf", page=1, score=0.9, metadata={})
                    ])

        assert "reason=timeout" in caplog.text

    def test_urlerror_connection_logs_connection_error(self, caplog):
        from urllib import error as urlerror
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            conn_exc = urlerror.URLError(OSError("connection refused"))
            with patch("backend.llm.request.urlopen", side_effect=conn_exc), \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                from backend.schemas import SearchResult
                with caplog.at_level(logging.ERROR, logger="rag.llm.diag"):
                    llm.generate_answer("test", [
                        SearchResult(text="test", source="t.pdf", page=1, score=0.9, metadata={})
                    ])

        assert "reason=connection_error" in caplog.text

    def test_malformed_response_logs_malformed(self, caplog):
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            # Return malformed JSON
            mock_response = MagicMock()
            mock_response.read.return_value = b"not json"
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)

            with patch("backend.llm.request.urlopen", return_value=mock_response), \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                from backend.schemas import SearchResult
                with caplog.at_level(logging.ERROR, logger="rag.llm.diag"):
                    llm.generate_answer("test", [
                        SearchResult(text="test", source="t.pdf", page=1, score=0.9, metadata={})
                    ])

        assert "reason=malformed_response" in caplog.text


# ---------------------------------------------------------------------------
# 2. Model format validation and fallback
# ---------------------------------------------------------------------------

class TestGroqModelFormatValidation:
    """Verify groq model '/' detection, warning, and fallback."""

    def test_groq_model_with_slash_warns_and_falls_back(self, caplog):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "openai/gpt-oss-20b",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            # Check the warning was logged during settings reload
            assert "contains '/'" in caplog.text or "invalid for Groq" in caplog.text
            # Check fallback happened
            assert s.LLM_MODEL == "llama-3.1-8b-instant"

    def test_groq_model_bare_id_no_warning(self, caplog):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "llama-3.1-8b-instant",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            # No warning for bare ID
            assert "contains '/'" not in caplog.text
            assert s.LLM_MODEL == "llama-3.1-8b-instant"

    def test_openrouter_model_not_affected(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "sk-or-test",
            "OPENROUTER_MODEL": "openai/gpt-4o-mini",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            assert s.LLM_MODEL == "openai/gpt-4o-mini"  # slash is OK for openrouter

    def test_groq_model_slash_warning_in_generate_answer(self, caplog):
        from urllib import error as urlerror
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "openai/gpt-oss-20b",
        }, clear=False):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            with patch("backend.llm.request.urlopen",
                        side_effect=urlerror.HTTPError(
                            "https://api.groq.com/openai/v1/chat/completions",
                            404, "Not Found", {}, None
                        )), \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                from backend.schemas import SearchResult
                with caplog.at_level(logging.WARNING):
                    llm.generate_answer("test", [
                        SearchResult(text="test", source="t.pdf", page=1, score=0.9, metadata={})
                    ])

        assert "contains '/'" in caplog.text


# ---------------------------------------------------------------------------
# 3. diagnostic_groq_ping
# ---------------------------------------------------------------------------

class TestDiagnosticGroqPing:
    """Test the diagnostic ping function."""

    def test_ping_returns_ok_when_groq_ok_in_response(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "choices": [{"message": {"content": "GROQ_OK"}}],
            }).encode()
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            with patch("backend.llm.request.urlopen", return_value=mock_response), \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                result = llm.diagnostic_groq_ping()
                assert result["ok"] is True
                assert result["reason"] == "ok"
                assert result["http"] == 200
                assert result["provider"] == "groq"

    def test_ping_returns_false_on_unexpected_response(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "choices": [{"message": {"content": "Some other response"}}],
            }).encode()
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            with patch("backend.llm.request.urlopen", return_value=mock_response), \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                result = llm.diagnostic_groq_ping()
                assert result["ok"] is False
                assert result["reason"] == "unexpected_response"

    def test_ping_returns_http_error_reason(self):
        from urllib import error as urlerror
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = True

            with patch("backend.llm.request.urlopen",
                        side_effect=urlerror.HTTPError(
                            "https://api.groq.com/openai/v1/chat/completions",
                            404, "Not Found", {}, None
                        )), \
                 patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                result = llm.diagnostic_groq_ping()
                assert result["ok"] is False
                assert result["reason"] == "model_not_found"
                assert result["http"] == 404

    def test_ping_returns_no_key_when_unavailable(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": ""}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            result = llm.diagnostic_groq_ping()
            assert result["ok"] is False
            assert result["reason"] == "no_api_key"

    def test_ping_circuit_breaker_open(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test"}):
            import importlib
            import backend.settings as s
            importlib.reload(s)
            import backend.llm as llm
            importlib.reload(llm)

            mock_breaker = MagicMock()
            mock_breaker.allow_request.return_value = False

            with patch("backend.llm.get_circuit_breaker", return_value=mock_breaker):
                result = llm.diagnostic_groq_ping()
                assert result["ok"] is False
                assert result["reason"] == "circuit_breaker_open"