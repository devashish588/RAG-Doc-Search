import json
import logging
import socket
import time
from urllib import error, request

from backend.circuit_breaker import get_circuit_breaker
from backend.schemas import SearchResult
from backend.settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER

log = logging.getLogger(__name__)
_diag = logging.getLogger("rag.llm.diag")


def llm_available() -> bool:
    return bool(LLM_API_KEY)


def _classify_http_error(code: int) -> str:
    if code == 400:
        return "invalid_request"
    if code == 401:
        return "invalid_api_key"
    if code == 402:
        return "payment_required"
    if code == 403:
        return "network_restriction_or_banned"
    if code == 404:
        return "model_not_found"
    if code == 429:
        return "rate_limited"
    if 500 <= code < 600:
        return "provider_error"
    return "http_error"


def _classify_url_error(exc) -> str:
    reason = getattr(exc, "reason", None)
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "timeout"
    return "connection_error"


def _messages(query: str, results: list[SearchResult]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{i + 1}] {r.source}" + (f" (page {r.page})" if r.page else "") + f":\n{r.text}"
        for i, r in enumerate(results)
    )
    system = (
        "You are a factual QA assistant. Answer the user's question clearly and concisely "
        "using ONLY the provided Context.\n\n"
        "FORMATTING RULES:\n"
        "1. Do NOT use bullet points starting with asterisks (*) or hyphens (-).\n"
        "2. Use bold titles for headings and major key points (e.g., **1. Section Title:**).\n"
        "3. Keep descriptions in numbered lists or clear paragraphs under bold headers.\n"
        "4. Always cite sources as [1], [2], etc."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_answer(query: str, results: list[SearchResult]) -> str | None:
    breaker = get_circuit_breaker()
    if not breaker.allow_request():
        _diag.warning(
            "llm_skip: provider=%s circuit_breaker=open",
            LLM_PROVIDER,
        )
        log.warning("Circuit breaker OPEN — skipping LLM call, using fallback")
        return None

    context_chars = sum(len(r.text) for r in results)
    _diag.info(
        "llm_request_start: provider=%s base_url=%s model=%s query_len=%d context_chunks=%d context_chars=%d",
        LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL, len(query), len(results), context_chars,
    )

    # Groq model IDs are bare (no '/'). A '/' means an OpenRouter-style
    # "provider/model" name was copied by mistake and will 404 on Groq.
    if LLM_PROVIDER == "groq" and "/" in LLM_MODEL:
        _diag.warning(
            "llm_model_format_warn: provider=groq model=%s contains '/'; "
            "Groq models are bare IDs (e.g. llama-3.1-8b-instant), not 'provider/model' "
            "(OpenRouter-style). This will likely 404.",
            LLM_MODEL,
        )

    payload = {
        "model": LLM_MODEL,
        "messages": _messages(query, results),
        "max_tokens": 768,
        "temperature": 0.0,
    }
    req = request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        _diag.info(
            "llm_request_ok: provider=%s base_url=%s model=%s http=200 latency_ms=%d "
            "response_chars=%d prompt_tokens=%s completion_tokens=%s",
            LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL, elapsed_ms, len(content),
            usage.get("prompt_tokens"), usage.get("completion_tokens"),
        )
        breaker.record_success()
        return content
    except error.HTTPError as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        body_snippet = ""
        try:
            body_snippet = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass

        reason = _classify_http_error(exc.code)
        _diag.error(
            "llm_request_error: provider=%s base_url=%s model=%s http=%d reason=%s category=%s latency_ms=%d "
            "exception=%s body=%s",
            LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL, exc.code, reason, reason, elapsed_ms,
            type(exc).__name__, body_snippet,
        )
        breaker.record_failure()
        return None
    except error.URLError as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        reason = _classify_url_error(exc)
        _diag.error(
            "llm_request_error: provider=%s base_url=%s model=%s reason=%s category=connection_or_timeout latency_ms=%d "
            "exception=%s message=%s",
            LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL, reason, elapsed_ms,
            type(exc).__name__, str(exc.reason)[:200],
        )
        breaker.record_failure()
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        _diag.error(
            "llm_request_error: provider=%s model=%s reason=malformed_response latency_ms=%d "
            "exception=%s message=%s",
            LLM_PROVIDER, LLM_MODEL, elapsed_ms,
            type(exc).__name__, str(exc)[:200],
        )
        breaker.record_failure()
        return None


def diagnostic_groq_ping() -> dict:
    """Minimal provider reachability check. Returns status dict, never raises."""
    if not llm_available():
        return {"ok": False, "reason": "no_api_key", "provider": LLM_PROVIDER}
    breaker = get_circuit_breaker()
    if not breaker.allow_request():
        return {"ok": False, "reason": "circuit_breaker_open", "provider": LLM_PROVIDER}

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: GROQ_OK"}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    req = request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            ok = "GROQ_OK" in content
            return {
                "ok": ok,
                "reason": "ok" if ok else "unexpected_response",
                "http": 200,
                "provider": LLM_PROVIDER,
                "model": LLM_MODEL,
                "response_snippet": content[:50],
            }
    except error.HTTPError as exc:
        return {"ok": False, "reason": _classify_http_error(exc.code), "http": exc.code, "provider": LLM_PROVIDER, "model": LLM_MODEL}
    except error.URLError as exc:
        return {"ok": False, "reason": _classify_url_error(exc), "provider": LLM_PROVIDER, "model": LLM_MODEL}
    except Exception as exc:
        return {"ok": False, "reason": type(exc).__name__, "provider": LLM_PROVIDER, "model": LLM_MODEL}
