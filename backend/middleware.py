"""In-memory sliding-window rate limiter (Phase 9)."""
import time
import threading
from collections import defaultdict
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from backend.settings import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS


class _SlidingWindow:
    """Thread-safe sliding-window counter per IP."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            hits = self._hits[key]
            # prune expired entries
            self._hits[key] = hits = [t for t in hits if t > cutoff]
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True

    def remaining(self, key: str) -> int:
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            hits = self._hits[key]
            hits[:] = [t for t in hits if t > cutoff]
            return max(0, self.max_requests - len(hits))

    def retry_after(self, key: str) -> float:
        """Seconds until the oldest hit in the window expires."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.get(key, [])
            if not hits:
                return 0.0
            return max(0.0, hits[0] + self.window - now)


# ponytail: module-level singleton, easy to reset in tests
_window = _SlidingWindow(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def reset_limiter(max_requests: Optional[int] = None, window_seconds: Optional[float] = None) -> None:
    """Reset the limiter (for tests)."""
    global _window
    _window = _SlidingWindow(
        max_requests or RATE_LIMIT_REQUESTS,
        window_seconds or RATE_LIMIT_WINDOW_SECONDS,
    )


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter.  Returns 429 with Retry-After when exceeded."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for health probes
        if request.url.path in ("/healthz", "/readyz", "/metrics"):
            return await call_next(request)

        ip = _client_ip(request)
        if not _window.allow(ip):
            retry = int(_window.retry_after(ip)) + 1
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please retry later.",
                    }
                },
                headers={"Retry-After": str(retry)},
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(_window.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(_window.remaining(ip))
        return response
