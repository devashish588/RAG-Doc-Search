"""Production smoke test — run against a live instance.

Usage:
    python evals/production_smoke_test.py [BASE_URL]

Default BASE_URL: http://127.0.0.1:9826
"""
import json
import sys
import urllib.request
import urllib.error

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9826"
passed = 0
failed = 0
skipped = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def get(path: str) -> tuple[int, str, dict]:
    url = f"{BASE_URL}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            headers = dict(resp.headers)
            return resp.status, resp.read().decode("utf-8"), headers
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace"), dict(e.headers)
    except Exception as e:
        return 0, str(e), {}


def post_json(path: str, data: dict) -> tuple[int, str, dict]:
    url = f"{BASE_URL}{path}"
    try:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8"), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace"), dict(e.headers)
    except Exception as e:
        return 0, str(e), {}


print(f"\n{'='*60}")
print(f"Production Smoke Test — {BASE_URL}")
print(f"{'='*60}\n")

# --- Health probes ---
print("[Health Probes]")
code, body, _ = get("/health")
check("GET /health returns 200", code == 200, f"got {code}")
try:
    data = json.loads(body)
    check("/health status is ok", data.get("status") == "ok")
except Exception:
    check("/health returns valid JSON", False, body[:100])

code, body, _ = get("/healthz")
check("GET /healthz returns 200", code == 200, f"got {code}")
try:
    data = json.loads(body)
    check("/healthz status is ok", data.get("status") == "ok")
except Exception:
    check("/healthz returns valid JSON", False, body[:100])

code, body, _ = get("/readyz")
check("GET /readyz returns 200 or 503", code in (200, 503), f"got {code}")

# --- Metrics ---
print("\n[Metrics]")
code, body, headers = get("/metrics")
check("GET /metrics returns 200", code == 200, f"got {code}")
check("Content-Type is text/plain", "text/plain" in headers.get("Content-Type", ""), headers.get("Content-Type"))
check("http_requests_total present", "http_requests_total" in body)
check("No API keys in metrics", "openrouter_api_key" not in body.lower())

# --- Request ID ---
print("\n[Request ID]")
_, _, headers = get("/health")
check("X-Request-ID header present", "X-Request-ID" in headers, str(list(headers.keys())))

# --- Rate limiting ---
print("\n[Rate Limiting]")
_, _, headers = get("/health")
check("X-RateLimit-Limit header present", "X-RateLimit-Limit" in headers)
check("X-RateLimit-Remaining header present", "X-RateLimit-Remaining" in headers)

# --- Ask endpoints ---
print("\n[Ask Endpoints - Retrieval Modes]")
modes = ["dense", "sparse", "hybrid", "hybrid_rerank"]
for mode in modes:
    code, body, _ = post_json("/v1/ask", {"question": "What is the maximum file upload size?", "retrieval_mode": mode})
    check(f"POST /v1/ask ({mode}) returns 200", code == 200, f"got {code}: {body[:150]}")
    try:
        data = json.loads(body)
        check(f"  response has 'answer'", "answer" in data)
        check(f"  response has 'confidence'", "confidence" in data)
        check(f"  response has 'retrieval_trace'", "retrieval_trace" in data)
        check(f"  response has 'grounding_metrics'", "grounding_metrics" in data)
    except Exception:
        check(f"  response is valid JSON", False)

# --- Error handling ---
print("\n[Error Handling]")
code, body, _ = post_json("/v1/ask", {"question": ""})
check("Empty question returns 422", code == 422, f"got {code}")

code, body, _ = post_json("/v1/ask", {"question": "test", "retrieval_mode": "bad"})
check("Invalid mode returns 422", code == 422, f"got {code}")

code, body, _ = get("/nonexistent")
check("404 returns structured error", code == 404, f"got {code}")

# --- Summary ---
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*60}")
sys.exit(1 if failed else 0)
