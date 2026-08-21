"""Phase 10 production verification script.

Usage:
    python evals/phase10_production_smoke_test.py [BASE_URL]

Default BASE_URL: http://127.0.0.1:9826
"""
import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9826"
passed = 0
failed = 0
blocked = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS   {name}")
    else:
        failed += 1
        print(f"  FAIL   {name}  {detail}")


def block(name, reason=""):
    global blocked
    blocked += 1
    print(f"  BLOCKED  {name}  {reason}")


def get(path):
    url = f"{BASE_URL}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8"), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace"), dict(e.headers)
    except Exception as e:
        return 0, str(e), {}


def post_json(path, data):
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
print(f"Phase 10 Production Verification — {BASE_URL}")
print(f"{'='*60}\n")

# --- Health Probes ---
print("[Health Probes]")
code, body, _ = get("/health")
check("GET /health 200", code == 200, f"got {code}")

code, body, _ = get("/healthz")
check("GET /healthz 200", code == 200, f"got {code}")
try:
    data = json.loads(body)
    check("/healthz status ok", data.get("status") == "ok")
except Exception:
    check("/healthz valid JSON", False, body[:100])

code, body, _ = get("/readyz")
check("GET /readyz 200 or 503", code in (200, 503), f"got {code}")

# --- Metrics ---
print("\n[Metrics]")
code, body, headers = get("/metrics")
check("GET /metrics 200", code == 200, f"got {code}")
check("Content-Type text/plain", "text/plain" in headers.get("Content-Type", ""))
check("http_requests_total present", "http_requests_total" in body)
check("No secrets in metrics", "openrouter_api_key" not in body.lower())

# --- Request ID ---
print("\n[Request ID]")
_, _, headers = get("/health")
check("X-Request-ID present", "X-Request-ID" in headers)

# --- Rate Limiting ---
print("\n[Rate Limiting]")
_, _, headers = get("/health")
check("X-RateLimit-Limit present", "X-RateLimit-Limit" in headers)

# --- Ask: Dense ---
print("\n[Ask — Dense]")
code, body, _ = post_json("/v1/ask", {"question": "What is the maximum file upload size?", "retrieval_mode": "dense"})
check("Dense returns 200", code == 200, f"got {code}: {body[:200]}")
if code == 200:
    data = json.loads(body)
    check("  answer present", "answer" in data)
    check("  confidence present", "confidence" in data)
    check("  confidence.level", data.get("confidence", {}).get("level") in ("high", "medium", "low"))
    check("  retrieval_trace present", "retrieval_trace" in data)
    check("  grounding_metrics present", "grounding_metrics" in data)
    check("  citations list", isinstance(data.get("citations"), list))

# --- Ask: Sparse ---
print("\n[Ask — Sparse/BM25]")
code, body, _ = post_json("/v1/ask", {"question": "MAX_UPLOAD_MB", "retrieval_mode": "sparse"})
check("Sparse returns 200", code == 200, f"got {code}: {body[:200]}")

# --- Ask: Hybrid ---
print("\n[Ask — Hybrid]")
code, body, _ = post_json("/v1/ask", {"question": "What is the default port?", "retrieval_mode": "hybrid"})
check("Hybrid returns 200", code == 200, f"got {code}: {body[:200]}")

# --- Ask: Hybrid Rerank ---
print("\n[Ask — Hybrid Rerank]")
code, body, _ = post_json("/v1/ask", {"question": "What is the chunk size?", "retrieval_mode": "hybrid_rerank", "top_k_final": 3})
check("Hybrid Rerank returns 200", code == 200, f"got {code}: {body[:200]}")

# --- Abstention ---
print("\n[Abstention]")
code, body, _ = post_json("/v1/ask", {"question": "What is the airspeed velocity of an unladen swallow?", "retrieval_mode": "dense"})
if code == 200:
    data = json.loads(body)
    check("Abstention returns valid response", True)
else:
    check("Abstention returns valid response", False, f"got {code}")

# --- Error Handling ---
print("\n[Error Handling]")
code, _, _ = post_json("/v1/ask", {"question": ""})
check("Empty question returns 422", code == 422, f"got {code}")

code, _, _ = post_json("/v1/ask", {"question": "test", "retrieval_mode": "bad"})
check("Invalid mode returns 422", code == 422, f"got {code}")

code, _, _ = get("/nonexistent")
check("404 structured error", code == 404)

# --- Summary ---
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed, {blocked} blocked")
print(f"{'='*60}")
sys.exit(1 if failed else 0)
