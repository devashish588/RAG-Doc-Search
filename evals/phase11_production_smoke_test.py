"""Phase 11 production smoke test.

Usage:
    python evals/phase11_production_smoke_test.py [BASE_URL]

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


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS   {name}")
    else:
        failed += 1
        print(f"  FAIL   {name}  {detail}")


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
print(f"Phase 11 Production Smoke Test — {BASE_URL}")
print(f"{'='*60}\n")

# --- Health ---
print("[Health Probes]")
for name, path in [("health", "/health"), ("healthz", "/healthz"), ("readyz", "/readyz")]:
    code, body, _ = get(path)
    check(f"GET /{name} → 200 or 503", code in (200, 503), f"got {code}")

# --- Metrics ---
print("\n[Metrics]")
code, body, headers = get("/metrics")
check("GET /metrics → 200", code == 200, f"got {code}")
check("Prometheus content type", "text/plain" in headers.get("Content-Type", ""))
check("http_requests_total present", "http_requests_total" in body)
check("No secrets in metrics", "openrouter_api_key" not in body.lower())

# --- Four-mode API ---
print("\n[Four-Mode API Verification]")
modes = [
    ("dense", {"question": "What is the maximum file upload size?", "retrieval_mode": "dense"}),
    ("sparse", {"question": "MAX_UPLOAD_MB", "retrieval_mode": "sparse"}),
    ("hybrid", {"question": "What is the default port?", "retrieval_mode": "hybrid"}),
    ("hybrid_rerank", {"question": "What is the chunk size?", "retrieval_mode": "hybrid_rerank", "top_k_final": 3}),
]

for mode_name, payload in modes:
    print(f"\n  [{mode_name}]")
    code, body, _ = post_json("/v1/ask", payload)
    check(f"  HTTP 200", code == 200, f"got {code}: {body[:150]}")
    if code == 200:
        data = json.loads(body)
        check("  answer present", "answer" in data)
        check("  citations present", "citations" in data)
        check("  confidence present", "confidence" in data)
        conf = data.get("confidence", {})
        check("  confidence.level", conf.get("level") in ("high", "medium", "low"))
        check("  retrieval_trace present", "retrieval_trace" in data)
        trace = data.get("retrieval_trace", {})
        check("  trace.dense exists", isinstance(trace.get("dense"), list))
        check("  trace.bm25 exists", isinstance(trace.get("bm25"), list))
        check("  trace.rrf exists", isinstance(trace.get("rrf"), list))
        check("  trace.reranker exists", isinstance(trace.get("reranker"), list))

# --- Abstention ---
print("\n[Abstention]")
code, body, _ = post_json("/v1/ask", {"question": "What is the airspeed velocity of an unladen swallow?", "retrieval_mode": "dense"})
if code == 200:
    data = json.loads(body)
    check("Abstention returns valid response", True)
else:
    check("Abstention returns valid response", False, f"got {code}")

# --- Error handling ---
print("\n[Error Handling]")
code, _, _ = post_json("/v1/ask", {"question": ""})
check("Empty question → 422", code == 422, f"got {code}")
code, _, _ = post_json("/v1/ask", {"question": "test", "retrieval_mode": "bad"})
check("Invalid mode → 422", code == 422, f"got {code}")

# --- Rate limit headers ---
print("\n[Rate Limit Headers]")
_, _, headers = get("/health")
check("X-RateLimit-Limit present", "X-RateLimit-Limit" in headers)
check("X-RateLimit-Remaining present", "X-RateLimit-Remaining" in headers)

# --- Request ID ---
print("\n[Request ID]")
_, _, headers = get("/health")
check("X-Request-ID present", "X-Request-ID" in headers)

# --- Summary ---
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*60}")
sys.exit(1 if failed else 0)
