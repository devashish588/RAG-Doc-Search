"""Cold-start vs warm-request latency measurement.

Usage:
    python evals/measure_production_latency.py [BASE_URL] [ITERATIONS]

Default BASE_URL: http://127.0.0.1:9826
Default ITERATIONS: 5
"""
import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9826"
ITERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 5


def measure(path, timeout=10):
    url = f"{BASE_URL}{path}"
    start = time.perf_counter()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            elapsed = (time.perf_counter() - start) * 1000
            return resp.status, elapsed, len(body)
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - start) * 1000
        return e.code, elapsed, 0
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return 0, elapsed, 0


def measure_post(path, data, timeout=60):
    url = f"{BASE_URL}{path}"
    body_bytes = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body_bytes, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            elapsed = (time.perf_counter() - start) * 1000
            return resp.status, elapsed, len(body)
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - start) * 1000
        return e.code, elapsed, 0
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return 0, elapsed, 0


print(f"\n{'='*60}")
print(f"Production Latency Measurement — {BASE_URL}")
print(f"{'='*60}\n")

# --- Phase 1: First request (potential cold start) ---
print("[Phase 1 — Initial Request (potential cold start)]")
results = {}
for endpoint, path in [
    ("healthz", "/healthz"),
    ("readyz", "/readyz"),
    ("ask_dense", None),
]:
    if endpoint == "ask_dense":
        status, latency_ms, size = measure_post("/v1/ask", {
            "question": "What is the maximum file upload size?",
            "retrieval_mode": "dense",
        })
    else:
        status, latency_ms, size = measure(path)
    results[endpoint] = {"status": status, "latency_ms": round(latency_ms, 1), "size": size}
    label = "COLD?" if latency_ms > 2000 else "warm"
    print(f"  {endpoint:15s}  HTTP {status}  {latency_ms:8.1f}ms  {size:6d}B  [{label}]")

cold_ask = results.get("ask_dense", {}).get("latency_ms", 0)

# --- Phase 2: Warm requests ---
print(f"\n[Phase 2 — Warm Requests ({ITERATIONS} iterations)]")
warm_latencies = []
for i in range(ITERATIONS):
    status, latency_ms, size = measure("/healthz")
    warm_latencies.append(latency_ms)
    print(f"  healthz #{i+1}  HTTP {status}  {latency_ms:8.1f}ms")

# --- Phase 3: Ask warm ---
print(f"\n[Phase 3 — Ask Warm ({ITERATIONS} iterations)]")
ask_latencies = []
for i in range(ITERATIONS):
    status, latency_ms, size = measure_post("/v1/ask", {
        "question": "What is the maximum file upload size?",
        "retrieval_mode": "dense",
    })
    ask_latencies.append(latency_ms)
    print(f"  ask_dense #{i+1}  HTTP {status}  {latency_ms:8.1}ms")

# --- Summary ---
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")

warm_health_mean = sum(warm_latencies) / len(warm_latencies) if warm_latencies else 0
warm_ask_mean = sum(ask_latencies) / len(ask_latencies) if ask_latencies else 0

print(f"  Initial healthz:     {results['healthz']['latency_ms']:8.1f}ms  {'COLD?' if results['healthz']['latency_ms'] > 2000 else 'warm'}")
print(f"  Initial readyz:      {results['readyz']['latency_ms']:8.1f}ms  {'COLD?' if results['readyz']['latency_ms'] > 2000 else 'warm'}")
print(f"  Initial ask_dense:   {cold_ask:8.1f}ms  {'COLD?' if cold_ask > 5000 else 'warm'}")
print(f"  Warm healthz mean:   {warm_health_mean:8.1f}ms")
print(f"  Warm ask_dense mean: {warm_ask_mean:8.1f}ms")

if cold_ask > 5000 and warm_ask_mean < 2000:
    print("\n  OBSERVATION: Cold start detected on initial ask request.")
    print(f"  Cold/warm ratio: {cold_ask / warm_ask_mean:.1f}x")
elif cold_ask > warm_ask_mean * 2:
    print(f"\n  OBSERVATION: Initial request was {cold_ask / warm_ask_mean:.1f}x slower than warm mean.")
else:
    print("\n  OBSERVATION: No significant cold start detected in this run.")

print(f"\n  Classification methodology:")
print(f"    healthz > 2000ms = potential cold start")
print(f"    ask > 5000ms = potential cold start")
print(f"    Otherwise = warm")
print(f"\n{'='*60}")
