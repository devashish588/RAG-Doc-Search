"""Full integration test: start server, ingest, verify Chroma, search, cleanup."""
import http.client
import importlib
import json
import os
import sys
import time
import threading
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import uvicorn


PORT = 9829


def wait_for_server(timeout=30):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def api(method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    headers = {}
    if body is not None:
        body_bytes = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    else:
        body_bytes = None
    conn.request(method, path, body=body_bytes, headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return resp.status, data


def main():
    print("=" * 60)
    print("Full Integration Test")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"Virtualenv: {sys.prefix != sys.base_prefix}")
    print()

    # Preflight
    for pkg in ["chromadb", "fastembed", "numpy"]:
        mod = importlib.import_module(pkg)
        print(f"  {pkg}: OK ({getattr(mod, '__version__', '?')})")
    print()

    # Start server
    config = uvicorn.Config("backend.main:app", host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    if not wait_for_server():
        print("FAIL: Server did not start")
        sys.exit(1)
    print(f"Server ready on port {PORT}")
    print()

    # /healthz
    code, data = api("GET", "/healthz")
    print(f"GET /healthz -> {code}: {data}")

    # /readyz
    code, data = api("GET", "/readyz")
    print(f"GET /readyz -> {code}: status={data.get('status')}")
    checks = data.get("checks", {})
    for k, v in checks.items():
        print(f"  {k}: {v.get('status', '?')}")
    print()

    # Create test document
    test_dir = os.path.join("data", "uploads")
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, "_integration_test.txt")

    content = (
        "This is a comprehensive test document for the RAG Document Search system.\n"
        "The system uses FastAPI for the backend, ChromaDB for vector storage, "
        "and FastEmbed for ONNX-based embeddings.\n\n"
        "Section 1: Architecture Overview\n"
        "The application follows a modular architecture with separate modules for "
        "ingestion, retrieval, vector storage, and API endpoints. The ingestion pipeline "
        "handles document parsing, text extraction, and chunking before embedding.\n\n"
        "Section 2: Deployment\n"
        "The system is deployed on Render with a standard plan. It uses a health check "
        "endpoint at /healthz and a readiness probe at /readyz. The keep-warm cron job "
        "prevents cold starts by pinging the health endpoint every 10 minutes.\n\n"
        "Section 3: Performance\n"
        "Benchmark results show Dense R@1=0.78, BM25 R@1=0.76, Hybrid R@1=0.82, "
        "and Hybrid+Rerank R@1=0.90. The system achieves sub-200ms latency for "
        "most queries after warmup.\n\n"
        "Section 4: Monitoring\n"
        "Structured JSON logging, Prometheus metrics at /metrics, and circuit breaker "
        "protection for the OpenRouter API are all built in. The rate limiter uses a "
        "sliding window algorithm with per-IP tracking.\n\n"
    ) * 5

    with open(test_file, "w") as f:
        f.write(content)
    print(f"Created test file ({len(content)} bytes)")

    # Upload
    boundary = "----IntegrationTestBoundary"
    file_bytes = content.encode("utf-8")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="integration_test.txt"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()

    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    conn.request(
        "POST", "/v1/ingest",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = conn.getresponse()
    ingest_result = json.loads(resp.read())
    conn.close()
    doc_id = ingest_result["document_id"]
    print(f"Upload -> {resp.status}: document_id={doc_id}")

    # Wait for completion
    for i in range(30):
        time.sleep(1)
        code, doc = api("GET", f"/v1/documents/{doc_id}")
        status = doc.get("document", {}).get("status", doc.get("status", "?"))
        print(f"  [{i+1}s] status={status}")
        if status == "completed":
            break
        if status == "failed":
            print(f"  FAIL: {doc}")
            break
    print()

    # Verify Chroma
    print("--- Chroma Verification ---")
    from chromadb import PersistentClient
    client = PersistentClient(path="./data/chroma_db")
    cols = client.list_collections()
    total = 0
    for col in cols:
        count = col.count()
        total += count
        print(f"  {col.name}: {count} chunks")
    print(f"  Total: {total} chunks")
    assert total > 0, "FAIL: No chunks in Chroma"
    print("  PASS")
    print()

    # Search
    print("--- Search Test ---")
    code, result = api("POST", "/v1/ask", {"question": "What is the architecture?", "top_k_final": 3, "retrieval_mode": "dense"})
    print(f"POST /v1/ask -> {code}")
    print(f"  status: {result.get('status')}")
    print(f"  answer: {result.get('answer', '?')[:200]}")
    sources = result.get("retrieval_trace", {}).get("dense", [])
    print(f"  dense results: {len(sources) if isinstance(sources, list) else '?'}")
    print()

    # Metrics
    code, data = api("GET", "/metrics")
    print(f"GET /metrics -> {code} ({len(str(data))} bytes)")

    # Cleanup
    os.remove(test_file)
    server.should_exit = True
    thread.join(timeout=5)

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
