"""Production dependency smoke test.

Usage:
    python evals/production_dependency_smoke_test.py [BASE_URL]

Default BASE_URL: http://127.0.0.1:9826
"""
import io
import json
import sys
import time
import uuid
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


def post_multipart(path, field_name, filename, content, content_type="text/plain"):
    url = f"{BASE_URL}{path}"
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    try:
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8"), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace"), dict(e.headers)
    except Exception as e:
        return 0, str(e), {}


print(f"\n{'='*60}")
print(f"Production Dependency Smoke Test — {BASE_URL}")
print(f"{'='*60}\n")

# --- Health probes ---
print("[Health Probes]")
for name, path in [("healthz", "/healthz"), ("readyz", "/readyz"), ("metrics", "/metrics")]:
    code, body, _ = get(path)
    check(f"GET /{name} → 200", code == 200, f"got {code}")

# --- Dependency check via readyz ---
print("\n[Dependency Health]")
code, body, _ = get("/readyz")
if code == 200:
    data = json.loads(body)
    dep_check = data.get("checks", {}).get("dependencies", {})
    check("dependencies status ok", dep_check.get("status") == "ok", f"got {dep_check}")
else:
    check("readyz returns 200", False, f"got {code}")

# --- Ingestion ---
print("\n[Ingestion]")
test_content = b"Dependency smoke test document. This verifies the production ingestion pipeline works end-to-end."
test_filename = f"dep_smoke_{uuid.uuid4().hex[:8]}.txt"
code, body, _ = post_multipart("/v1/ingest", "file", test_filename, test_content)
check("Upload accepted", code == 202, f"got {code}: {body[:200]}")
if code == 202:
    doc_id = json.loads(body).get("document_id")
    # Poll until complete
    for _ in range(60):
        time.sleep(0.5)
        code, body, _ = get(f"/v1/documents/{doc_id}")
        if code == 200:
            status = json.loads(body)
            if status.get("status") in ("complete", "failed"):
                break
    check("Ingestion complete", status.get("status") == "complete", f"got {status.get('status')}")
    check("Chunks indexed >= 1", status.get("chunks_indexed", 0) >= 1, f"got {status.get('chunks_indexed')}")

# --- Four-mode retrieval ---
print("\n[Four-Mode Retrieval]")
for mode in ["dense", "sparse", "hybrid", "hybrid_rerank"]:
    code, body, _ = post_json("/v1/ask", {
        "question": "dependency smoke test",
        "retrieval_mode": mode,
    })
    if code == 200:
        data = json.loads(body)
        check(f"{mode}: answer present", "answer" in data)
        check(f"{mode}: confidence present", "confidence" in data)
        check(f"{mode}: trace present", "retrieval_trace" in data)
    else:
        check(f"{mode}: HTTP 200", False, f"got {code}")

# --- Summary ---
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*60}")
sys.exit(1 if failed else 0)
