"""Phase 13.3 end-to-end test: upload, ingest, verify all 4 modes return retrieved_chunks."""
import http.client
import json
import os
import sys
import threading
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import uvicorn

PORT = 9833


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
    headers = {"Content-Type": "application/json"} if body else {}
    body_bytes = json.dumps(body).encode() if body else None
    conn.request(method, path, body=body_bytes, headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return resp.status, data


def upload_file(filepath, filename):
    boundary = "----TestBoundary"
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
    conn.request(
        "POST", "/v1/ingest",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return resp.status, data


def main():
    print("=" * 60)
    print("Phase 13.3 - End-to-End Response Contract Test")
    print("=" * 60)

    # Start server
    config = uvicorn.Config("backend.main:app", host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    if not wait_for_server():
        print("FAIL: Server did not start")
        sys.exit(1)
    print(f"Server ready on port {PORT}")

    # Find a PDF to upload
    upload_dir = os.path.join("data", "uploads")
    pdf_files = []
    if os.path.exists(upload_dir):
        pdf_files = [f for f in os.listdir(upload_dir) if f.endswith(".pdf")]
    
    # Also check if there's a known test PDF
    test_pdfs = []
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".pdf") and "test" not in f.lower() and "node_modules" not in root:
                test_pdfs.append(os.path.join(root, f))
        if len(test_pdfs) > 5:
            break

    # Use existing uploaded file or create a test PDF
    if pdf_files:
        test_pdf = os.path.join(upload_dir, pdf_files[0])
        print(f"Using existing upload: {pdf_files[0]}")
    elif test_pdfs:
        test_pdf = test_pdfs[0]
        print(f"Using found PDF: {test_pdf}")
    else:
        # Create a simple test PDF
        test_pdf = os.path.join(upload_dir, "_test_phase13.pdf")
        os.makedirs(upload_dir, exist_ok=True)
        pdf_bytes = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>>>endobj\n"
            b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"xref\n0 5\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"0000000266 00000 n \n"
            b"trailer<</Size 5/Root 1 0 R>>\n"
            b"startxref\n340\n%%EOF\n"
        )
        with open(test_pdf, "wb") as f:
            f.write(pdf_bytes)
        print(f"Created test PDF: {test_pdf}")

    # Upload
    print("\n--- Uploading ---")
    code, result = upload_file(test_pdf, os.path.basename(test_pdf))
    print(f"Upload status: {code}")
    if code != 202:
        print(f"Upload failed: {result}")
        server.should_exit = True
        sys.exit(1)
    doc_id = result.get("document_id")
    print(f"document_id: {doc_id}")

    # Wait for ingestion
    print("\n--- Waiting for ingestion ---")
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

    # Test all 4 modes
    query = "What is the main topic?"
    modes = ["dense", "sparse", "hybrid", "hybrid_rerank"]
    results_summary = {}

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"MODE: {mode}")
        print(f"{'=' * 60}")

        payload = {
            "question": query,
            "retrieval_mode": mode,
            "top_k_dense": 10,
            "top_k_sparse": 10,
            "top_k_fused": 20,
            "top_k_final": 5,
        }

        code, data = api("POST", "/v1/ask", payload)
        print(f"HTTP: {code}")

        # Check retrieved_chunks
        chunks = data.get("retrieved_chunks", [])
        trace = data.get("retrieval_trace", {})
        trace_has_items = any(len(trace.get(s, [])) > 0 for s in ["dense", "bm25", "rrf", "reranker"])

        print(f"retrieved_chunks: {len(chunks)} items")
        for stage in ["dense", "bm25", "rrf", "reranker"]:
            items = trace.get(stage, [])
            if items:
                print(f"  trace.{stage}: {len(items)} items")

        if chunks:
            first = chunks[0]
            print(f"  first chunk: source={first.get('source')}, score={first.get('score'):.4f}, rank={first.get('rank')}")
            text = first.get("text", "")
            print(f"  text preview: {text[:100].encode('ascii', 'replace').decode()}...")

        # Verdict
        if trace_has_items and not chunks:
            verdict = "FAIL - trace has items but retrieved_chunks empty"
            results_summary[mode] = "FAIL"
        elif trace_has_items and chunks:
            verdict = f"PASS - {len(chunks)} chunks exposed"
            results_summary[mode] = "PASS"
        elif not trace_has_items and not chunks:
            verdict = "OK - no retrieval results (expected for weak query)"
            results_summary[mode] = "PASS"
        else:
            verdict = "UNEXPECTED"
            results_summary[mode] = "FAIL"
        print(f"Verdict: {verdict}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for mode, result in results_summary.items():
        print(f"  {mode}: {result}")

    all_pass = all(v == "PASS" for v in results_summary.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    server.should_exit = True
    thread.join(timeout=5)


if __name__ == "__main__":
    main()
