"""Phase 13.3 diagnostic: starts server in-process, captures /v1/ask response for all 4 modes."""
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

PORT = 9830


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


def main():
    print("=" * 60)
    print("Phase 13.3 - Response Contract Diagnostic (Post-Fix)")
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

    # List documents
    print("\n--- GET /documents ---")
    code, docs = api("GET", "/documents")
    doc_list = docs.get("documents", docs) if isinstance(docs, dict) else docs
    for d in doc_list:
        print(f"  {d.get('filename', '?')}: status={d.get('status')}, chunks={d.get('chunks_indexed')}/{d.get('total_chunks')}")

    if not doc_list:
        print("No documents found.")
        server.should_exit = True
        return

    query = "What is the attendance policy?"
    modes = ["dense", "sparse", "hybrid", "hybrid_rerank"]

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
        print(f"Status: {code}")
        print(f"Response keys: {list(data.keys())}")

        # Check retrieved_chunks
        chunks = data.get("retrieved_chunks", [])
        print(f"\nretrieved_chunks: {len(chunks)} items")
        if chunks:
            first = chunks[0]
            print(f"  first chunk keys: {list(first.keys())}")
            print(f"  chunk_id: {first.get('chunk_id')}")
            print(f"  source: {first.get('source')}")
            print(f"  score: {first.get('score')}")
            print(f"  rank: {first.get('rank')}")
            print(f"  page: {first.get('page')}")
            text = first.get("text", "")
            print(f"  text preview: {text[:120]}...")

        # Check trace counts
        trace = data.get("retrieval_trace", {})
        for stage in ["dense", "bm25", "rrf", "reranker"]:
            items = trace.get(stage, [])
            print(f"  trace.{stage}: {len(items)} items")

        # Check citations
        citations = data.get("citations", [])
        print(f"  citations: {len(citations)} items")

        # Check confidence
        conf = data.get("confidence", {})
        print(f"  confidence: {conf.get('overall_score')} ({conf.get('level')})")
        print(f"  abstention: {conf.get('abstention_flag')}")
        print(f"  answer: {data.get('answer', '?')[:100]}")

        # Verify retrieved_chunks are present when trace has items
        trace_has_items = any(len(trace.get(s, [])) > 0 for s in ["dense", "bm25", "rrf", "reranker"])
        if trace_has_items and not chunks:
            print(f"\n  *** BUG STILL PRESENT: trace has items but retrieved_chunks is empty ***")
        elif trace_has_items and chunks:
            print(f"\n  OK: retrieved_chunks has {len(chunks)} items matching trace")
        else:
            print(f"\n  INFO: no trace items for this mode")

    server.should_exit = True
    thread.join(timeout=5)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
