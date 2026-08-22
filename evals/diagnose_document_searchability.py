import os
import sys
import json
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ingestion import register_document, get_document_status
from backend.ingestion_v2 import ingest_document_v2
from backend.vector_store import get_vector_store
from backend.bm25_index import get_bm25_index
from backend.retrieval_dense import get_dense_retriever
from backend.retrieval_bm25 import get_bm25_retriever
from backend.retrieval_hybrid import get_hybrid_retriever
from backend.retrieval import run_search
from backend.schemas import SearchRequest
from backend.vector_store import get_embeddings


def diagnose(pdf_file_path: str = None, query: str = "System Discipline syllabus architectural patterns"):
    print("=" * 80)
    print("HYBRIDRAG END-TO-END SEARCHABILITY DIAGNOSTIC")
    print("=" * 80)

    # 1. Runtime Environment
    print(f"[1] RUNTIME ENVIRONMENT")
    print(f"    Executable:      {sys.executable}")
    print(f"    Python Version:  {sys.version.split()[0]}")
    print(f"    Working Dir:     {os.getcwd()}")

    # 2. Document Ingestion / Targeting
    doc_id = "diag_test_doc_9826"
    filename = "Syllabus_System_Discipline.pdf"

    if pdf_file_path and Path(pdf_file_path).exists():
        target_path = Path(pdf_file_path).resolve()
    else:
        candidate = PROJECT_ROOT / "data" / "uploads" / "eaf19b71022943d58c8d5a6263206de7_Syllabus_System_Discipline.pdf"
        target_path = candidate if candidate.exists() else PROJECT_ROOT / "docs" / "00_Document_Index_and_Start_Here.pdf"

    print(f"\n[2] TARGET DOCUMENT INGESTION")
    print(f"    Target Path:     {target_path}")
    print(f"    Document ID:     {doc_id}")

    register_document(doc_id, filename, target_path)
    ingest_res = ingest_document_v2(doc_id, target_path, filename)
    doc_status = get_document_status(doc_id)

    print(f"    Ingestion Status: {doc_status.get('status')}")
    print(f"    Chunks Indexed:   {doc_status.get('chunks_indexed')}")
    print(f"    Total Chunks:     {doc_status.get('total_chunks')}")
    print(f"    Progress Pct:     {doc_status.get('progress_pct')}%")
    print(f"    Message:          {doc_status.get('message')}")

    # 3. Direct Chroma Verification
    print(f"\n[3] CHROMA COLLECTION VERIFICATION")
    vs = get_vector_store()
    col = getattr(vs, "_collection", None)
    total_records = col.count() if col else 0
    print(f"    Total Chroma Records: {total_records}")

    # Query Chroma for target document_id in metadata
    target_chunks = []
    if col:
        try:
            get_res = col.get(where={"document_id": doc_id}, include=["documents", "metadatas"])
            target_ids = get_res.get("ids") or []
            target_metas = get_res.get("metadatas") or []
            target_docs = get_res.get("documents") or []
            print(f"    Target Document Records: {len(target_ids)}")
            print(f"    Target Chunk IDs:        {target_ids}")
            for i, (cid, m, text) in enumerate(zip(target_ids, target_metas, target_docs)):
                print(f"      [{i}] ID={cid} | source={m.get('source')} | doc_id={m.get('document_id')} | snippet={text[:60]!r}")
        except Exception as exc:
            print(f"    Chroma get error: {exc}")

    # 4. Direct BM25 Verification
    print(f"\n[4] BM25 INDEX VERIFICATION")
    bm25 = get_bm25_index()
    bm25_chunk_ids = getattr(bm25, "_chunk_ids", [])
    print(f"    Total BM25 Chunks:   {len(bm25_chunk_ids)}")
    bm25_doc_chunks = [cid for cid in bm25_chunk_ids if doc_id in cid or any(m.get('document_id') == doc_id for m in getattr(bm25, '_chunk_metadatas', []))]
    print(f"    Target BM25 Chunks:  {len(bm25_doc_chunks)} ({bm25_doc_chunks})")

    # 5. Layer-by-Layer Retrieval Matrix
    print(f"\n[5] LAYER-BY-LAYER RETRIEVAL MATRIX")
    print(f"    Query: {query!r}")

    # 5a. Direct Embeddings + Chroma Query
    print(f"\n    [5a] Direct Embeddings + Chroma query:")
    embedder = get_embeddings()
    q_vec = embedder.embed_query(query)
    if col:
        query_res = col.query(query_embeddings=[q_vec], n_results=5, include=["documents", "metadatas", "distances"])
        q_ids = query_res.get("ids", [[]])[0]
        q_dists = query_res.get("distances", [[]])[0]
        q_docs = query_res.get("documents", [[]])[0]
        q_metas = query_res.get("metadatas", [[]])[0]
        for i, (cid, dist, txt, meta) in enumerate(zip(q_ids, q_dists, q_docs, q_metas)):
            is_target = "YES" if meta.get("document_id") == doc_id else "NO"
            print(f"      Rank {i+1}: dist={dist:.4f} | target={is_target} | ID={cid} | source={meta.get('source')} | snippet={txt[:60]!r}")

    # 5b. DenseRetriever
    print(f"\n    [5b] DenseRetriever:")
    dense_retriever = get_dense_retriever()
    dense_res = dense_retriever.search(query, k=5)
    for i, r in enumerate(dense_res):
        is_target = "YES" if r.metadata.get("document_id") == doc_id else "NO"
        print(f"      Rank {i+1}: score={r.score:.4f} | target={is_target} | doc_id={r.metadata.get('document_id')} | source={r.source} | snippet={r.content[:60]!r}")

    # 5c. BM25Retriever
    print(f"\n    [5c] BM25Retriever:")
    bm25_retriever = get_bm25_retriever()
    bm25_res = bm25_retriever.search(query, k=5)
    for i, r in enumerate(bm25_res):
        is_target = "YES" if r.metadata.get("document_id") == doc_id else "NO"
        print(f"      Rank {i+1}: score={r.score:.4f} | target={is_target} | doc_id={r.metadata.get('document_id')} | source={r.source} | snippet={r.content[:60]!r}")

    # 5d. HybridRetriever
    print(f"\n    [5d] HybridRetriever:")
    hybrid_retriever = get_hybrid_retriever()
    hybrid_res = hybrid_retriever.search(query, k=5)
    for i, r in enumerate(hybrid_res):
        is_target = "YES" if r.metadata.get("document_id") == doc_id else "NO"
        print(f"      Rank {i+1}: score={r.score:.4f} | target={is_target} | doc_id={r.metadata.get('document_id')} | source={r.source} | snippet={r.content[:60]!r}")

    # 5e. Application API run_search
    print(f"\n    [5e] Application API run_search:")
    search_req = SearchRequest(query=query, top_k=5)
    api_res = run_search(search_req)
    for i, r in enumerate(api_res.results):
        is_target = "YES" if r.metadata.get("document_id") == doc_id else "NO"
        print(f"      Rank {i+1}: score={r.score:.4f} | target={is_target} | source={r.source} | snippet={r.text[:60]!r}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    diagnose()
