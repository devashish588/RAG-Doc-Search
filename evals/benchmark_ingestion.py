"""Phase 13 — Ingestion Performance Benchmarking Suite.

Measures stage-by-stage timing (parsing, normalization, chunking, deduplication,
embedding & vector store upsert, BM25 indexing, total time) and memory overhead
for target documents.
"""
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.bm25_index import get_bm25_index
from backend.chunking import get_chunker
from backend.deduplication import Deduplicator, deduplicate_chunks
from backend.loaders import get_loader
from backend.normalizer import get_normalizer
from backend.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKING_STRATEGY,
    EMBEDDING_BATCH_SIZE,
    FASTEMBED_BATCH_SIZE,
    NEAR_DUPLICATE_THRESHOLD,
)
from backend.vector_store import add_documents, get_embeddings, get_vector_store


def benchmark_single_document(file_path: Path) -> dict[str, Any]:
    """Run isolated ingestion benchmark for a single file and record stage metrics."""
    doc_id = f"bench_{file_path.stem}"
    file_bytes = file_path.stat().st_size

    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Parsing / Loading
    t0 = time.perf_counter()
    loader = get_loader(file_path)
    raw_docs = loader.load(file_path)
    t_parse = time.perf_counter() - t0

    # 2. Normalization
    t0 = time.perf_counter()
    normalizer = get_normalizer("standard")
    normalized_docs = [
        type(d)(page_content=normalizer.normalize(d.page_content), metadata=dict(d.metadata))
        for d in raw_docs
    ]
    t_norm = time.perf_counter() - t0

    # 3. Chunking
    t0 = time.perf_counter()
    chunker = get_chunker(CHUNKING_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = chunker.chunk(normalized_docs, doc_id, file_path.name)
    t_chunk = time.perf_counter() - t0

    raw_chunk_count = len(chunks)

    # 4. Deduplication
    t0 = time.perf_counter()
    embeddings = get_embeddings()

    def embed_fn(text: str) -> list[float]:
        import numpy as np
        emb = embeddings.embed_query(text)
        arr = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(arr)
        return (arr / norm).tolist() if norm > 0 else arr.tolist()

    kept_chunks, dup_results = deduplicate_chunks(
        chunks,
        near_duplicate_threshold=NEAR_DUPLICATE_THRESHOLD,
        embedder_fn=embed_fn,
    )
    t_dedup = time.perf_counter() - t0

    dedup_chunk_count = len(kept_chunks)

    # 5. Embedding & Vector Store Upsert
    t0 = time.perf_counter()
    langchain_chunks = [c.to_langchain_document() for c in kept_chunks]
    chunk_ids = [c.id for c in kept_chunks]

    indexed_count = add_documents(langchain_chunks, chunk_ids)
    t_embed_vector = time.perf_counter() - t0

    # 6. BM25 Indexing
    t0 = time.perf_counter()
    bm25 = get_bm25_index()
    bm25.build(kept_chunks)
    bm25.save()
    t_bm25 = time.perf_counter() - t0

    total_time = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    num_batches = (dedup_chunk_count + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE if EMBEDDING_BATCH_SIZE > 0 else 1
    chunks_per_sec = round(dedup_chunk_count / total_time, 2) if total_time > 0 else 0.0

    return {
        "filename": file_path.name,
        "document_size_bytes": file_bytes,
        "raw_chunk_count": raw_chunk_count,
        "deduplicated_chunk_count": dedup_chunk_count,
        "fastembed_batch_size": FASTEMBED_BATCH_SIZE,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "number_of_embedding_batches": num_batches,
        "parsing_time_ms": round(t_parse * 1000, 2),
        "normalization_time_ms": round(t_norm * 1000, 2),
        "chunking_time_ms": round(t_chunk * 1000, 2),
        "deduplication_time_ms": round(t_dedup * 1000, 2),
        "embedding_and_vector_upsert_time_ms": round(t_embed_vector * 1000, 2),
        "bm25_indexing_time_ms": round(t_bm25 * 1000, 2),
        "total_ingestion_time_ms": round(total_time * 1000, 2),
        "chunks_per_second": chunks_per_sec,
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2),
    }


def run_full_ingestion_benchmark() -> dict[str, Any]:
    """Benchmark all corpus files in evals/corpus/ and docs/."""
    corpus_dir = BASE_DIR / "evals" / "corpus"
    docs_dir = BASE_DIR / "docs"

    files = sorted(list(corpus_dir.glob("*.*")) + [f for f in docs_dir.glob("*.pdf")])
    results = []

    print("=" * 80)
    print("PHASE 13 INGESTION PERFORMANCE BENCHMARK")
    print(f"FastEmbed Batch Size: {FASTEMBED_BATCH_SIZE}")
    print(f"Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")
    print("=" * 80)

    for f in files:
        if not f.is_file():
            continue
        print(f"Benchmarking {f.name} ({f.stat().st_size} bytes)...")
        res = benchmark_single_document(f)
        results.append(res)
        print(
            f"  -> Total: {res['total_ingestion_time_ms']}ms | "
            f"Embed+Vector: {res['embedding_and_vector_upsert_time_ms']}ms | "
            f"Chunks: {res['deduplicated_chunk_count']} ({res['chunks_per_second']} ch/s) | "
            f"Peak RAM: {res['peak_memory_mb']}MB"
        )

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fastembed_batch_size": FASTEMBED_BATCH_SIZE,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "total_documents_benchmarked": len(results),
        "total_chunks_indexed": sum(r["deduplicated_chunk_count"] for r in results),
        "average_chunks_per_second": round(
            sum(r["chunks_per_second"] for r in results) / len(results), 2
        ) if results else 0,
        "document_results": results,
    }

    out_file = BASE_DIR / "evals" / "results" / "ingestion_benchmark_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 80)
    print(f"Benchmark summary saved to {out_file}")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    run_full_ingestion_benchmark()
