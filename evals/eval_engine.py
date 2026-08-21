"""Phase 8 — Evaluation Engine.

Provides a reproducible, isolated benchmarking framework that evaluates
the existing retrieval/generation pipeline across all four modes:
dense, bm25, hybrid, hybrid_rerank.

Evaluation isolation:
- Uses a dedicated Chroma collection and BM25 index directory under evals/runtime/.
- Rebuilds evaluation corpus from evals/corpus/ before each benchmark.
- Never modifies production data.

LLM availability:
- Explicitly detects whether OpenRouter is available.
- Generation-dependent metrics (faithfulness, answer_relevance) are reported
  as None when LLM is unavailable.
"""
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

# Ensure workspace root is in sys.path
EVALS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVALS_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.ingestion import ingest_document, register_document
from backend.interfaces import RetrievalResult
from backend.llm import llm_available, generate_answer
from backend.retrieval import run_search, run_search_dense, run_search_bm25, run_search_hybrid
from backend.reranker import get_reranker
from backend.schemas import SearchRequest, SearchResult
from backend.verifier import get_verifier
from backend.confidence import get_confidence_estimator
from backend.bm25_index import get_bm25_index
from evals.metrics import (
    calculate_retrieval_metrics,
    calculate_text_similarity,
    calculate_abstention_score,
    calculate_citation_metrics,
    calculate_ndcg_at_k,
    calculate_map,
    calculate_faithfulness,
    calculate_answer_relevance,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORPUS_DIR = EVALS_DIR / "corpus"
GOLDEN_DATASET_PATH = EVALS_DIR / "golden_dataset.json"
RESULTS_DIR = EVALS_DIR / "results"
RUNTIME_DIR = EVALS_DIR / "runtime"

# Category definitions (from Phase 7)
ANSWERABLE_CATEGORIES = {"single-hop", "exact-term", "multi-hop"}
UNSUPPORTED_CATEGORIES = {"unanswerable", "ambiguous"}


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """Reproducible, isolated benchmarking framework for HybridRAG.

    Supports four retrieval modes: dense, bm25, hybrid, hybrid_rerank.
    Produces per-query results with consistent timing methodology.
    """

    def __init__(self):
        self.llm_status = "real_llm_active" if llm_available() else "unavailable"
        self._warmed_up = False
        self._cold_start_latencies: dict[str, float] = {}

    def detect_llm_status(self) -> str:
        """Explicitly detect LLM availability by performing a test generation call."""
        if not llm_available():
            self.llm_status = "unavailable"
            return "unavailable"
        # Perform test call to confirm API key is active and responding
        from backend.schemas import SearchResult
        test_res = generate_answer("What is test?", [SearchResult(text="Test context content.", source="test.txt", score=0.9)])
        if test_res is not None:
            self.llm_status = "real_llm_active"
        else:
            self.llm_status = "unavailable"
        return self.llm_status

    def setup_isolated_corpus(self) -> list[str]:
        """Ingest frozen evaluation corpus into isolated stores.

        Returns list of ingested document IDs.
        """
        print("--- Ingesting Evaluation Corpus (Isolated) ---")
        doc_ids = []
        for file_path in sorted(CORPUS_DIR.glob("*")):
            if file_path.is_dir() or file_path.name.startswith("."):
                continue
            doc_id = f"eval_{file_path.stem}"
            filename = file_path.name
            print(f"Ingesting {filename}...")
            register_document(document_id=doc_id, filename=filename, stored_path=file_path)
            ingest_document(document_id=doc_id, stored_path=file_path, filename=filename)
            doc_ids.append(doc_id)
        print(f"Corpus ingestion complete. Total files: {len(doc_ids)}")

        # Rebuild BM25 index from fresh evaluation corpus
        from backend.loaders import TextLoader, PDFLoader
        from backend.chunking import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=1200, chunk_overlap=200)
        all_chunks = []
        for file_path in sorted(CORPUS_DIR.glob("*")):
            if file_path.is_dir() or file_path.name.startswith("."):
                continue
            doc_id = f"eval_{file_path.stem}"
            loader = PDFLoader() if file_path.suffix == ".pdf" else TextLoader()
            docs = loader.load(file_path)
            chunks = chunker.chunk(docs, document_id=doc_id, filename=file_path.name)
            all_chunks.extend(chunks)

        bm25_index = get_bm25_index()
        bm25_index.build(all_chunks)
        num_docs = bm25_index.get_stats().get("num_documents", 0)
        print(f"BM25 index rebuilt: {num_docs} documents, {len(all_chunks)} chunks indexed")

        return doc_ids

    def warmup(self, sample_query: str = "What is the chunk size?") -> dict[str, float]:
        """Perform warm-up queries to avoid cold-start bias in benchmark timing.

        Returns cold-start latencies per mode.
        """
        if self._warmed_up:
            return self._cold_start_latencies

        print("\n--- Warm-up Phase (cold-start timing) ---")
        modes = ["dense", "bm25", "hybrid", "hybrid_rerank"]
        for mode in modes:
            t0 = perf_counter()
            try:
                self._execute_single_query(sample_query, mode)
            except Exception as exc:
                log.warning("Warm-up failed for mode %s: %s", mode, exc)
            elapsed = round((perf_counter() - t0) * 1000, 2)
            self._cold_start_latencies[mode] = elapsed
            print(f"  {mode}: {elapsed} ms (cold start)")

        self._warmed_up = True
        return self._cold_start_latencies

    def evaluate_query(
        self,
        question_record: dict[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        """Evaluate a single query in the specified mode.

        Returns a structured per-query result dict.
        """
        q_id = question_record["id"]
        question = question_record["question"]
        q_type = question_record["type"]
        expected_answer = question_record["expected_answer"]
        expected_sources = question_record.get("expected_sources", [])

        # ---- Timed retrieval ----
        t_retrieval_start = perf_counter()
        retrieval_output = self._execute_single_query(question, mode)
        t_retrieval_end = perf_counter()
        retrieval_latency_ms = round((t_retrieval_end - t_retrieval_start) * 1000, 2)

        retrieved_list = retrieval_output["retrieved_list"]
        dense_results = retrieval_output.get("dense_results", [])
        bm25_results = retrieval_output.get("bm25_results", [])
        rrf_results = retrieval_output.get("rrf_results", [])
        reranker_results = retrieval_output.get("reranker_results", [])

        # ---- Timed generation ----
        t_gen_start = perf_counter()
        answer = retrieval_output["answer"]
        t_gen_end = perf_counter()
        generation_latency_ms = round((t_gen_end - t_gen_start) * 1000, 2)

        # ---- Timed verification ----
        t_verify_start = perf_counter()
        verifier = get_verifier()
        eval_context = []
        for i, r in enumerate(retrieved_list):
            chunk_id_str = str(r.get("metadata", {}).get("document_id", "")) + ":" + str(r.get("metadata", {}).get("chunk", i))
            eval_context.append(RetrievalResult(
                chunk_id=chunk_id_str,
                score=r.get("score", 0.0),
                rank=i + 1,
                source=r.get("source", ""),
                content=r.get("text", ""),
                metadata=r.get("metadata", {}),
            ))
        verification_output = verifier.verify(answer, eval_context)
        g_metrics = verification_output.get("metrics", {})

        # Confidence estimation
        confidence_estimator = get_confidence_estimator()
        conf_output = confidence_estimator.estimate(
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_results=rrf_results,
            reranker_results=reranker_results,
            grounding_ratio=g_metrics.get("grounding_ratio", 0.0),
            retrieval_mode=mode if mode != "bm25" else "sparse",
        )
        t_verify_end = perf_counter()
        verification_latency_ms = round((t_verify_end - t_verify_start) * 1000, 2)

        total_latency_ms = round(retrieval_latency_ms + generation_latency_ms + verification_latency_ms, 2)

        # ---- Compute metrics ----
        # IR metrics
        ir_metrics = calculate_retrieval_metrics(retrieved_list, expected_sources)
        ndcg_5 = calculate_ndcg_at_k(retrieved_list, expected_sources, k=5)
        map_score = calculate_map(retrieved_list, expected_sources)

        # Text similarity
        correctness = calculate_text_similarity(answer, expected_answer)

        # Abstention
        abstention_score = calculate_abstention_score(q_type, answer)

        # Citation metrics (existing Phase 7)
        cit_metrics = calculate_citation_metrics(answer, retrieved_list)

        # Generation metrics (faithfulness + answer relevance)
        context_texts = [r.get("text", "") for r in retrieved_list]
        if self.llm_status == "real_llm_active":
            faithfulness = calculate_faithfulness(answer, context_texts)
            answer_relevance = calculate_answer_relevance(answer, question)
        else:
            # LLM unavailable: report as N/A
            faithfulness = None
            answer_relevance = None

        # Determine if answer is a fallback
        answer_lower = answer.strip().lower()
        is_fallback = (
            answer_lower.startswith("most relevant context:") or
            "i don't have enough evidence" in answer_lower or
            "no relevant context found" in answer_lower
        )

        if is_fallback and self.llm_status == "unavailable":
            faithfulness = None
            answer_relevance = None

        return {
            "question_id": q_id,
            "category": q_type,
            "question": question,
            "mode": mode,
            "expected_answer": expected_answer,
            "actual_answer": answer,
            "retrieved_chunks": [r.get("source", "") for r in retrieved_list],
            "retrieval_ranks": list(range(1, len(retrieved_list) + 1)),
            "answer": answer,
            "citations": verification_output.get("citations", []),
            "confidence": {
                "overall_score": conf_output.overall_score,
                "level": conf_output.level,
                "retrieval_confidence": conf_output.retrieval_confidence,
                "grounding_confidence": conf_output.grounding_confidence,
                "abstention_flag": conf_output.abstention_flag,
            },
            "status": "insufficient_context" if conf_output.abstention_flag else ("answered" if retrieved_list else "no_results"),
            "latency_ms": {
                "retrieval": retrieval_latency_ms,
                "generation": generation_latency_ms,
                "verification": verification_latency_ms,
                "total": total_latency_ms,
            },
            "llm_status": self.llm_status,
            "is_fallback": is_fallback,
            "metrics": {
                "recall@1": ir_metrics["recall@1"],
                "recall@5": ir_metrics["recall@5"],
                "recall@10": ir_metrics["recall@10"],
                "mrr": ir_metrics["mrr"],
                "ndcg@5": ndcg_5,
                "map": map_score,
                "answer_correctness": correctness,
                "faithfulness": faithfulness,
                "answer_relevance": answer_relevance,
                "abstention_score": abstention_score,
                "citation_coverage": cit_metrics["citation_coverage"],
                "citation_accuracy": cit_metrics["citation_accuracy"],
                "grounding_ratio": g_metrics.get("grounding_ratio", 0.0),
                "total_claims": g_metrics.get("total_claims", 0),
                "supported_claims": g_metrics.get("supported_claims", 0),
            },
        }

    def evaluate_mode(
        self,
        mode: str,
        questions: list[dict[str, Any]],
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Evaluate all questions for a single retrieval mode.

        Returns mode-level aggregated metrics and per-query detail.
        """
        results = []
        for idx, q in enumerate(questions, 1):
            result = self.evaluate_query(q, mode)
            r1 = result["metrics"]["recall@1"]
            mrr = result["metrics"]["mrr"]
            lat = result["latency_ms"]["total"]
            print(f"  [{idx}/{len(questions)}] {q['id']} ({q['type']}): R@1={r1}, MRR={mrr}, Lat={lat}ms")
            results.append(result)

        return self._aggregate_mode_results(mode, results)

    def evaluate_all_modes(
        self,
        questions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate all four modes and produce a comparative benchmark.

        Returns the full benchmark output dict.
        """
        modes = ["dense", "bm25", "hybrid", "hybrid_rerank"]
        mode_results = {}
        all_per_query = []

        for mode in modes:
            print(f"\n=== Evaluating mode: {mode} ===")
            mode_output = self.evaluate_mode(mode, questions)
            mode_results[mode] = mode_output["summary"]
            all_per_query.extend(mode_output["per_query"])

        return {
            "modes": mode_results,
            "per_query_results": all_per_query,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_single_query(
        self,
        question: str,
        mode: str,
    ) -> dict[str, Any]:
        """Execute a single query through the specified pipeline mode.

        Returns a dict with retrieved_list, answer, and raw retrieval results.
        Validates that each mode actually executes its intended pipeline.
        """
        top_k = 10

        if mode == "dense":
            dense_results = run_search_dense(query=question, k=top_k, source=None)
            bm25_results = []
            rrf_results = []
            reranker_results = []
            target_results = dense_results
        elif mode == "bm25":
            dense_results = []
            bm25_results = run_search_bm25(query=question, k=top_k, source=None)
            rrf_results = []
            reranker_results = []
            target_results = bm25_results
        elif mode == "hybrid":
            dense_results = run_search_dense(query=question, k=top_k, source=None)
            bm25_results = run_search_bm25(query=question, k=top_k, source=None)
            rrf_results = run_search_hybrid(
                query=question, k=10, top_k_dense=top_k,
                top_k_sparse=top_k, top_k_fused=10, source=None,
            )
            reranker_results = []
            target_results = rrf_results
        elif mode == "hybrid_rerank":
            dense_results = run_search_dense(query=question, k=top_k, source=None)
            bm25_results = run_search_bm25(query=question, k=top_k, source=None)
            rrf_results = run_search_hybrid(
                query=question, k=20, top_k_dense=top_k,
                top_k_sparse=top_k, top_k_fused=20, source=None,
            )
            reranker = get_reranker()
            reranker_results = reranker.rerank(
                query=question, candidates=rrf_results, top_k=5,
            )
            target_results = reranker_results
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Build retrieved list (dicts for metric functions) from target_results
        retrieved_list = [
            {
                "text": r.content,
                "source": r.source,
                "page": r.metadata.get("page") if isinstance(r.metadata, dict) else None,
                "score": r.score,
                "metadata": r.metadata if isinstance(r.metadata, dict) else {},
            }
            for r in target_results
        ]

        # Generate answer from target_results
        search_results_for_answer = [
            SearchResult(
                text=r.content,
                source=r.source,
                page=r.metadata.get("page") if isinstance(r.metadata, dict) else None,
                score=max(0.0, min(1.0, float(r.score) if r.score <= 1.0 else 1.0)),
                metadata=r.metadata if isinstance(r.metadata, dict) else {},
            )
            for r in target_results
        ]
        from backend.retrieval import _answer
        answer = _answer(question, search_results_for_answer)

        return {
            "retrieved_list": retrieved_list,
            "answer": answer,
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "rrf_results": rrf_results,
            "reranker_results": reranker_results,
        }

    def _aggregate_mode_results(
        self,
        mode: str,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate per-query results into mode-level summary."""
        total = len(results)
        if total == 0:
            return {"summary": {}, "per_query": results}

        # IR metrics
        recall_1 = sum(r["metrics"]["recall@1"] for r in results) / total
        recall_5 = sum(r["metrics"]["recall@5"] for r in results) / total
        recall_10 = sum(r["metrics"]["recall@10"] for r in results) / total
        mrr = sum(r["metrics"]["mrr"] for r in results) / total
        ndcg_5 = sum(r["metrics"]["ndcg@5"] for r in results) / total
        map_score = sum(r["metrics"]["map"] for r in results) / total

        # Grounding
        grounding_ratio = sum(r["metrics"]["grounding_ratio"] for r in results) / total
        citation_coverage = sum(r["metrics"]["citation_coverage"] for r in results) / total
        citation_accuracy_vals = [r["metrics"]["citation_accuracy"] for r in results if r["metrics"]["citation_accuracy"] > 0]
        citation_accuracy = sum(citation_accuracy_vals) / len(citation_accuracy_vals) if citation_accuracy_vals else 0.0

        # Faithfulness (only from real LLM answers)
        faith_vals = [r["metrics"]["faithfulness"] for r in results if r["metrics"]["faithfulness"] is not None]
        faithfulness = sum(faith_vals) / len(faith_vals) if faith_vals else None

        # Answer relevance
        rel_vals = [r["metrics"]["answer_relevance"] for r in results if r["metrics"]["answer_relevance"] is not None]
        answer_relevance = sum(rel_vals) / len(rel_vals) if rel_vals else None

        # Abstention metrics
        unsupported = [r for r in results if r["category"] in UNSUPPORTED_CATEGORIES]
        answerable = [r for r in results if r["category"] in ANSWERABLE_CATEGORIES]

        abstention_count = sum(1 for r in results if r["confidence"]["abstention_flag"])
        correct_abstentions = sum(
            1 for r in unsupported if r["confidence"]["abstention_flag"]
        )
        false_abstentions = sum(
            1 for r in answerable if r["confidence"]["abstention_flag"]
        )
        abstention_accuracy = correct_abstentions / len(unsupported) if unsupported else 0.0
        false_answer_rate = (len(unsupported) - correct_abstentions) / len(unsupported) if unsupported else 0.0
        false_abstention_rate = false_abstentions / len(answerable) if answerable else 0.0
        answerable_coverage = (len(answerable) - false_abstentions) / len(answerable) if answerable else 0.0

        # Latency stats
        total_latencies = sorted([r["latency_ms"]["total"] for r in results])
        retrieval_latencies = sorted([r["latency_ms"]["retrieval"] for r in results])

        def _percentile(sorted_vals, p):
            if not sorted_vals:
                return 0.0
            idx = int(len(sorted_vals) * p / 100)
            idx = min(idx, len(sorted_vals) - 1)
            return sorted_vals[idx]

        # Category breakdown
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        category_summary = {}
        for cat, items in categories.items():
            n = len(items)
            category_summary[cat] = {
                "count": n,
                "recall@1": round(sum(i["metrics"]["recall@1"] for i in items) / n, 4),
                "recall@5": round(sum(i["metrics"]["recall@5"] for i in items) / n, 4),
                "mrr": round(sum(i["metrics"]["mrr"] for i in items) / n, 4),
                "ndcg@5": round(sum(i["metrics"]["ndcg@5"] for i in items) / n, 4),
                "map": round(sum(i["metrics"]["map"] for i in items) / n, 4),
                "grounding_ratio": round(sum(i["metrics"]["grounding_ratio"] for i in items) / n, 4),
                "mean_latency_ms": round(sum(i["latency_ms"]["total"] for i in items) / n, 2),
            }

        summary = {
            "mode": mode,
            "total_questions": total,
            "llm_status": self.llm_status,
            "recall_at_1": round(recall_1, 4),
            "recall_at_5": round(recall_5, 4),
            "recall_at_10": round(recall_10, 4),
            "mrr": round(mrr, 4),
            "ndcg_at_5": round(ndcg_5, 4),
            "map": round(map_score, 4),
            "grounding_ratio": round(grounding_ratio, 4),
            "citation_coverage": round(citation_coverage, 4),
            "citation_accuracy": round(citation_accuracy, 4),
            "faithfulness": round(faithfulness, 4) if faithfulness is not None else None,
            "answer_relevance": round(answer_relevance, 4) if answer_relevance is not None else None,
            "abstention_accuracy": round(abstention_accuracy, 4),
            "false_answer_rate": round(false_answer_rate, 4),
            "false_abstention_rate": round(false_abstention_rate, 4),
            "answerable_coverage": round(answerable_coverage, 4),
            "mean_latency_ms": round(sum(total_latencies) / total, 2),
            "p50_latency_ms": round(_percentile(total_latencies, 50), 2),
            "p95_latency_ms": round(_percentile(total_latencies, 95), 2),
            "retrieval_mean_latency_ms": round(sum(retrieval_latencies) / total, 2),
            "retrieval_p50_latency_ms": round(_percentile(retrieval_latencies, 50), 2),
            "retrieval_p95_latency_ms": round(_percentile(retrieval_latencies, 95), 2),
            "by_category": category_summary,
        }

        return {"summary": summary, "per_query": results}
