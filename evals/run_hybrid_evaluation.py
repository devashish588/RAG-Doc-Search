"""Hybrid RRF evaluation for Phase 5.

Evaluates hybrid RRF retrieval independently using the frozen golden dataset.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
EVALS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVALS_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.ingestion import ingest_document, register_document
from backend.retrieval_hybrid import get_hybrid_retriever
from backend.retrieval import run_search_dense, run_search_bm25, run_search_hybrid
from evals.metrics import (
    aggregate_metrics,
    calculate_abstention_score,
    calculate_citation_metrics,
    calculate_retrieval_metrics,
    calculate_text_similarity,
)


EVALS_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVALS_DIR / "corpus"
GOLDEN_DATASET_PATH = EVALS_DIR / "golden_dataset.json"
RESULTS_DIR = EVALS_DIR / "results"
HYBRID_RESULTS_DIR = RESULTS_DIR / "hybrid"
HYBRID_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HYBRID_RESULTS_PATH = HYBRID_RESULTS_DIR / "hybrid_results.json"
HYBRID_REPORT_PATH = EVALS_DIR / "hybrid_report.md"
ROOT_HYBRID_REPORT_PATH = EVALS_DIR.parent / "hybrid_report.md"


def setup_corpus() -> list[str]:
    """Ingest fixed evaluation corpus into the vector store and BM25 index."""
    print("--- Ingesting Fixed Evaluation Corpus ---")
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

    print(f"Corpus ingestion complete. Total files ingested: {len(doc_ids)}")
    return doc_ids


def evaluate_hybrid(
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60,
    top_k_dense: int = 10,
    top_k_sparse: int = 10,
    top_k_fused: int = 10,
    experiment_id: str = "",
) -> dict[str, Any]:
    """Run hybrid RRF-only evaluation using run_search_hybrid."""
    print(f"\n--- Running Hybrid RRF Evaluation ({experiment_id}) ---")
    print(f"  Dense weight: {dense_weight}, Sparse weight: {sparse_weight}")
    print(f"  RRF k: {rrf_k}")
    print(f"  Top-k dense: {top_k_dense}, Top-k sparse: {top_k_sparse}, Top-k fused: {top_k_fused}")

    retriever = None  # will be initialized on first use

    results_detail = []

    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["records"]
    print(f"Loaded {len(questions)} evaluation questions.")

    print("\n--- Running Hybrid RRF Search Queries & Computing Metrics ---")
    for idx, item in enumerate(questions, 1):
        q_id = item["id"]
        query = item["question"]
        q_type = item["type"]
        expected_ans = item["expected_answer"]
        expected_srcs = item.get("expected_sources", [])

        # Use hybrid retriever directly
        start = time.perf_counter()
        hybrid_results = run_search_hybrid(
            query=query,
            k=top_k_fused,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            top_k_fused=top_k_fused,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Convert hybrid RetrievalResult to format expected by metrics
        retrieved_list = [
            {
                "text": r.content,
                "source": r.source,
                "page": r.metadata.get("page"),
                "score": r.score,
            }
            for r in hybrid_results
        ]

        ret_metrics = calculate_retrieval_metrics(retrieved_list, expected_srcs)

        # For generation metrics, we can't easily use hybrid results with LLM
        # since run_search_hybrid doesn't generate answers
        # We'll mark these as unavailable
        correctness = 0.0  # N/A for retrieval-only evaluation
        abstention = 0.0  # N/A
        cit_cov = 0.0
        cit_acc = 0.0
        faithfulness = 0.0

        query_record = {
            "id": q_id,
            "question": query,
            "type": q_type,
            "expected_answer": expected_ans,
            "actual_answer": "[Hybrid RRF retrieval-only evaluation - no answer generated]",
            "retrieved_results": retrieved_list,
            "latency_ms": latency_ms,
            "metrics": {
                "recall@1": ret_metrics["recall@1"],
                "recall@5": ret_metrics["recall@5"],
                "recall@10": ret_metrics["recall@10"],
                "mrr": ret_metrics["mrr"],
                "answer_correctness": correctness,
                "faithfulness": faithfulness,
                "abstention_score": abstention,
                "citation_coverage": cit_cov,
                "citation_accuracy": cit_acc,
            }
        }
        results_detail.append(query_record)
        print(f"[{idx}/{len(questions)}] {q_id} ({q_type}): Recall@1={ret_metrics['recall@1']:.4f}, MRR={ret_metrics['mrr']:.4f}, Latency={latency_ms}ms")

    summary_metrics = aggregate_metrics(results_detail)

    output = {
        "metadata": {
            "evaluation_phase": "Phase 5 - Hybrid RRF Retrieval Evaluation",
            "git_tag": "v1.5-hybrid",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chunk_size": 1200,
            "chunk_overlap": 200,
            "embedding_backend": os.getenv("EMBEDDING_BACKEND", "auto"),
            "total_questions": len(questions),
            "experiment_id": experiment_id,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "rrf_k": rrf_k,
            "top_k_dense": top_k_dense,
            "top_k_sparse": top_k_sparse,
            "top_k_fused": top_k_fused,
        },
        "summary": summary_metrics,
        "details": results_detail,
    }

    # Save results
    exp_path = HYBRID_RESULTS_DIR / f"hybrid_{experiment_id}.json"
    with exp_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nHybrid Evaluation complete. Full results written to {exp_path}")
    return output


def run_weight_experiments():
    """Run RRF weight experiments."""
    print("\n" + "="*60)
    print("RUNNING RRF WEIGHT EXPERIMENTS")
    print("="*60)

    weight_configs = [
        (1.0, 0.0, "dense_only"),
        (0.9, 0.1, "w0.9_0.1"),
        (0.7, 0.3, "w0.7_0.3"),
        (0.5, 0.5, "w0.5_0.5"),
        (0.3, 0.7, "w0.3_0.7"),
        (0.1, 0.9, "w0.1_0.9"),
        (0.0, 1.0, "sparse_only"),
    ]

    results = {}
    for dw, sw, exp_id in weight_configs:
        result = evaluate_hybrid(
            dense_weight=dw,
            sparse_weight=sw,
            rrf_k=60,
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=10,
            experiment_id=exp_id,
        )
        results[exp_id] = result["summary"]["overall"]

    # Save comparison
    comparison_path = HYBRID_RESULTS_DIR / "rrf_weight_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nWeight comparison saved to {comparison_path}")
    return results


def run_k_experiments():
    """Run RRF k experiments."""
    print("\n" + "="*60)
    print("RUNNING RRF K EXPERIMENTS")
    print("="*60)

    k_values = [10, 30, 60, 100]

    results = {}
    for k in k_values:
        exp_id = f"k{k}"
        result = evaluate_hybrid(
            dense_weight=0.7,
            sparse_weight=0.3,
            rrf_k=k,
            top_k_dense=10,
            top_k_sparse=10,
            top_k_fused=10,
            experiment_id=exp_id,
        )
        results[exp_id] = result["summary"]["overall"]

    comparison_path = HYBRID_RESULTS_DIR / "rrf_k_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nK comparison saved to {comparison_path}")
    return results


def run_candidate_depth_experiments():
    """Run candidate depth experiments."""
    print("\n" + "="*60)
    print("RUNNING CANDIDATE DEPTH EXPERIMENTS")
    print("="*60)

    configs = [
        (5, 5, "d5_s5"),
        (10, 10, "d10_s10"),
        (20, 20, "d20_s20"),
        (10, 20, "d10_s20"),
        (20, 10, "d20_s10"),
    ]

    results = {}
    for d, s, exp_id in configs:
        result = evaluate_hybrid(
            dense_weight=0.7,
            sparse_weight=0.3,
            rrf_k=60,
            top_k_dense=d,
            top_k_sparse=s,
            top_k_fused=10,
            experiment_id=exp_id,
        )
        results[exp_id] = result["summary"]["overall"]

    comparison_path = HYBRID_RESULTS_DIR / "candidate_depth_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nCandidate depth comparison saved to {comparison_path}")
    return results


def run_full_evaluation():
    """Run complete Phase 5 evaluation suite."""
    print("\n" + "="*60)
    print("PHASE 5 HYBRID RRF FULL EVALUATION")
    print("="*60)

    # Setup corpus
    setup_corpus()

    # Run baseline hybrid (default config)
    print("\n--- Running Baseline Hybrid (Default Config) ---")
    baseline = evaluate_hybrid(
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        top_k_dense=10,
        top_k_sparse=10,
        top_k_fused=10,
        experiment_id="baseline",
    )

    # Weight experiments
    weight_results = run_weight_experiments()

    # K experiments
    k_results = run_k_experiments()

    # Candidate depth experiments
    depth_results = run_candidate_depth_experiments()

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nBaseline (0.7/0.3, k=60): Recall@1={baseline['summary']['overall']['recall@1']:.4f}, MRR={baseline['summary']['overall']['mrr']:.4f}")

    print("\nWeight Experiments:")
    for exp_id, metrics in sorted(baseline.items()):
        if exp_id.startswith("w"):
            print(f"  {exp_id}: Recall@1={metrics['recall@1']:.4f}, MRR={metrics['mrr']:.4f}")

    print("\nK Experiments:")
    for k, metrics in sorted(k_results.items()):
        print(f"  {k}: Recall@1={metrics['recall@1']:.4f}, MRR={metrics['mrr']:.4f}")

    print("\nCandidate Depth Experiments:")
    for exp_id, metrics in sorted(depth_results.items()):
        print(f"  {exp_id}: Recall@1={metrics['recall@1']:.4f}, MRR={metrics['mrr']:.4f}")

    return {
        "baseline": baseline["summary"]["overall"],
        "weight_experiments": weight_results,
        "k_experiments": k_results,
        "depth_experiments": depth_results,
    }


if __name__ == "__main__":
    run_full_evaluation()