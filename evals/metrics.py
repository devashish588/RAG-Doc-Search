import math
import re
from typing import Any


def calculate_retrieval_metrics(
    retrieved_results: list[dict[str, Any]],
    expected_sources: list[dict[str, str]]
) -> dict[str, float]:
    """Calculate Recall@1, Recall@5, Recall@10, and MRR for a single query."""
    if not expected_sources:
        return {"recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0, "mrr": 1.0}

    expected_docs = {s["document"].lower() for s in expected_sources if "document" in s}

    ranks = []
    for rank_idx, res in enumerate(retrieved_results, 1):
        source = res.get("source", "").lower()
        # Match if expected document filename matches retrieved chunk source basename
        if any(exp_doc in source or source in exp_doc for exp_doc in expected_docs):
            ranks.append(rank_idx)

    if not ranks:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}

    first_rank = min(ranks)
    return {
        "recall@1": 1.0 if first_rank <= 1 else 0.0,
        "recall@5": 1.0 if first_rank <= 5 else 0.0,
        "recall@10": 1.0 if first_rank <= 10 else 0.0,
        "mrr": round(1.0 / first_rank, 4),
    }


def tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase words."""
    words = re.findall(r"\b\w+\b", text.lower())
    stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "is", "are", "was", "were", "be", "with"}
    return {w for w in words if w not in stop_words and len(w) > 1}


def calculate_text_similarity(actual: str, expected: str) -> float:
    """Calculate token Jaccard similarity score between actual and expected answers."""
    actual_tokens = tokenize(actual)
    expected_tokens = tokenize(expected)

    if not expected_tokens:
        return 1.0 if not actual_tokens else 0.0

    intersection = actual_tokens.intersection(expected_tokens)
    union = actual_tokens.union(expected_tokens)
    return round(len(intersection) / len(union), 4) if union else 0.0


def calculate_abstention_score(question_type: str, actual_answer: str) -> float:
    """Check if the system abstains correctly on unanswerable or ambiguous questions."""
    actual_lower = actual_answer.lower()
    abstention_keywords = [
        "cannot answer", "no relevant context", "unsupported",
        "not mentioned", "not provided", "not documented",
        "multiple", "depends", "two default", "two ports", "dynamically"
    ]

    if question_type == "unanswerable":
        return 1.0 if any(kw in actual_lower for kw in abstention_keywords) else 0.0
    elif question_type == "ambiguous":
        return 1.0 if any(kw in actual_lower for kw in abstention_keywords) else 0.5
    return 1.0


def calculate_citation_metrics(actual_answer: str, retrieved_results: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate citation coverage and citation accuracy."""
    if not actual_answer or not retrieved_results:
        return {"citation_coverage": 0.0, "citation_accuracy": 0.0}

    # Match numeric citations like [1], [2] or document name citations
    has_citations = bool(re.search(r"\[\d+\]", actual_answer)) or any(
        res.get("source", "").lower() in actual_answer.lower() for res in retrieved_results
    )

    coverage = 1.0 if has_citations else 0.0
    accuracy = 1.0 if has_citations else 0.0

    return {
        "citation_coverage": coverage,
        "citation_accuracy": accuracy,
    }


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query evaluation metrics into overall summary statistics."""
    total = len(results)
    if total == 0:
        return {}

    recall_1 = sum(r["metrics"]["recall@1"] for r in results) / total
    recall_5 = sum(r["metrics"]["recall@5"] for r in results) / total
    recall_10 = sum(r["metrics"]["recall@10"] for r in results) / total
    mrr = sum(r["metrics"]["mrr"] for r in results) / total

    correctness = sum(r["metrics"]["answer_correctness"] for r in results) / total
    faithfulness = sum(r["metrics"]["faithfulness"] for r in results) / total
    citation_cov = sum(r["metrics"]["citation_coverage"] for r in results) / total
    citation_acc = sum(r["metrics"]["citation_accuracy"] for r in results) / total
    abstention_acc = sum(r["metrics"]["abstention_score"] for r in results if r["type"] in {"unanswerable", "ambiguous"})
    unans_ambig_count = sum(1 for r in results if r["type"] in {"unanswerable", "ambiguous"})
    abstention_rate = (abstention_acc / unans_ambig_count) if unans_ambig_count > 0 else 1.0

    latencies = [r["latency_ms"] for r in results]
    latencies.sort()

    mean_lat = sum(latencies) / total
    p50_lat = latencies[int(total * 0.50)]
    p95_lat = latencies[int(total * 0.95)] if total >= 20 else latencies[-1]

    # Category breakdown
    categories = {}
    for r in results:
        qtype = r["type"]
        if qtype not in categories:
            categories[qtype] = []
        categories[qtype].append(r)

    category_summary = {}
    for cat, items in categories.items():
        cat_total = len(items)
        category_summary[cat] = {
            "count": cat_total,
            "recall@1": round(sum(i["metrics"]["recall@1"] for i in items) / cat_total, 4),
            "recall@5": round(sum(i["metrics"]["recall@5"] for i in items) / cat_total, 4),
            "mrr": round(sum(i["metrics"]["mrr"] for i in items) / cat_total, 4),
            "answer_correctness": round(sum(i["metrics"]["answer_correctness"] for i in items) / cat_total, 4),
            "mean_latency_ms": round(sum(i["latency_ms"] for i in items) / cat_total, 2),
        }

    return {
        "overall": {
            "recall@1": round(recall_1, 4),
            "recall@5": round(recall_5, 4),
            "recall@10": round(recall_10, 4),
            "mrr": round(mrr, 4),
            "answer_correctness": round(correctness, 4),
            "faithfulness": round(faithfulness, 4),
            "citation_coverage": round(citation_cov, 4),
            "citation_accuracy": round(citation_acc, 4),
            "abstention_accuracy": round(abstention_rate, 4),
            "latency_ms": {
                "mean": round(mean_lat, 2),
                "p50": round(p50_lat, 2),
                "p95": round(p95_lat, 2),
            }
        },
        "by_category": category_summary
    }
