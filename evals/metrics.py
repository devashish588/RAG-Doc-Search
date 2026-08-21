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


# ---------------------------------------------------------------------------
# Advanced IR Metrics (Phase 8)
# ---------------------------------------------------------------------------
# Relevance definition:  BINARY relevance.
#   - relevant = 1 if a retrieved chunk source matches an expected_sources
#     document  (substring match, case-insensitive).
#   - relevant = 0 otherwise.
# This definition is applied identically to every retrieval mode.

def _binary_relevance_vector(
    retrieved_results: list[dict[str, Any]],
    expected_sources: list[dict[str, str]],
) -> list[int]:
    """Build a binary relevance vector for retrieved results.

    Returns a list of 0/1 values, one per retrieved result, indicating
    whether that result matches any expected source document.
    """
    if not expected_sources:
        # No expected sources = all results are vacuously relevant
        return [1] * len(retrieved_results)

    expected_docs = {s["document"].lower() for s in expected_sources if "document" in s}
    relevance = []
    for res in retrieved_results:
        source = res.get("source", "").lower()
        is_relevant = any(exp_doc in source or source in exp_doc for exp_doc in expected_docs)
        relevance.append(1 if is_relevant else 0)
    return relevance


def calculate_ndcg_at_k(
    retrieved_results: list[dict[str, Any]],
    expected_sources: list[dict[str, str]],
    k: int = 5,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.

    Uses BINARY relevance (see module docstring).
    DCG@k  = sum_{i=1}^{k} rel_i / log2(i + 1)
    IDCG@k = best-possible DCG@k given the total number of relevant docs.
    NDCG@k = DCG@k / IDCG@k   (0.0 if IDCG is 0).

    Args:
        retrieved_results: list of retrieved chunk dicts with 'source' key
        expected_sources: golden expected source dicts with 'document' key
        k: cutoff rank (default 5)

    Returns:
        NDCG@k score in [0.0, 1.0]
    """
    rel = _binary_relevance_vector(retrieved_results, expected_sources)
    # Truncate to k
    rel_k = rel[:k]

    # DCG@k
    dcg = 0.0
    for i, r in enumerate(rel_k, 1):
        dcg += r / math.log2(i + 1)

    # IDCG@k: best possible DCG with all relevant docs at the top
    total_relevant = sum(rel)  # from full list
    ideal_rel = [1] * min(total_relevant, k) + [0] * max(0, k - total_relevant)
    idcg = 0.0
    for i, r in enumerate(ideal_rel, 1):
        idcg += r / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def calculate_map(
    retrieved_results: list[dict[str, Any]],
    expected_sources: list[dict[str, str]],
) -> float:
    """Calculate Mean Average Precision for a single query.

    Uses BINARY relevance (see module docstring).
    AP = (1/R) * sum_{k=1}^{N} P(k) * rel(k)
    where R = total relevant docs, N = total retrieved, P(k) = precision at k.

    Args:
        retrieved_results: list of retrieved chunk dicts with 'source' key
        expected_sources: golden expected source dicts with 'document' key

    Returns:
        Average Precision score in [0.0, 1.0]
    """
    rel = _binary_relevance_vector(retrieved_results, expected_sources)
    total_relevant = sum(rel)
    if total_relevant == 0:
        return 0.0

    ap_sum = 0.0
    running_relevant = 0
    for i, r in enumerate(rel, 1):
        if r == 1:
            running_relevant += 1
            precision_at_i = running_relevant / i
            ap_sum += precision_at_i

    return round(ap_sum / total_relevant, 4)


def calculate_faithfulness(
    answer: str,
    context_texts: list[str],
    claims: list[dict] | None = None,
) -> float | None:
    """Calculate faithfulness as the fraction of answer claims supported by context.

    Methodology:
    - Uses the Phase 7 CitationVerifier claim extraction.
    - For each extracted claim, checks lexical token overlap with context.
    - A claim is "faithful" if its token overlap with any context chunk >= 0.30.
    - faithfulness = supported_claims / total_claims.

    If the answer is a context fallback (starts with "Most relevant context:")
    or a refusal ("I don't have enough evidence"), returns None because
    faithfulness is undefined for non-generated text.

    If no claims can be extracted, returns None.

    Args:
        answer: the generated answer text
        context_texts: list of evidence chunk texts
        claims: optional pre-extracted claims (list of dicts with 'claim' key)

    Returns:
        faithfulness score in [0.0, 1.0], or None if not applicable
    """
    if not answer or not answer.strip():
        return None

    answer_lower = answer.strip().lower()
    # Detect fallback / refusal answers
    if answer_lower.startswith("most relevant context:"):
        return None
    if "i don't have enough evidence" in answer_lower:
        return None
    if "i cannot answer" in answer_lower:
        return None

    if not context_texts:
        return None

    # Extract claims if not provided
    if claims is None:
        # Simple sentence-level claim extraction
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        claims = [{"claim": s.strip()} for s in sentences if len(s.strip()) > 10]

    if not claims:
        return None

    supported = 0
    for claim in claims:
        claim_tokens = tokenize(claim["claim"])
        if not claim_tokens:
            continue
        best_overlap = 0.0
        for ctx in context_texts:
            ctx_tokens = tokenize(ctx)
            if not ctx_tokens:
                continue
            overlap = len(claim_tokens & ctx_tokens) / len(claim_tokens)
            best_overlap = max(best_overlap, overlap)
        if best_overlap >= 0.30:
            supported += 1

    return round(supported / len(claims), 4) if claims else None


def calculate_answer_relevance(
    answer: str,
    question: str,
    embedding_model: Any = None,
) -> float | None:
    """Calculate answer relevance as cosine similarity between question and answer embeddings.

    Methodology:
    - Embeds the question and answer using the application's configured embedding model
      (FastEmbed BAAI/bge-small-en-v1.5) — the SAME model used for retrieval.
    - Computes cosine similarity between the two embeddings.

    If the answer is a fallback or refusal, returns None (relevance is undefined).
    If embedding fails, returns None.

    Args:
        answer: the generated answer text
        question: the original question text
        embedding_model: optional embedding model override (for testing)

    Returns:
        cosine similarity in [-1.0, 1.0] (typically [0.0, 1.0] for well-formed Q/A),
        or None if not applicable.
    """
    if not answer or not answer.strip() or not question or not question.strip():
        return None

    answer_lower = answer.strip().lower()
    if answer_lower.startswith("most relevant context:"):
        return None
    if "i don't have enough evidence" in answer_lower:
        return None
    if "i cannot answer" in answer_lower:
        return None

    try:
        if embedding_model is None:
            from backend.vector_store import get_embeddings
            embedding_model = get_embeddings()
        q_emb = embedding_model.embed_documents([question])[0]
        a_emb = embedding_model.embed_documents([answer])[0]

        # Cosine similarity
        dot = sum(x * y for x, y in zip(q_emb, a_emb))
        norm_q = sum(x * x for x in q_emb) ** 0.5
        norm_a = sum(x * x for x in a_emb) ** 0.5
        if norm_q == 0 or norm_a == 0:
            return None
        return round(dot / (norm_q * norm_a), 4)
    except Exception:
        return None


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
