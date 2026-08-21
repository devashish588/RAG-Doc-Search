"""Phase 6 evaluation: Cross-Encoder reranking of Hybrid RRF candidates.

Runs on the SAME frozen golden dataset as prior phases, with consistent
retrieval-only timing (warm-up + measurement), and adds the reranker stage.

Experiments (all reuse the frozen eval corpus + golden dataset):
  A. Baseline 4-way comparison:
       Dense | BM25 | Hybrid RRF | Hybrid RRF + Reranker
  B. Candidate-pool depth (top_k_final=5): candidate_k in {5, 10, 20}
  C. Final top-k (candidate_k=20):        top_k_final in {3, 5, 10}

Reranker consumes the RRF fused pool (candidate_k) and returns top_k_final.
Provenance (RRF top-1 vs Reranker top-1), rank-promotion, latency (RRF /
reranker / total, cold vs warm), and memory are all recorded.

Generation metrics are marked N/A (OpenRouter unavailable in this env).
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = EVALS_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.ingestion import ingest_document, register_document
from backend.retrieval import run_search_dense, run_search_bm25
from backend.retrieval_hybrid import get_hybrid_retriever
from backend.reranker import CrossEncoderReranker
from backend.settings import RERANKER_MODEL
import os as _os

EVAL_RERANKER_MODEL = _os.environ.get("RERANKER_MODEL", RERANKER_MODEL)
from evals.metrics import calculate_retrieval_metrics

CORPUS_DIR = EVALS_DIR / "corpus"
GOLDEN_DATASET_PATH = EVALS_DIR / "golden_dataset.json"
RESULTS_DIR = EVALS_DIR / "results"
RERANKER_RESULTS_DIR = RESULTS_DIR / "reranker"
RERANKER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def setup_corpus() -> None:
    print("--- Ingesting Fixed Evaluation Corpus ---")
    for file_path in sorted(CORPUS_DIR.glob("*")):
        if file_path.is_dir() or file_path.name.startswith("."):
            continue
        doc_id = f"eval_{file_path.stem}"
        register_document(document_id=doc_id, filename=file_path.name, stored_path=file_path)
        ingest_document(document_id=doc_id, stored_path=file_path, filename=file_path.name)


def build_eval_bm25_index() -> int:
    """Build a FRESH BM25 index from ONLY the frozen eval corpus (reproducibility)."""
    from backend.bm25_index import reset_bm25_index, get_bm25_index
    from backend.loaders import get_loader
    from backend.chunking import get_chunker
    from backend.normalizer import get_normalizer
    from backend.settings import CHUNKING_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP
    from langchain_core.documents import Document

    reset_bm25_index()
    index = get_bm25_index()
    all_chunks = []
    for file_path in sorted(CORPUS_DIR.glob("*")):
        if file_path.is_dir() or file_path.name.startswith("."):
            continue
        loader = get_loader(file_path)
        docs = loader.load(file_path)
        normalizer = get_normalizer("standard")
        norm = [Document(page_content=normalizer.normalize(d.page_content), metadata=dict(d.metadata)) for d in docs]
        chunker = get_chunker(CHUNKING_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunker.chunk(norm, f"eval_{file_path.stem}", file_path.name))
    index.build(all_chunks)
    index.save()
    return len(all_chunks)


def to_metric_format(results: list) -> list[dict[str, Any]]:
    return [
        {"text": r.content, "source": r.source, "page": r.metadata.get("page"), "score": r.score}
        for r in results
    ]


# ---------------------------------------------------------------------------
# Memory sampling
# ---------------------------------------------------------------------------
try:
    import psutil
    _PROC = psutil.Process()
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False


def rss_mb() -> float:
    if not _HAVE_PSUTIL:
        return 0.0
    return _PROC.memory_info().rss / 1e6


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def run_evaluation() -> dict[str, Any]:
    setup_corpus()
    n_bm25 = build_eval_bm25_index()
    print(f"BM25 index built: {n_bm25} chunks")

    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    questions = dataset["records"]
    print(f"Loaded {len(questions)} golden questions.")

    hybrid = get_hybrid_retriever()

    # ---- Warm-up (no measurement) ----
    print("\n--- Warm-up pass ---")
    for item in questions:
        q = item["question"]
        run_search_dense(q, k=10)
        run_search_bm25(q, k=10)
        hybrid.search(q, top_k_dense=10, top_k_sparse=10, top_k_fused=20)
    # Warm the reranker model once (cold load happens here, excluded from timing)
    warm_reranker = CrossEncoderReranker(model_name=EVAL_RERANKER_MODEL, candidate_k=20, top_k=10)
    for item in questions[:3]:
        warm_reranker.rerank(item["question"], hybrid.search(item["question"], top_k_dense=10, top_k_sparse=10, top_k_fused=20), top_k=10)
    cold_init_ms = warm_reranker.cold_init_ms
    print(f"Reranker cold init: {cold_init_ms} ms")

    # Memory: baseline (post-ingest, post-warm) and after reranker load
    mem_baseline = rss_mb()
    mem_after_model = rss_mb()  # already loaded during warm-up
    mem_peak = mem_after_model

    details = []
    # Latency accumulators (baseline reranker config: cand=20, top=10)
    dense_lat, bm25_lat, rrf_lat, hyb_lat, rer_lat = [], [], [], [], []
    reranker_statuses = []

    for idx, item in enumerate(questions, 1):
        q_id = item["id"]; query = item["question"]; q_type = item["type"]
        expected = item.get("expected_sources", [])

        t0 = time.perf_counter(); dense_r = run_search_dense(query, k=10); d_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); bm25_r = run_search_bm25(query, k=10); b_ms = (time.perf_counter() - t0) * 1000
        hyb_r, hyb_timing = hybrid.search_with_timing(query, top_k_dense=10, top_k_sparse=10, top_k_fused=20)
        rrf_ms = hyb_timing["rrf_ms"]; hyb_ms = hyb_timing["total_ms"]

        # Baseline reranker (candidate_k=20, top_k_final=10) - reuse warmed instance
        rer_r, rer_info = warm_reranker.rerank_with_timing(query, hyb_r, top_k=10)
        rer_ms = rer_info["reranker_ms"]
        reranker_statuses.append(rer_info["status"])
        if _HAVE_PSUTIL:
            mem_peak = max(mem_peak, rss_mb())

        dense_lat.append(d_ms); bm25_lat.append(b_ms); rrf_lat.append(rrf_ms)
        hyb_lat.append(hyb_ms); rer_lat.append(rer_ms)

        d_m = calculate_retrieval_metrics(to_metric_format(dense_r), expected)
        b_m = calculate_retrieval_metrics(to_metric_format(bm25_r), expected)
        h_m = calculate_retrieval_metrics(to_metric_format(hyb_r), expected)
        r_m = calculate_retrieval_metrics(to_metric_format(rer_r), expected)

        # Provenance
        d_top = dense_r[0].chunk_id if dense_r else None
        b_top = bm25_r[0].chunk_id if bm25_r else None
        h_top = hyb_r[0].chunk_id if hyb_r else None
        r_top = rer_r[0].chunk_id if rer_r else None

        def _hit(cid):
            if not cid or not expected:
                return False
            exp_docs = {s["document"].lower() for s in expected if "document" in s}
            return any(exp in cid.lower() or cid.lower() in exp for exp in exp_docs)

        h_correct = _hit(h_top); r_correct = _hit(r_top)
        if h_top == r_top:
            prov = "unchanged"
        elif not h_correct and r_correct:
            prov = "corrected"
        elif h_correct and not r_correct:
            prov = "degraded"
        else:
            prov = "neutral_change"

        details.append({
            "id": q_id, "type": q_type, "question": query,
            "expected_answer": item["expected_answer"],
            "dense": {"top1": d_top, "metrics": d_m},
            "bm25": {"top1": b_top, "metrics": b_m},
            "hybrid": {"top1": h_top, "metrics": h_m},
            "reranked": {"top1": r_top, "metrics": r_m, "status": rer_info["status"]},
            "reranked_list": [
                {"chunk_id": r.chunk_id,
                 "original_rrf_rank": r.metadata.get("original_rrf_rank"),
                 "final_rank": r.rank,
                 "reranker_score": r.metadata.get("reranker_score")}
                for r in rer_r
            ],
            "provenance": {"rrf_top1": h_top, "reranker_top1": r_top, "class": prov,
                          "rrf_correct": h_correct, "reranker_correct": r_correct},
        })
        print(f"[{idx}/{len(questions)}] {q_id} ({q_type}): "
              f"D@1={d_m['recall@1']:.0f} B@1={b_m['recall@1']:.0f} "
              f"H@1={h_m['recall@1']:.0f} R@1={r_m['recall@1']:.0f} "
              f"rrf={rrf_ms:.2f}ms rer={rer_ms:.2f}ms prov={prov}")

    baseline = build_summary(details, "baseline")
    latency = {
        "dense_retrieval": stat(dense_lat),
        "bm25_retrieval": stat(bm25_lat),
        "rrf_fusion": stat(rrf_lat),
        "hybrid_total_retrieval": stat(hyb_lat),
        "reranker": stat(rer_lat),
    }

    # ---- Candidate depth experiment ----
    cand_exp = candidate_depth_experiment(questions, hybrid, [5, 10, 20], top_final=5)

    # ---- Final top-k experiment ----
    topk_exp = final_topk_experiment(questions, hybrid, cand=20, tops=[3, 5, 10])

    # ---- Provenance / rank-promotion / failure analysis ----
    prov_counts, promo, failures = analyze_behavior(details)
    cat = by_category(details)

    output = {
        "metadata": {
            "evaluation_phase": "Phase 6 - Cross-Encoder Reranking Evaluation",
            "git_tag": "feature/phase-06-reranker",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reranker_model": EVAL_RERANKER_MODEL,
            "methodology": "retrieval-only timing, warm-up pass; reranker consumes RRF pool",
            "cold_reranker_init_ms": cold_init_ms,
            "memory_mb": {"baseline": round(mem_baseline, 1), "after_model_load": round(mem_after_model, 1),
                          "peak_during_rerank": round(mem_peak, 1), "psutil_available": _HAVE_PSUTIL},
            "reranker_statuses": dict(sorted(__import__("collections").Counter(reranker_statuses).items())),
        },
        "baseline_4way": baseline,
        "latency_table": latency,
        "candidate_depth": cand_exp,
        "final_topk": topk_exp,
        "query_type": cat,
        "provenance": prov_counts,
        "rank_promotion": promo,
        "failures": failures,
        "details": details,
    }

    with (RERANKER_RESULTS_DIR / "reranker_baseline.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {RERANKER_RESULTS_DIR / 'reranker_baseline.json'}")

    # Also write the focused experiment files
    with (RERANKER_RESULTS_DIR / "reranker_candidate_depth.json").open("w", encoding="utf-8") as f:
        json.dump(cand_exp, f, indent=2)
    with (RERANKER_RESULTS_DIR / "reranker_top_k.json").open("w", encoding="utf-8") as f:
        json.dump(topk_exp, f, indent=2)
    with (RERANKER_RESULTS_DIR / "reranker_provenance.json").open("w", encoding="utf-8") as f:
        json.dump({"provenance": prov_counts, "rank_promotion": promo, "failures": failures}, f, indent=2)
    with (RERANKER_RESULTS_DIR / "reranker_failures.json").open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    print_summary(baseline, latency, cand_exp, topk_exp, prov_counts, promo, mem_baseline, mem_peak, cat)
    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stat(vals: list[float]) -> dict[str, float]:
    return {"mean": round(sum(vals) / len(vals), 3),
            "p50": round(percentile(vals, 50), 3),
            "p95": round(percentile(vals, 95), 3)}


def _metrics_for(results: list, expected: list or dict) -> dict:
    return calculate_retrieval_metrics(to_metric_format(results), expected)


def candidate_depth_experiment(questions, hybrid, depths, top_final) -> dict:
    out = {}
    for ck in depths:
        rr = CrossEncoderReranker(model_name=EVAL_RERANKER_MODEL, candidate_k=ck, top_k=top_final)
        sys_metrics = {"recall@1": [], "recall@5": [], "recall@10": [], "mrr": []}
        lat = []
        for item in questions:
            q = item["question"]; exp = item.get("expected_sources", [])
            hyb = hybrid.search(q, top_k_dense=10, top_k_sparse=10, top_k_fused=ck)
            t0 = time.perf_counter()
            rer, info = rr.rerank_with_timing(q, hyb, top_k=top_final)
            lat.append((time.perf_counter() - t0) * 1000)
            m = calculate_retrieval_metrics(to_metric_format(rer), exp)
            for k in sys_metrics:
                sys_metrics[k].append(m[k])
        out[f"cand{ck}_top{top_final}"] = {
            "candidate_k": ck, "top_k_final": top_final,
            "metrics": {k: round(sum(v) / len(v), 4) for k, v in sys_metrics.items()},
            "latency": stat(lat),
        }
    return out


def final_topk_experiment(questions, hybrid, cand, tops) -> dict:
    out = {}
    for tk in tops:
        rr = CrossEncoderReranker(model_name=EVAL_RERANKER_MODEL, candidate_k=cand, top_k=tk)
        sys_metrics = {"recall@1": [], "recall@5": [], "mrr": []}
        lat = []
        for item in questions:
            q = item["question"]; exp = item.get("expected_sources", [])
            hyb = hybrid.search(q, top_k_dense=10, top_k_sparse=10, top_k_fused=cand)
            t0 = time.perf_counter()
            rer, info = rr.rerank_with_timing(q, hyb, top_k=tk)
            lat.append((time.perf_counter() - t0) * 1000)
            m = calculate_retrieval_metrics(to_metric_format(rer), exp)
            for k in sys_metrics:
                sys_metrics[k].append(m[k])
        out[f"cand{cand}_top{tk}"] = {
            "candidate_k": cand, "top_k_final": tk,
            "metrics": {k: round(sum(v) / len(v), 4) for k, v in sys_metrics.items()},
            "latency": stat(lat),
        }
    return out


def build_summary(details, name) -> dict:
    def agg(field):
        systems = {}
        for sys_name in ("dense", "bm25", "hybrid", "reranked"):
            vals = [d[sys_name]["metrics"][field] for d in details]
            systems[sys_name] = round(sum(vals) / len(vals), 4)
        return systems
    return {
        "recall@1": agg("recall@1"),
        "recall@5": agg("recall@5"),
        "recall@10": agg("recall@10"),
        "mrr": agg("mrr"),
    }


def by_category(details) -> dict:
    cats: dict[str, list] = {}
    for d in details:
        cats.setdefault(d["type"], []).append(d)
    out = {}
    for cat, items in cats.items():
        n = len(items)
        out[cat] = {
            "count": n,
            "hybrid": {m: round(sum(i["hybrid"]["metrics"][m] for i in items) / n, 4)
                       for m in ("recall@1", "recall@5", "mrr")},
            "reranked": {m: round(sum(i["reranked"]["metrics"][m] for i in items) / n, 4)
                         for m in ("recall@1", "recall@5", "mrr")},
        }
    return out


def analyze_behavior(details) -> tuple[dict, dict, dict]:
    prov_counts: dict[str, int] = {}
    promoted = demoted = unchanged = 0
    failures = {"corrected": [], "degraded": []}
    for d in details:
        cls = d["provenance"]["class"]
        prov_counts[cls] = prov_counts.get(cls, 0) + 1
        # Rank promotion: each reranked final vs its original RRF rank.
        for r in d.get("reranked_list", []):
            o = r.get("original_rrf_rank")
            f = r.get("final_rank")
            if o is None or f is None:
                continue
            delta = o - f
            if delta > 0:
                promoted += 1
            elif delta < 0:
                demoted += 1
            else:
                unchanged += 1
    promo = {"promoted": promoted, "demoted": demoted, "unchanged": unchanged}
    for d in details:
        if d["provenance"]["class"] == "corrected":
            failures["corrected"].append({"id": d["id"], "type": d["type"],
                                          "rrf_top1": d["provenance"]["rrf_top1"],
                                          "reranker_top1": d["provenance"]["reranker_top1"]})
        elif d["provenance"]["class"] == "degraded":
            failures["degraded"].append({"id": d["id"], "type": d["type"],
                                         "rrf_top1": d["provenance"]["rrf_top1"],
                                         "reranker_top1": d["provenance"]["reranker_top1"]})
    return prov_counts, promo, failures


def print_summary(baseline, latency, cand_exp, topk_exp, prov, promo, mem_base, mem_peak, cat):
    print("\n=== BASELINE 4-WAY ===")
    for metric in ("recall@1", "recall@5", "recall@10", "mrr"):
        r = baseline[metric]
        print(f"  {metric:10s} D={r['dense']:.4f} B={r['bm25']:.4f} H={r['hybrid']:.4f} R={r['reranked']:.4f}")
    print("\n=== LATENCY (ms) ===")
    for k, v in latency.items():
        print(f"  {k:24s} mean={v['mean']:.3f} p50={v['p50']:.3f} p95={v['p95']:.3f}")
    print("\n=== CANDIDATE DEPTH (top5) ===")
    for k, v in cand_exp.items():
        print(f"  {k}: R@1={v['metrics']['recall@1']:.4f} R@5={v['metrics']['recall@5']:.4f} R@10={v['metrics']['recall@10']:.4f} MRR={v['metrics']['mrr']:.4f} rer={v['latency']['mean']:.2f}ms")
    print("\n=== FINAL TOP-K (cand20) ===")
    for k, v in topk_exp.items():
        print(f"  {k}: R@1={v['metrics']['recall@1']:.4f} R@5={v['metrics']['recall@5']:.4f} MRR={v['metrics']['mrr']:.4f} rer={v['latency']['mean']:.2f}ms")
    print("\n=== QUERY-TYPE (Hybrid RRF vs Reranked) ===")
    for cat_name, c in cat.items():
        print(f"  {cat_name:10s} n={c['count']:2d}  H@1={c['hybrid']['recall@1']:.4f}->R@1={c['reranked']['recall@1']:.4f}"
              f"  H@5={c['hybrid']['recall@5']:.4f}->R@5={c['reranked']['recall@5']:.4f}"
              f"  H_mrr={c['hybrid']['mrr']:.4f}->R_mrr={c['reranked']['mrr']:.4f}")
    print(f"\nProvenance: {prov}")
    print(f"Rank promotion (across reranked finals): {promo}")
    print(f"Memory baseline={mem_base:.1f}MB peak={mem_peak:.1f}MB")


if __name__ == "__main__":
    run_evaluation()
