"""Isolated reranker latency benchmark (Phase 6)."""
from time import perf_counter
from backend.reranker import CrossEncoderReranker
from backend.retrieval_hybrid import get_hybrid_retriever

hybrid = get_hybrid_retriever()
qs = [
    "What was the effective date of the 2024 Employee Handbook?",
    "Summarize the remote work policy.",
    "What are the eligibility requirements for the performance bonus?",
    "How does the company handle confidential information?",
    "What is the procedure for requesting leave of absence?",
]
hybs = [hybrid.search(q, top_k_dense=10, top_k_sparse=10, top_k_fused=20) for q in qs]

for model in ["ms-marco-MiniLM-L-12-v2", "ms-marco-TinyBERT-L-2-v2"]:
    rr = CrossEncoderReranker(model_name=model, candidate_k=20, top_k=10)
    rr.rerank(qs[0], hybs[0], top_k=10)  # warm
    times = []
    for q, h in zip(qs, hybs):
        t0 = perf_counter()
        rr.rerank(q, h, top_k=10)
        times.append((perf_counter() - t0) * 1000)
    print(f"{model}: cold={rr.cold_init_ms:.1f}ms  warm_rerank_mean={sum(times)/len(times):.1f}ms "
          f"min={min(times):.1f}ms max={max(times):.1f}ms")
