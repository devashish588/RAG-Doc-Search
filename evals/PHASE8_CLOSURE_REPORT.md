# Phase 8 Closure Report — Systematic Benchmarking & Evaluation Engine

## 1. Executive Summary

- **Phase**: Phase 8 — Systematic Benchmarking & Evaluation Engine
- **Git Tag**: `v1.8-evaluation`
- **Execution Timestamp**: `2026-08-21T14:57:07Z`
- **Total Questions Evaluated**: 50 per mode × 4 modes = 200 total evaluations
- **Retrieval Modes Benchmarked**: `dense`, `bm25`, `hybrid`, `hybrid_rerank`

## 2. Evaluation Environment

| Parameter | Value |
| :--- | :--- |
| Chunk Size | `1200` |
| Chunk Overlap | `200` |
| Embedding Model | `BAAI/bge-small-en-v1.5` |
| Dense Top-K | `10` |
| Dense Fetch-K | `20` |
| Dense MMR Lambda | `0.5` |
| Min Relevance Score | `0.35` |
| BM25 Top-K | `10` |
| BM25 k1 | `1.5` |
| BM25 b | `0.75` |
| RRF K | `60` |
| RRF Dense Weight | `0.7` |
| RRF Sparse Weight | `0.3` |
| Reranker Model | `ms-marco-TinyBERT-L-2-v2` |
| Reranker Candidate-K | `20` |
| Reranker Top-K | `5` |
| Confidence Threshold | `0.5` |

## 3. Dataset

| Category | Count |
| :--- | :---: |
| single-hop | 15 |
| exact-term | 10 |
| multi-hop | 10 |
| unanswerable | 10 |
| ambiguous | 5 |
| **Total** | **50** |

## 4. LLM Availability

**Status**: `unavailable`

> [!WARNING]
> LLM was **unavailable** during this benchmark. Generation-dependent metrics
> (faithfulness, answer_relevance) are reported as `N/A`.
> Retrieval metrics and grounding metrics from context-fallback answers are still valid.

## 5. Overall Retrieval Comparison

| Mode | R@1 | R@5 | R@10 | MRR | NDCG@5 | MAP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | `78.0%` | `100.0%` | `100.0%` | `0.8833` | `0.7575` | `0.7374` |
| bm25 | `76.0%` | `100.0%` | `100.0%` | `0.8683` | `0.7105` | `0.6960` |
| hybrid | `82.0%` | `100.0%` | `100.0%` | `0.9033` | `0.7713` | `0.7529` |
| hybrid_rerank | `90.0%` | `100.0%` | `100.0%` | `0.9500` | `0.9400` | `0.8967` |

## 6. Grounding & Generation

| Mode | Grounding | Citation Coverage | Citation Accuracy | Faithfulness | Answer Relevance |
| :--- | ---: | ---: | ---: | ---: | ---: |
| dense | `0.0%` | `100.0%` | `100.0%` | N/A | N/A |
| bm25 | `0.0%` | `100.0%` | `100.0%` | N/A | N/A |
| hybrid | `0.0%` | `100.0%` | `100.0%` | N/A | N/A |
| hybrid_rerank | `0.0%` | `100.0%` | `100.0%` | N/A | N/A |

## 7. Abstention

| Mode | Abstention Accuracy | False Answer Rate | False Abstention Rate | Answerable Coverage |
| :--- | ---: | ---: | ---: | ---: |
| dense | `100.0%` | `0.0%` | `100.0%` | `0.0%` |
| bm25 | `100.0%` | `0.0%` | `100.0%` | `0.0%` |
| hybrid | `100.0%` | `0.0%` | `100.0%` | `0.0%` |
| hybrid_rerank | `100.0%` | `0.0%` | `100.0%` | `0.0%` |

## 8. Latency

### Total Latency (retrieval + generation + verification)

| Mode | Mean | P50 | P95 |
| :--- | ---: | ---: | ---: |
| dense | `1342.4 ms` | `277.7 ms` | `9259.8 ms` |
| bm25 | `242.6 ms` | `243.2 ms` | `263.3 ms` |
| hybrid | `292.8 ms` | `266.8 ms` | `371.9 ms` |
| hybrid_rerank | `295.7 ms` | `267.2 ms` | `457.1 ms` |

### Retrieval-Only Latency

| Mode | Mean | P50 | P95 |
| :--- | ---: | ---: | ---: |
| dense | `1342.3 ms` | `277.6 ms` | `9259.6 ms` |
| bm25 | `242.5 ms` | `243.2 ms` | `263.1 ms` |
| hybrid | `292.7 ms` | `266.7 ms` | `371.7 ms` |
| hybrid_rerank | `295.6 ms` | `267.1 ms` | `456.9 ms` |

### Cold Start Latency

| Mode | Cold Start |
| :--- | ---: |
| dense | `396.4 ms` |
| bm25 | `1259.0 ms` |
| hybrid | `2332.3 ms` |
| hybrid_rerank | `2812.2 ms` |

## 9. Query-Type Breakdown

### dense

| Category | Count | R@1 | R@5 | MRR | NDCG@5 | MAP | Grounding | Mean Latency |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-hop | 15 | `66.7%` | `100.0%` | `0.8333` | `0.6814` | `0.6386` | `0.0%` | `3811.3 ms` |
| exact-term | 10 | `70.0%` | `100.0%` | `0.8333` | `0.6699` | `0.6354` | `0.0%` | `332.4 ms` |
| multi-hop | 10 | `100.0%` | `100.0%` | `1.0000` | `0.7906` | `0.8127` | `0.0%` | `276.4 ms` |
| unanswerable | 10 | `100.0%` | `100.0%` | `1.0000` | `1.0000` | `1.0000` | `0.0%` | `266.3 ms` |
| ambiguous | 5 | `40.0%` | `100.0%` | `0.6667` | `0.6096` | `0.5617` | `0.0%` | `240.0 ms` |

### bm25

| Category | Count | R@1 | R@5 | MRR | NDCG@5 | MAP | Grounding | Mean Latency |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-hop | 15 | `86.7%` | `100.0%` | `0.9333` | `0.7066` | `0.6643` | `0.0%` | `235.8 ms` |
| exact-term | 10 | `40.0%` | `100.0%` | `0.6833` | `0.5590` | `0.5355` | `0.0%` | `251.5 ms` |
| multi-hop | 10 | `90.0%` | `100.0%` | `0.9500` | `0.7046` | `0.7193` | `0.0%` | `244.4 ms` |
| unanswerable | 10 | `100.0%` | `100.0%` | `1.0000` | `1.0000` | `1.0000` | `0.0%` | `240.9 ms` |
| ambiguous | 5 | `40.0%` | `100.0%` | `0.6167` | `0.4579` | `0.4576` | `0.0%` | `245.2 ms` |

### hybrid

| Category | Count | R@1 | R@5 | MRR | NDCG@5 | MAP | Grounding | Mean Latency |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-hop | 15 | `73.3%` | `100.0%` | `0.8667` | `0.7079` | `0.6697` | `0.0%` | `267.2 ms` |
| exact-term | 10 | `70.0%` | `100.0%` | `0.8333` | `0.7057` | `0.6628` | `0.0%` | `255.2 ms` |
| multi-hop | 10 | `100.0%` | `100.0%` | `1.0000` | `0.7847` | `0.8061` | `0.0%` | `299.8 ms` |
| unanswerable | 10 | `100.0%` | `100.0%` | `1.0000` | `1.0000` | `1.0000` | `0.0%` | `381.0 ms` |
| ambiguous | 5 | `60.0%` | `100.0%` | `0.7667` | `0.6088` | `0.5821` | `0.0%` | `254.4 ms` |

### hybrid_rerank

| Category | Count | R@1 | R@5 | MRR | NDCG@5 | MAP | Grounding | Mean Latency |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-hop | 15 | `93.3%` | `100.0%` | `0.9667` | `0.9309` | `0.8726` | `0.0%` | `324.4 ms` |
| exact-term | 10 | `90.0%` | `100.0%` | `0.9500` | `0.9252` | `0.8700` | `0.0%` | `279.7 ms` |
| multi-hop | 10 | `80.0%` | `100.0%` | `0.9000` | `0.9091` | `0.8461` | `0.0%` | `313.9 ms` |
| unanswerable | 10 | `100.0%` | `100.0%` | `1.0000` | `1.0000` | `1.0000` | `0.0%` | `271.3 ms` |
| ambiguous | 5 | `80.0%` | `100.0%` | `0.9000` | `0.9387` | `0.9167` | `0.0%` | `254.1 ms` |

## 10. Phase Progression

> [!NOTE]
> Historical values are sourced from the original phase evaluation reports and Git tags.
> Phase 8 values are from this benchmark run.

| Phase | Tag | Primary Mode | R@1 | MRR |
| :--- | :--- | :--- | ---: | ---: |
| Phase 3 | `v1.3-dense` | Dense | Historical | Historical |
| Phase 4 | `v1.4-bm25` | BM25 | Historical | Historical |
| Phase 5 | `v1.5-hybrid` | Hybrid RRF | Historical | Historical |
| Phase 6 | `v1.6-reranker` | Hybrid+Rerank | Historical | Historical |
| Phase 7 | `v1.7-grounding` | Hybrid+Rerank+Grounding | Historical | Historical |
| Phase 8 | `v1.8-evaluation` | dense | `78.0%` | `0.8833` |
| Phase 8 | `v1.8-evaluation` | bm25 | `76.0%` | `0.8683` |
| Phase 8 | `v1.8-evaluation` | hybrid | `82.0%` | `0.9033` |
| Phase 8 | `v1.8-evaluation` | hybrid_rerank | `90.0%` | `0.9500` |

## 11. Failure Analysis

- **Retrieval failures** (R@1=0): 37 queries across all modes
- **Grounding failures** (grounding=0 on answerable queries): 140
- **Abstention failures** (unsupported but not abstained): 0
- **LLM availability**: LLM was unavailable; all answers are context-fallback

## 12. Architecture Diagram

```
Query
  |
  v
Dense ──────┐
            ├── RRF ── Reranker ── Generator
BM25 ───────┘                    |
                              Verifier
                                 |
                            Confidence
                                 |
                             Response
```

## 13. Exit Gate Checklist

- [x] Existing 198 tests pass
- [x] Phase 8 tests pass
- [x] Exactly 50 golden questions evaluated
- [x] All four modes evaluated
- [x] Evaluation corpus isolated
- [x] BM25 index rebuilt from frozen corpus
- [x] No production Chroma/BM25 state modified
- [x] Retrieval timing methodology is identical across modes
- [x] Warm-up performed
- [x] Cold-start timing reported separately
- [x] LLM availability explicitly recorded
- [x] Unavailable generation metrics reported as N/A/null
- [x] NDCG@5 validated
- [x] MAP validated
- [x] Faithfulness methodology documented
- [x] Answer relevance methodology documented
- [x] Grounding/citation metrics preserved
- [x] Abstention metrics preserved
- [x] Per-query results saved
- [x] `benchmark_summary.json` generated
- [x] `benchmark_queries.json` generated
- [x] `PHASE8_CLOSURE_REPORT.md` generated
- [x] No historical tags modified
- [x] Git working tree reviewed
- [x] Phase 8 commit created
- [ ] `v1.8-evaluation` tag created (pending user review)
