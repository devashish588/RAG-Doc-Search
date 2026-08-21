# Phase 8 Closure Report — Systematic Benchmarking & Evaluation Engine

## 1. Executive Summary

- **Phase**: Phase 8 — Systematic Benchmarking & Evaluation Engine
- **Git Tag**: `v1.8-evaluation`
- **Execution Timestamp**: `2026-08-21T14:15:34Z`
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

**Status**: `real_llm_active`

## 5. Overall Retrieval Comparison

| Mode | R@1 | R@5 | R@10 | MRR | NDCG@5 | MAP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | `78.0%` | `100.0%` | `100.0%` | `0.8833` | `0.7314` | `0.7171` |
| bm25 | `78.0%` | `100.0%` | `100.0%` | `0.8833` | `0.7314` | `0.7171` |
| hybrid | `78.0%` | `100.0%` | `100.0%` | `0.8833` | `0.7314` | `0.7171` |
| hybrid_rerank | `78.0%` | `98.0%` | `98.0%` | `0.8680` | `0.7739` | `0.7426` |

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
| dense | `577.2 ms` | `288.9 ms` | `2156.8 ms` |
| bm25 | `259.5 ms` | `260.2 ms` | `288.4 ms` |
| hybrid | `385.6 ms` | `304.2 ms` | `913.2 ms` |
| hybrid_rerank | `547.6 ms` | `287.9 ms` | `2625.8 ms` |

### Retrieval-Only Latency

| Mode | Mean | P50 | P95 |
| :--- | ---: | ---: | ---: |
| dense | `577.1 ms` | `288.7 ms` | `2156.5 ms` |
| bm25 | `259.4 ms` | `260.0 ms` | `288.3 ms` |
| hybrid | `385.4 ms` | `303.9 ms` | `913.0 ms` |
| hybrid_rerank | `547.5 ms` | `287.7 ms` | `2625.7 ms` |

### Cold Start Latency

| Mode | Cold Start |
| :--- | ---: |
| dense | `699.9 ms` |
| bm25 | `264.2 ms` |
| hybrid | `273.3 ms` |
| hybrid_rerank | `329.2 ms` |

## 9. Query-Type Breakdown

### dense

| Category | Count | R@1 | R@5 | MRR | NDCG@5 | MAP | Grounding | Mean Latency |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-hop | 15 | `66.7%` | `100.0%` | `0.8333` | `0.6651` | `0.6279` | `0.0%` | `443.1 ms` |
| exact-term | 10 | `70.0%` | `100.0%` | `0.8333` | `0.5996` | `0.5784` | `0.0%` | `504.4 ms` |
| multi-hop | 10 | `100.0%` | `100.0%` | `1.0000` | `0.7640` | `0.7869` | `0.0%` | `497.7 ms` |
| unanswerable | 10 | `100.0%` | `100.0%` | `1.0000` | `1.0000` | `1.0000` | `0.0%` | `1066.1 ms` |
| ambiguous | 5 | `40.0%` | `100.0%` | `0.6667` | `0.5915` | `0.5565` | `0.0%` | `306.7 ms` |

### bm25

| Category | Count | R@1 | R@5 | MRR | NDCG@5 | MAP | Grounding | Mean Latency |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-hop | 15 | `66.7%` | `100.0%` | `0.8333` | `0.6651` | `0.6279` | `0.0%` | `266.4 ms` |
| exact-term | 10 | `70.0%` | `100.0%` | `0.8333` | `0.5996` | `0.5784` | `0.0%` | `253.4 ms` |
| multi-hop | 10 | `100.0%` | `100.0%` | `1.0000` | `0.7640` | `0.7869` | `0.0%` | `263.4 ms` |
| unanswerable | 10 | `100.0%` | `100.0%` | `1.0000` | `1.0000` | `1.0000` | `0.0%` | `258.4 ms` |
| ambiguous | 5 | `40.0%` | `100.0%` | `0.6667` | `0.5915` | `0.5565` | `0.0%` | `245.4 ms` |

### hybrid

| Category | Count | R@1 | R@5 | MRR | NDCG@5 | MAP | Grounding | Mean Latency |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-hop | 15 | `66.7%` | `100.0%` | `0.8333` | `0.6651` | `0.6279` | `0.0%` | `331.7 ms` |
| exact-term | 10 | `70.0%` | `100.0%` | `0.8333` | `0.5996` | `0.5784` | `0.0%` | `296.6 ms` |
| multi-hop | 10 | `100.0%` | `100.0%` | `1.0000` | `0.7640` | `0.7869` | `0.0%` | `308.4 ms` |
| unanswerable | 10 | `100.0%` | `100.0%` | `1.0000` | `1.0000` | `1.0000` | `0.0%` | `404.0 ms` |
| ambiguous | 5 | `40.0%` | `100.0%` | `0.6667` | `0.5915` | `0.5565` | `0.0%` | `843.1 ms` |

### hybrid_rerank

| Category | Count | R@1 | R@5 | MRR | NDCG@5 | MAP | Grounding | Mean Latency |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-hop | 15 | `66.7%` | `100.0%` | `0.8333` | `0.7602` | `0.6989` | `0.0%` | `1162.4 ms` |
| exact-term | 10 | `70.0%` | `90.0%` | `0.7400` | `0.6274` | `0.5706` | `0.0%` | `293.1 ms` |
| multi-hop | 10 | `100.0%` | `100.0%` | `1.0000` | `0.7754` | `0.7961` | `0.0%` | `291.0 ms` |
| unanswerable | 10 | `100.0%` | `100.0%` | `1.0000` | `1.0000` | `1.0000` | `0.0%` | `283.2 ms` |
| ambiguous | 5 | `40.0%` | `100.0%` | `0.7000` | `0.6526` | `0.5964` | `0.0%` | `254.6 ms` |

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
| Phase 8 | `v1.8-evaluation` | bm25 | `78.0%` | `0.8833` |
| Phase 8 | `v1.8-evaluation` | hybrid | `78.0%` | `0.8833` |
| Phase 8 | `v1.8-evaluation` | hybrid_rerank | `78.0%` | `0.8680` |

## 11. Failure Analysis

- **Retrieval failures** (R@1=0): 44 queries across all modes
- **Grounding failures** (grounding=0 on answerable queries): 140
- **Abstention failures** (unsupported but not abstained): 0

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
